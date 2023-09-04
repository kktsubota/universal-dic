import argparse
from pathlib import Path
import re
import subprocess

import numpy as np
import pandas as pd
import torch
from torchvision import transforms

import dataset
from main import QuantizedModelWrapper
from aof.ops.cdf import SpikeAndSlabCDF, LogisticCDF
from aof.entropy_models.weight_entropy_module import WeightEntropyModule
from aof.models.utils import get_model
from aof.models import Cheng2020AttnAdapter, WACNNAdapter
from aof.utils.image import read_image, crop, pad


torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="wacnn", choices={"wacnn", "cheng2020-attn"})
    parser.add_argument("--weight_root", type=Path, default="results/ours")
    parser.add_argument("--rozendaal", action="store_true", default=False)
    parser.add_argument(
        "--domain",
        default="vector",
        choices=dataset.root_dict.keys(),
    )
    parser.add_argument("--stage", default="2nd")
    parser.add_argument("--data_type", default="uint8")
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--width", type=float, default=0.06)
    parser.add_argument("--sigma", type=float, default=0.05)
    parser.add_argument("--distrib", default="logistic")
    parser.add_argument("--n-dim-1", type=int, default=0)
    parser.add_argument("--n-dim-2", type=int, default=2)
    parser.add_argument("--groups", type=int, default=1)
    parser.add_argument("--position", type=str, default="last")
    parser.add_argument("--pipeline", default="default", choices={"default", "swap", "end2end"})
    parser.add_argument("--regex", default="g_s\.5\.adapter.*")
    parser.add_argument("--save-image", action="store_true", default=False)
    args = parser.parse_args()

    paths = dataset.get_paths(args.domain)

    cache_root: Path = Path("cache")
    weight_root = args.weight_root / args.model

    if args.rozendaal:
        args.width = 0.005
        args.sigma = 0.05
        args.alpha = 1000
        args.distrib = "spike-and-slab"
        args.regex = ".*"
        args.stage = "2nd"
        args.pipeline = "end2end"

    device = "cuda"
    for quality in range(1, 7):
        model = get_model(args.model, quality)
        if args.distrib == "spike-and-slab":
            distrib = SpikeAndSlabCDF(args.width, args.sigma, args.alpha)
        elif args.distrib == "logistic":
            distrib = LogisticCDF(scale=args.sigma)
        went = WeightEntropyModule(
            distrib,
            args.width,
            data_type="float32" if args.data_type == "float64+7z" else args.data_type,
        )

        # model with adapters
        if args.stage == "2nd" and not args.rozendaal:
            state_dict = model.state_dict()
            if args.model == "wacnn":
                model = WACNNAdapter(
                    model.N,
                    model.M,
                    args.n_dim_1,
                    args.n_dim_2,
                    args.groups,
                    position=args.position,
                )
            elif args.model == "cheng2020-attn":
                assert args.regex == "g_s\.[8-8]\.adapter_1.*"
                assert args.n_dim_2 == 0
                n_dim_1 = [0, 0, 0, 0, 0, 0, args.n_dim_1]
                n_dim_2 = [0, 0, 0, 0, 0, 0, args.n_dim_2]
                model = Cheng2020AttnAdapter(
                    model.N,
                    n_dim_1,
                    n_dim_2,
                    args.groups,
                    connection="serial",
                )
            else:
                raise NotImplementedError

            info = model.load_state_dict(state_dict, strict=False)
            print(info)

        model.to(device)
        went.to(device)

        if args.stage == "2nd":
            model_qua = QuantizedModelWrapper(model, went, regex=args.regex)
            with torch.no_grad():
                model_qua.update_parameters(model)

            for key in model_qua.params_init.keys():
                if "adapter" in key:
                    model_qua.params_init[key].fill_(0)
        else:
            model_qua = QuantizedModelWrapper(model, went, regex="none")

        model_qua.update_ent(force=True)
        model_qua.eval()
        model_qua.report_params()

        score_dict = dict(bpp=list(), psnr=list())
        for path in paths:
            data_path: Path = weight_root / f"q{quality}" / args.domain / path.name
            img = read_image(path)
            transform = transforms.ToTensor()
            x: torch.Tensor = transform(img)[None]
            x = x.cuda()
            n_pixels = x.shape[2] * x.shape[3]

            # decode
            if args.pipeline == "swap":
                cache_path: Path = data_path / "compressed.pt"
            elif args.pipeline == "end2end":
                cache_path: Path = data_path / "latent.pt"
            else:
                cache_path: Path = (
                    cache_root / f"{path.name}-{quality}-{args.model}-0.001-2000.pt"
                )
            weight_path = data_path / "weights.pt"
            if args.stage in {"1st", "2nd"}:
                compressed = torch.load(cache_path)
            else:
                x_pad = pad(x)
                # NOTE: entropy model is updated in model_qua.__init__
                compressed = model_qua.model.compress(x_pad)
            if args.stage == "2nd":
                compressed["weights"] = torch.load(weight_path)
            else:
                compressed["weights"] = dict()
            out_dict = model_qua.decompress(**compressed)
            x_hat = crop(out_dict["x_hat"], x.shape[2:])
            # following conventional codecs
            x_hat = torch.round(x_hat * 255) / 255
            if args.save_image:
                transforms.ToPILImage()(x_hat[0]).save(data_path / f"{args.stage}.png")

            # evaluate
            bpp_c = (
                sum(len(string[0]) for string in compressed["strings"]) * 8 / n_pixels
            )
            if args.stage == "2nd":
                if args.data_type == "float64+7z":
                    arrs = list()
                    state_dict = model_qua.model.state_dict()

                    for key in state_dict.keys():
                        if re.match(args.regex, key):
                            arrs.append(state_dict[key].cpu().numpy().flatten())
                    arrs = np.concatenate(arrs)
                    assert arrs.shape == (9283,)
                    path_bin = (
                        weight_root
                        / f"q{quality}"
                        / args.domain
                        / path.name
                        / "weights.bin"
                    )
                    with (path_bin).open("wb") as f:
                        f.write(arrs.astype(np.float64).tobytes())

                    path_7z = path_bin.with_suffix(".7z")
                    subprocess.run(
                        f"7z a -t7z -m0=lzma -mx=9 -mfb=64 -md=32m -ms=on {path_7z} {path_bin}",
                        shell=True,
                    )
                    bpp_m = path_7z.stat().st_size * 8 / n_pixels

                else:
                    bpp_m = (
                        sum(len(weight[0]) for weight in compressed["weights"].values())
                        * 8
                        / n_pixels
                    )
            else:
                bpp_m = 0
            # height of the input image (uint16), width of the input image (uint16), quality (4 bits)
            # compressed["shape"] can be estimated from (height, width) of the input image
            bpp_meta = (16 * 2 + 4) / n_pixels

            mse: torch.Tensor = (x - x_hat).square().mean(dim=(1, 2, 3))
            psnr: torch.Tensor = -10 * mse.log10().mean()
            assert not torch.isnan(mse), path
            score_dict["psnr"].append(psnr.item())
            score_dict["bpp"].append(bpp_c + bpp_m + bpp_meta)

        df = pd.DataFrame.from_dict(score_dict)
        df.index = paths

        csv_path = f"{args.domain}_{args.stage}.csv"
        df.to_csv(weight_root / f"q{quality}" / csv_path)

        del model_qua
        del model
        del went


if __name__ == "__main__":
    main()
