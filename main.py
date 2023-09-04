import argparse
import copy
import logging
import operator
from pathlib import Path
import re
import time

import numpy as np
from PIL import Image
import torch
from torch import nn
from torchvision import transforms

from aof.ops.cdf import LogisticCDF, SpikeAndSlabCDF
from aof.ops.quantize import quantize_sga
from aof.entropy_models.weight_entropy_module import WeightEntropyModule
from aof.training import configure_optimizers
from aof.training.losses import RateDistortionModelLoss
from aof.models.utils import (
    get_model,
    forward_enc,
    forward_dec,
    decode_latent,
    encode_latent,
)
from aof.models import Cheng2020AttnAdapter, WACNNAdapter
from aof.utils.image import read_image, pad, crop

torch.backends.cudnn.benchmark = True
# inference flag
CUDNN_INFERENCE_FLAGS = {"benchmark": False, "deterministic": True, "enabled": True}


class QuantizedModelWrapper:
    """Wrapper for Quantized Model"""

    def __init__(self, model, w_ent: WeightEntropyModule, regex: str) -> None:
        # regex: regex of keys to update
        self.model = copy.deepcopy(model)
        for p in self.model.parameters():
            p.requires_grad = False

        self.w_ent = w_ent
        self.regex = regex

        # self.params_init is defined here.
        self._register_params_init(model)

        # self.training is defined here.
        self.train()

    def train(self) -> None:
        self.training = True
        self.model.train()
        self.w_ent.train()

    def eval(self) -> None:
        self.training = False
        self.model.eval()
        self.w_ent.eval()

    def _register_params_init(self, model) -> None:
        params_dict = dict()
        for name, p in model.named_parameters():
            # encoder
            if name.startswith("g_a"):
                continue
            # hyper encoder
            if name.startswith("h_a"):
                continue

            params_dict[name] = p

        self.params_init: dict = copy.deepcopy(params_dict)
        for p in self.params_init.values():
            p.requires_grad = False

    def report_params(self):
        n_params_total: int = 0
        n_params_update: int = 0
        for key, p in self.params_init.items():
            n_param = np.prod(p.shape)
            n_params_total += n_param

            if re.match(self.regex, key) is not None:
                n_params_update += n_param
                print(key, n_param)

        print(f"#updating params/#total params: {n_params_update}/{n_params_total}")

    def __call__(self, x_pad: torch.Tensor, shape=None) -> dict:
        raise NotImplementedError("Please use models.helper instead.")

    def eval_enc(self):
        self.model.g_a.eval()
        self.model.h_a.eval()
        self.model.entropy_bottleneck.eval()
        self.model.gaussian_conditional.eval()

    def to(self, device):
        self.model.to(device)
        self.w_ent.to(device)
        for key in self.params_init.keys():
            self.params_init[key] = self.params_init[key].to(device)

    @torch.no_grad()
    def compress(self, x_pad: torch.Tensor, y=None, z=None) -> dict:
        if y is not None and z is not None:
            compressed = encode_latent(self, y, z)
        else:
            compressed = self.model.compress(x_pad)
        compressed["weights"] = self.compress_weight()
        return compressed

    @torch.no_grad()
    def compress_weight(self) -> dict:
        weights = dict()
        for key, p_init in self.params_init.items():
            if re.match(self.regex, key) is not None:
                getter = operator.attrgetter(key)
                p_qua = getter(self.model)

                w_shape = p_init.reshape(1, 1, -1).shape
                diff = (p_qua - p_init).reshape(w_shape)
                weight = self.w_ent.compress(diff)
                weights[key] = weight
        return weights

    @torch.no_grad()
    def decompress(self, strings, shape, weights) -> dict:
        self.decompress_weight(weights)
        # out_dict has "x_hat" as a key.
        out_dict = self.model.decompress(strings, shape)
        return out_dict

    @torch.no_grad()
    def decompress_weight(self, weights: dict) -> None:
        for key, p_init in self.params_init.items():
            getter = operator.attrgetter(key)
            p_qua = getter(self.model)

            if key in weights.keys():
                weight = weights[key]
                diff = self.w_ent.decompress(weight, (p_init.numel(),))
                p_qua.copy_(p_init + diff.reshape(p_init.shape))

            else:
                p_qua.copy_(p_init)

    def update_parameters(self, model) -> dict:
        """update model_qua parameters

        Args:
            model (CompressionModel): non-quantized model

        Returns:
            dict: m_likelihoods
        """
        # replace encoder params with the model one
        for p, p_qua in zip(model.parameters(), self.model.parameters()):
            p_qua.detach_()
            p_qua.copy_(p)

        # replace decoder params with the quantized one
        m_likelihoods = dict()
        for key, p_init in self.params_init.items():
            getter = operator.attrgetter(key)
            p = getter(model)
            p_qua = getter(self.model)

            # p_qua = p_init
            if re.match(self.regex, key) is None:
                p_qua.detach_()
                p_qua.copy_(p_init)
                m_likelihoods[key] = None

            # p_qua = p_init + diff_qua
            else:
                diff = p - p_init
                diff_qua, likelihood = self.w_ent(diff.reshape(1, 1, -1))
                p_new = p_init + diff_qua.reshape(diff.shape)
                p_qua.detach_()
                p_qua.copy_(p_new)
                m_likelihoods[key] = likelihood

        return m_likelihoods

    def update_ent(self, force: bool = False):
        self.model.update(force=force)
        self.w_ent.update(force=force)
        device = next(self.model.parameters()).device
        self.w_ent.to(device)


@torch.no_grad()
def test(
    model,
    x: torch.Tensor,
    m_likelihoods=None,
    actual: bool = False,
    y=None,
    z=None,
    y_hat=None,
    strings=None,
    shape=None,
) -> tuple:
    assert not model.training
    height, width = x.shape[2:]
    n_pixels: int = height * width
    x_ = pad(x)
    if actual:
        with torch.backends.cudnn.flags(**CUDNN_INFERENCE_FLAGS):
            # rename the variable
            assert isinstance(model, QuantizedModelWrapper)
            model_qua = model

            if strings is not None and shape is not None:
                c = {
                    "strings": strings,
                    "shape": shape,
                    "weights": model_qua.compress_weight(),
                }
            elif y is not None and z is not None:
                # x_ is not accessed
                c = model_qua.compress(x_, y, z)
            else:
                c = model_qua.compress(x_)
            out_dict = model_qua.decompress(**c)
            x_hat = crop(out_dict["x_hat"], x.shape[2:])
            x_hat = x_hat.mul(255).round().div(255)
            bpp = sum(len(string[0]) for string in c["strings"]) * 8 / n_pixels
            if "weights" in c:
                bpp_m = (
                    sum(len(weight[0]) for weight in c["weights"].values())
                    * 8
                    / n_pixels
                )
            else:
                bpp_m = 0

    else:
        assert not isinstance(model, QuantizedModelWrapper)
        # entropy estimation
        if y_hat is not None:
            out_dict = {
                "x_hat": model.g_s(y_hat),
                "likelihoods": {
                    "y": None,
                    "z": None,
                },
            }
        elif y is not None and z is not None:
            out_dict = forward_dec(model, y, z)
        else:
            out_dict = model(x_)
        x_hat = crop(out_dict["x_hat"], x.shape[2:])
        x_hat = x_hat.clamp(0, 1)
        if out_dict["likelihoods"]["y"] is None:
            bpp: float = 0.0
        else:
            bpp_ent: torch.Tensor = sum(
                (-torch.log2(likelihoods).sum() / n_pixels)
                for likelihoods in out_dict["likelihoods"].values()
            )
            bpp: float = bpp_ent.item() / x.shape[0]
        bpp_m: float = 0.0
        if m_likelihoods is not None:
            for likelihood in m_likelihoods.values():
                if likelihood is None:
                    continue
                bpp_m += -likelihood.log2().sum().item()
            bpp_m = bpp_m / n_pixels

    mse: torch.Tensor = (x - x_hat).square().mean(dim=(1, 2, 3))
    psnr: torch.Tensor = -10 * mse.log10().mean()
    return psnr.item(), bpp, bpp_m, mse.mean().item(), x_hat


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        argument_default=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("path", type=Path)
    parser.add_argument("--quality", type=int, default=1)
    parser.add_argument("--opt-enc", action="store_true", default=False)
    parser.add_argument("--pipeline", default="default", choices={"default", "swap", "end2end"})
    parser.add_argument("--model", type=str, default="wacnn")
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--lmbda", type=float, required=True)
    parser.add_argument("--dim_adapter_wacnn", nargs=2, type=int, default=[0, 2])
    parser.add_argument(
        "--dim_adapter_1", nargs=7, type=int, default=[0, 0, 0, 0, 0, 2, 0]
    )
    parser.add_argument(
        "--dim_adapter_2", nargs=7, type=int, default=[0, 0, 0, 0, 0, 0, 0]
    )
    parser.add_argument("--groups", type=int, default=1)
    parser.add_argument("--position", type=str, default="last")
    parser.add_argument("--connection", default="serial")
    parser.add_argument(
        "--distrib",
        type=str,
        choices={"spike-and-slab", "logistic"},
        default="logistic",
    )
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--sigma", type=float, default=0.05)
    parser.add_argument("--width", type=float, default=0.06)
    parser.add_argument("--data_type", default="uint8")
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-3)

    # options for the 2nd stage
    parser.add_argument("--lr_2", type=float, default=1e-3)
    parser.add_argument("--iterations_2", type=int, default=500)
    parser.add_argument("--regex", type=str, default="g_s\.5\.adapter.*")
    parser.add_argument("--no-cuda", action="store_true", default=False)
    parser.add_argument("--out", type=Path, default="temp/")
    args = parser.parse_args()
    return args


def optimize(
    model_qua: QuantizedModelWrapper,
    model: nn.Module,
    criterion: RateDistortionModelLoss,
    x: torch.Tensor,
    x_pad: torch.Tensor,
    iterations: int,
    lr: float,
) -> None:
    optimizer, aux_optimizer = configure_optimizers(model, lr, 1e-3)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=(iterations * 8 // 10), gamma=0.1
    )
    start = time.time()
    for it in range(iterations):
        optimizer.zero_grad()
        m_likelihoods = model_qua.update_parameters(model)
        # output = model_qua(x_pad, shape=x.shape[2:])
        output = model_qua.model(x_pad)
        output["x_hat"] = output["x_hat"].clamp(0, 1)
        output["x_hat"] = crop(output["x_hat"], x.shape[2:])
        output["m_likelihoods"] = m_likelihoods

        out_criterion: dict = criterion(output, x)
        out_criterion["loss"].backward()
        optimizer.step()
        lr_scheduler.step()

        # loss is much higher at the training time than at the test time
        # because - y_likelihoods.log2().sum() is large due to additive noise approx.
        logging.info(
            "Loss: {:.4f}, Time: {:.2f}s, lr: {}".format(
                out_criterion["loss"].item(),
                time.time() - start,
                optimizer.param_groups[0]["lr"],
            )
        )
        aux_loss = model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()

    # quantize the model parameters
    with torch.no_grad():
        m_likelihoods = model_qua.update_parameters(model)


def optimize_latent(
    model: nn.Module,
    criterion: RateDistortionModelLoss,
    x: torch.Tensor,
    x_pad: torch.Tensor,
    iterations: int,
    lr: float,
) -> tuple:
    with torch.no_grad():
        out_net = forward_enc(model, x_pad)

    out_net["y"].requires_grad_(True)
    out_net["z"].requires_grad_(True)
    optimizer = torch.optim.Adam([out_net["y"], out_net["z"]], lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=(iterations * 8 // 10), gamma=0.1
    )
    start = time.time()

    # following [Yang+, NeurIPS 20]
    tau_decay_it = 0
    tau_decay_factor = 0.001

    _quantize_ent = model.entropy_bottleneck.quantize
    _quantize_cond = model.gaussian_conditional.quantize

    for it in range(iterations):
        decaying_iter: int = it - tau_decay_it
        # if decaying_iter < 0, tau should be 0.5.
        tau: float = min(0.5, 0.5 * np.exp(-tau_decay_factor * decaying_iter))

        model.entropy_bottleneck.quantize = lambda x, mode, medians=None: quantize_sga(
            x, tau, medians
        )
        model.gaussian_conditional.quantize = (
            lambda x, mode, medians=None: quantize_sga(x, tau, medians)
        )
        optimizer.zero_grad()
        output = forward_dec(model, out_net["y"].clone(), out_net["z"].clone())
        output["x_hat"] = crop(output["x_hat"], x.shape[2:])
        output["m_likelihoods"] = dict()
        out_criterion: dict = criterion(output, x)
        out_criterion["loss"].backward()
        optimizer.step()
        lr_scheduler.step()

        if (it + 1) % 10 == 0:
            # loss is much higher at the training time than at the test time
            # because - y_likelihoods.log2().sum() is large due to additive noise approx.
            logging.info(
                "Loss: {:.4f}, Time: {:.2f}s, lr: {}".format(
                    out_criterion["loss"].item(),
                    time.time() - start,
                    optimizer.param_groups[0]["lr"],
                )
            )

    model.entropy_bottleneck.quantize = lambda self, x, medians=None: _quantize_ent(
        self, x, medians
    )
    model.gaussian_conditional.quantize = lambda self, x, medians=None: _quantize_cond(
        self, x, medians
    )

    out_net["y"].requires_grad_(False)
    out_net["z"].requires_grad_(False)

    return out_net["y"], out_net["z"]


def optimize_dec(
    model_qua: QuantizedModelWrapper,
    model: nn.Module,
    criterion: RateDistortionModelLoss,
    x: torch.Tensor,
    iterations: int,
    lr: float,
    y=None,
    z=None,
    y_hat=None,
) -> None:
    optimizer, aux_optimizer = configure_optimizers(model, lr, 1e-3)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=(iterations * 8 // 10), gamma=0.1
    )
    start = time.time()
    for it in range(iterations):
        optimizer.zero_grad()
        # model weights -> model_qua weights -> loss
        m_likelihoods = model_qua.update_parameters(model)
        if y_hat is not None:
            output = {
                "x_hat": model_qua.model.g_s(y_hat).clamp(0, 1),
                "likelihoods": {
                    "y": None,
                    "z": None,
                },
            }

        else:
            output = forward_dec(model_qua.model, y, z)

        output["x_hat"] = crop(output["x_hat"], x.shape[2:])
        output["m_likelihoods"] = m_likelihoods
        out_criterion: dict = criterion(output, x)
        out_criterion["loss"].backward()
        optimizer.step()
        lr_scheduler.step()

        logging.info(
            "Loss: {:.4f}, Time: {:.2f}s, lr: {}".format(
                out_criterion["loss"].item(),
                time.time() - start,
                optimizer.param_groups[0]["lr"],
            )
        )

    # final update
    with torch.no_grad(), torch.backends.cudnn.flags(**CUDNN_INFERENCE_FLAGS):
        model_qua.update_parameters(model)


def optimize_latent_and_dec(
    model_qua: QuantizedModelWrapper,
    model: nn.Module,
    criterion: RateDistortionModelLoss,
    x: torch.Tensor,
    x_pad: torch.Tensor,
    iterations: int,
    lr: float,
    lr_2: float,
) -> tuple:
    model_qua.eval()
    model_qua.w_ent.train()

    with torch.no_grad():
        out_net = forward_enc(model, x_pad)

    out_net["y"].requires_grad_(True)
    out_net["z"].requires_grad_(True)
    optimizer, aux_optimizer = configure_optimizers(model, lr_2, 1e-3, model_qua.regex)

    param_group = copy.deepcopy(optimizer.param_groups[0])
    param_group["params"] = [out_net["y"], out_net["z"]]
    param_group["lr"] = lr
    optimizer.add_param_group(param_group)
    logging.info(param_group)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=(iterations * 8 // 10), gamma=0.1
    )
    start = time.time()

    # following [Yang+, NeurIPS 20]
    tau_decay_it = 0
    tau_decay_factor = 0.001

    _quantize_ent = model.entropy_bottleneck.quantize
    _quantize_cond = model.gaussian_conditional.quantize

    for it in range(iterations):
        decaying_iter: int = it - tau_decay_it
        # if decaying_iter < 0, tau should be 0.5.
        tau: float = min(0.5, 0.5 * np.exp(-tau_decay_factor * decaying_iter))

        model.entropy_bottleneck.quantize = lambda x, mode, medians=None: quantize_sga(
            x, tau, medians
        )
        model.gaussian_conditional.quantize = (
            lambda x, mode, medians=None: quantize_sga(x, tau, medians)
        )
        optimizer.zero_grad()

        # model weights -> model_qua weights -> loss
        m_likelihoods = model_qua.update_parameters(model)

        output = forward_dec(model, out_net["y"].clone(), out_net["z"].clone())
        output["x_hat"] = crop(output["x_hat"], x.shape[2:])
        output["m_likelihoods"] = m_likelihoods
        out_criterion: dict = criterion(output, x)
        out_criterion["loss"].backward()
        optimizer.step()
        lr_scheduler.step()

        if (it + 1) % 10 == 0:
            # loss is much higher at the training time than at the test time
            # because - y_likelihoods.log2().sum() is large due to additive noise approx.
            logging.info(
                "Loss: {:.4f}, Time: {:.2f}s, lr: {}, lr_2: {}".format(
                    out_criterion["loss"].item(),
                    time.time() - start,
                    optimizer.param_groups[0]["lr"],
                    optimizer.param_groups[1]["lr"],
                )
            )

    model.entropy_bottleneck.quantize = lambda self, x, medians=None: _quantize_ent(
        self, x, medians
    )
    model.gaussian_conditional.quantize = lambda self, x, medians=None: _quantize_cond(
        self, x, medians
    )

    out_net["y"].requires_grad_(False)
    out_net["z"].requires_grad_(False)

    # final update
    with torch.no_grad(), torch.backends.cudnn.flags(**CUDNN_INFERENCE_FLAGS):
        model_qua.update_parameters(model)
    return out_net["y"], out_net["z"]


@torch.no_grad()
def evaluate(
    model_qua: QuantizedModelWrapper,
    x: torch.Tensor,
    lmbda: float,
    actual: bool = False,
    model: nn.Module = None,
    y=None,
    z=None,
    y_hat=None,
    strings=None,
    shape=None,
) -> torch.Tensor:
    if actual:
        if model_qua.w_ent.width != 0:
            psnr, bpp, bpp_m, mse, x_hat = test(
                model_qua,
                x,
                actual=True,
                y=y,
                z=z,
                strings=strings,
                shape=shape,
            )
            logging.info(
                "<ACTUAL>  PSNR: {:.3f}, BPP: {:.4f} [model {:.4f}], Loss: {:.4f}".format(
                    psnr, bpp + bpp_m, bpp_m, bpp + bpp_m + lmbda * mse * 255 ** 2
                )
            )
        else:
            assert model_qua.w_ent.data_type in {"uint8", "float16", "float32"}
            if model_qua.w_ent.data_type == "uint8":
                data_types = ["float32", "uint8"]
            if model_qua.w_ent.data_type == "float16":
                data_types = ["float32", "float16", "uint8"]
            elif model_qua.w_ent.data_type == "float32":
                data_types = ["float32"]

            for data_type in data_types:
                # quantize model_qua.model with different data_type
                model_qua_ = copy.deepcopy(model_qua)

                model_qua_.w_ent.data_type = data_type
                psnr, bpp, bpp_m, mse, x_hat = test(
                    model_qua_,
                    x,
                    actual=True,
                    y=y,
                    z=z,
                    strings=strings,
                    shape=shape,
                )
                logging.info(
                    "<ACTUAL {}>  PSNR: {:.3f}, BPP: {:.4f} [model {:.4f}], Loss: {:.4f}".format(
                        data_type,
                        psnr,
                        bpp + bpp_m,
                        bpp_m,
                        bpp + bpp_m + lmbda * mse * 255 ** 2,
                    )
                )
                del model_qua_

    else:
        assert model is not None
        model.eval()
        psnr, bpp, _, mse, x_hat = test(model, x, y=y, z=z, y_hat=y_hat)
        logging.info(
            "<Before Q.> PSNR: {:.3f}, BPP: {:.4f}, Loss: {:.4f}".format(
                psnr, bpp, bpp + lmbda * mse * 255 ** 2
            )
        )
        m_likelihoods = model_qua.update_parameters(model)
        psnr, bpp, bpp_m, mse, x_hat = test(
            model_qua.model, x, m_likelihoods, y=y, z=z, y_hat=y_hat
        )

        loss = bpp + bpp_m + lmbda * mse * 255 ** 2
        logging.info(
            "PSNR: {:.3f}, BPP: {:.4f} [model {:.4f}], Loss: {:.4f}".format(
                psnr, bpp + bpp_m, bpp_m, loss
            )
        )
    return x_hat


def main(args: argparse.Namespace) -> None:
    @torch.no_grad()
    def prepare_model_with_adapters(model):
        # if n_adapters = 0 use ZeroLayer -- equivalent with no adapter
        if args.model in {"cheng2020-attn", "wacnn"}:
            state_dict = model.state_dict()
            if args.model == "cheng2020-attn":
                model = Cheng2020AttnAdapter(
                    model.N,
                    args.dim_adapter_1,
                    args.dim_adapter_2,
                    args.groups,
                    connection=args.connection,
                )
            elif args.model == "wacnn":
                model = WACNNAdapter(
                    model.N,
                    model.M,
                    args.dim_adapter_wacnn[0],
                    args.dim_adapter_wacnn[1],
                    args.groups,
                    position=args.position,
                )

            info = model.load_state_dict(state_dict, strict=False)
            print(info)
            model.to(device)

            model_qua = QuantizedModelWrapper(model, w_ent, regex=args.regex)
            # compute diff. from zero
            for key in model_qua.params_init.keys():
                if "adapter" in key:
                    model_qua.params_init[key].fill_(0)

        else:
            assert (
                args.dim_adapter_1 == [0, 0, 0, 0, 0, 0, 0]
                and args.dim_adapter_2 is None
            )
            model_qua.regex = args.regex

        model_qua.report_params()
        model_qua.update_ent(force=True)
        model_qua.to(device)

        return model, model_qua

    args.out.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s %(levelname)s %(name)s :%(message)s"
    logging.basicConfig(
        filename=(args.out / "log"), filemode="w", level=logging.INFO, format=fmt
    )

    device = "cpu" if args.no_cuda else "cuda"

    img: Image.Image = read_image(args.path)

    transform = transforms.ToTensor()
    x: torch.Tensor = transform(img)[None]  # .repeat(16, 1, 1, 1)
    x = x.to(device)
    x_pad = pad(x)

    model = get_model(args.model, args.quality, args.model_path)

    if args.distrib == "spike-and-slab":
        distrib = SpikeAndSlabCDF(args.width, args.sigma, args.alpha)
    elif args.distrib == "logistic":
        distrib = LogisticCDF(scale=args.sigma)
    else:
        raise NotImplementedError
    w_ent = WeightEntropyModule(distrib, args.width, data_type=args.data_type)
    criterion = RateDistortionModelLoss(args.lmbda)

    model.to(device)
    w_ent.to(device)
    criterion.to(device)

    model_qua = QuantizedModelWrapper(
        model, w_ent, regex=args.regex if args.opt_enc else "none"
    )
    model_qua.report_params()
    model_qua.eval()
    x_hat = evaluate(model_qua, x, args.lmbda, actual=True)
    transforms.ToPILImage()(x[0]).save(args.out / "input.png")
    transforms.ToPILImage()(x_hat[0]).save(args.out / "init.png")

    if args.pipeline == "swap":
        model.eval()
        with torch.no_grad(), torch.backends.cudnn.flags(**CUDNN_INFERENCE_FLAGS):
            compressed = model.compress(x_pad)
            y_hat = decode_latent(model, **compressed)
            y_hat.requires_grad_(False)
            y_hat = y_hat.to(device)

        model, model_qua = prepare_model_with_adapters(model)

        model_qua.eval()
        model_qua.w_ent.train()
        optimize_dec(
            model_qua,
            model,
            criterion,
            x,
            args.iterations_2,
            args.lr_2,
            y_hat=y_hat,
        )

        model_qua.eval()
        with torch.no_grad(), torch.backends.cudnn.flags(**CUDNN_INFERENCE_FLAGS):
            x_hat = evaluate(
                model_qua,
                x,
                args.lmbda,
                actual=True,
                **compressed,
            )
        torch.save(model_qua.compress_weight(), args.out / "weights.pt")
        transforms.ToPILImage()(x_hat[0]).save(args.out / "opt.png")

        # 2. optimize latent
        y, z = optimize_latent(
            model_qua.model,
            criterion,
            x,
            x_pad,
            args.iterations,
            args.lr,
        )
        with torch.no_grad(), torch.backends.cudnn.flags(**CUDNN_INFERENCE_FLAGS):
            compressed = encode_latent(model_qua.model, y, z)
            torch.save(compressed, args.out / "compressed.pt")
            x_hat = evaluate(model_qua, x, args.lmbda, actual=True, **compressed)
            transforms.ToPILImage()(x_hat[0]).save(args.out / "opt_2.png")
        return

    model_qua.train()
    model_qua.update_ent(force=True)

    if args.pipeline == "end2end":
        # implementation for [Rozendaal+, ICLR 21]
        if args.opt_enc:
            optimize(
                model_qua,
                model,
                criterion,
                x,
                x_pad,
                args.iterations,
                args.lr,
            )
        else:
            model, model_qua = prepare_model_with_adapters(model)
            # encoder and entropy models are in evaluation mode.
            model_qua.eval()
            model_qua.w_ent.train()
            y, z = optimize_latent_and_dec(
                model_qua,
                model,
                criterion,
                x,
                x_pad,
                args.iterations,
                args.lr,
                args.lr_2,
            )
        model_qua.eval()
        with torch.no_grad(), torch.backends.cudnn.flags(**CUDNN_INFERENCE_FLAGS):
            if args.opt_enc:
                compressed = dict()
            else:
                compressed = encode_latent(model_qua.model, y, z)
            x_hat = evaluate(model_qua, x, args.lmbda, actual=True, **compressed)
            compressed = model_qua.compress(x_pad)
            torch.save(compressed["weights"], args.out / "weights.pt")
            x_hat = model_qua.decompress(**compressed)["x_hat"]
            print(-10 * (x - x_hat.mul(255).round().div(255)).square().mean().log10())
            compressed.pop("weights")
            torch.save(compressed, args.out / "latent.pt")
            transforms.ToPILImage()(x_hat[0]).save(args.out / "opt.png")
            return

    cache_root: Path = Path("cache")
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_path: Path = (
        cache_root
        / f"{args.path.name}-{args.quality}-{args.model}-{args.lr}-{args.iterations}.pt"
    )

    if not cache_path.exists():
        y, z = optimize_latent(model, criterion, x, x_pad, args.iterations, args.lr)
        with torch.no_grad(), torch.backends.cudnn.flags(**CUDNN_INFERENCE_FLAGS):
            compressed = encode_latent(model_qua.model, y, z)
        torch.save(compressed, cache_path)
        del y, z
    else:
        compressed = torch.load(cache_path)

    with torch.no_grad(), torch.backends.cudnn.flags(**CUDNN_INFERENCE_FLAGS):
        y_hat = decode_latent(model_qua.model, **compressed)
        y_hat.requires_grad_(False)
        model_qua.eval()
        x_hat = evaluate(model_qua, x, args.lmbda, actual=True, **compressed)
        transforms.ToPILImage()(x_hat[0]).save(args.out / "opt.png")

    logging.info("Preparing a compression model with adapters.")
    y_hat = y_hat.to(device)

    model, model_qua = prepare_model_with_adapters(model)

    # encoder and entropy models are in evaluation mode.
    model_qua.eval()
    model_qua.w_ent.train()
    optimize_dec(
        model_qua,
        model,
        criterion,
        x,
        args.iterations_2,
        args.lr_2,
        y_hat=y_hat,
    )
    model_qua.eval()
    with torch.no_grad(), torch.backends.cudnn.flags(**CUDNN_INFERENCE_FLAGS):
        x_hat = evaluate(
            model_qua,
            x,
            args.lmbda,
            actual=True,
            **compressed,
        )
        assert model_qua.w_ent.data_type == args.data_type
        torch.save(model_qua.compress_weight(), args.out / "weights.pt")
        transforms.ToPILImage()(x_hat[0]).save(args.out / "opt_2.png")


if __name__ == "__main__":
    main(parse_args())
