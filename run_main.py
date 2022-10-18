import argparse
from pathlib import Path
import subprocess
import sys

sys.path = ["./"] + sys.path
import dataset


lam_dict = {
    1: 0.0018,
    2: 0.0035,
    3: 0.0067,
    4: 0.0130,
    5: 0.0250,
    6: 0.0483,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "domain", type=str, choices=dataset.root_dict.keys(),
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", type=str, default="wacnn")
    parser.add_argument("--quality", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=0)
    parser.add_argument("--width", type=float, default=0.06)
    parser.add_argument("--sigma", type=float, default=0.05)
    parser.add_argument("--data_type", type=str, default="uint8")
    parser.add_argument("--distrib", type=str, default="logistic")
    parser.add_argument("--groups", type=int, default=1)
    parser.add_argument("--opt-enc", action="store_true", default=False)
    parser.add_argument("--regex", default="'g_s\.5\.adapter.*'")
    parser.add_argument("--n-dim", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--iteration", type=int, default=2000)
    parser.add_argument("--lr_2", type=float, default=1e-3)
    parser.add_argument("--iteration_2", type=int, default=500)
    parser.add_argument("--connection", type=str, default="serial")
    parser.add_argument("--position", type=str, default="last")
    parser.add_argument("--swap", action="store_true", default=False)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    paths = dataset.get_paths(args.domain)
    for path in paths:
        cmd = [
            "python",
            "main.py",
            path.as_posix(),
            "--lmbda",
            str(lam_dict[args.quality]),
            "--quality",
            str(args.quality),
            "--model",
            args.model,
            "--groups",
            str(args.groups),
            "--distrib",
            args.distrib,
            "--connection",
            args.connection,
            "--position",
            args.position,
        ]

        out: Path = args.out / args.model / f"q{args.quality}" / args.domain / path.name
        cmd += [
            "--iterations",
            str(args.iteration),
            "--lr",
            str(args.lr),
        ]

        if "adapter" in args.regex:
            if args.model == "cheng2020-attn":
                if args.regex.endswith("adapter_2.*'"):
                    dim_adapter_1 = [0] * 7
                    if args.regex.endswith("g_s\.[6-8]\.adapter_2.*'"):
                        dim_adapter_2 = [0] * 4 + [args.n_dim] * 3
                    elif args.regex.endswith("g_s\.[6,8]\.adapter_2.*'"):
                        dim_adapter_2 = [0] * 4 + [args.n_dim, 0, args.n_dim]
                    elif args.regex.endswith("g_s\.[7-8]\.adapter_2.*'"):
                        dim_adapter_2 = [0] * 5 + [args.n_dim] * 2
                    elif args.regex.endswith("g_s\.[8-8]\.adapter_2.*'"):
                        dim_adapter_2 = [0] * 6 + [args.n_dim] * 1
                    elif args.regex.endswith("g_s\.[7-7]\.adapter_2.*'"):
                        dim_adapter_2 = [0] * 5 + [args.n_dim] * 1 + [0]
                    elif args.regex.endswith("g_s\.[6-6]\.adapter_2.*'"):
                        dim_adapter_2 = [0] * 4 + [args.n_dim] * 1 + [0] * 2
                    else:
                        raise NotImplementedError
                elif args.regex.endswith("adapter_1.*'"):
                    dim_adapter_2 = [0] * 7
                    if args.regex.endswith("g_s\.[6-8]\.adapter_1.*'"):
                        dim_adapter_1 = [0] * 4 + [args.n_dim] * 3
                    elif args.regex.endswith("g_s\.[6,8]\.adapter_1.*'"):
                        dim_adapter_1 = [0] * 4 + [args.n_dim, 0, args.n_dim]
                    elif args.regex.endswith("g_s\.[7-8]\.adapter_1.*'"):
                        dim_adapter_1 = [0] * 5 + [args.n_dim] * 2
                    elif args.regex.endswith("g_s\.[8-8]\.adapter_1.*'"):
                        dim_adapter_1 = [0] * 6 + [args.n_dim] * 1
                    elif args.regex.endswith("g_s\.[7-7]\.adapter_1.*'"):
                        dim_adapter_1 = [0] * 5 + [args.n_dim] * 1 + [0]
                    elif args.regex.endswith("g_s\.[6-6]\.adapter_1.*'"):
                        dim_adapter_1 = [0] * 4 + [args.n_dim] * 1 + [0] * 2
                    else:
                        raise NotImplementedError
                else:
                    dim_adapter_2 = None
                    if args.regex.endswith("g_s\.[6-8]\.adapter.*'"):
                        dim_adapter_1 = [0] * 4 + [args.n_dim] * 3
                    elif args.regex.endswith("g_s\.[6,8]\.adapter.*'"):
                        dim_adapter_1 = [0] * 4 + [args.n_dim, 0, args.n_dim]
                    elif args.regex.endswith("g_s\.[7-8]\.adapter.*'"):
                        dim_adapter_1 = [0] * 5 + [args.n_dim] * 2
                    elif args.regex.endswith("g_s\.[8-8]\.adapter.*'"):
                        dim_adapter_1 = [0] * 6 + [args.n_dim] * 1
                    else:
                        raise NotImplementedError
                cmd += [
                    "--dim_adapter_1",
                    " ".join(map(str, dim_adapter_1)),
                ]

                if dim_adapter_2 is not None:
                    cmd += [
                        "--dim_adapter_2",
                        " ".join(map(str, dim_adapter_2)),
                    ]
            elif args.model == "wacnn":
                # g_s.5.adapter.0.weight', 'g_s.5.adapter.1.weight'
                if args.regex.startswith("'g_s\.5\.") and args.regex.endswith(
                    "adapter.*'"
                ):
                    dim_adapter_wacnn = [0, args.n_dim]
                elif args.regex.startswith("'g_s\.0\.") and args.regex.endswith(
                    "adapter.*'"
                ):
                    dim_adapter_wacnn = [args.n_dim, 0]
                elif args.regex.startswith("'g_s\.[0,5]\.") and args.regex.endswith(
                    "adapter.*'"
                ):
                    dim_adapter_wacnn = [args.n_dim, args.n_dim]
                else:
                    raise NotImplementedError

                cmd += [
                    "--dim_adapter_wacnn",
                    " ".join(map(str, dim_adapter_wacnn)),
                ]

            else:
                raise NotImplementedError

        cmd += [
            "--iterations_2",
            str(args.iteration_2),
            "--lr_2",
            str(args.lr_2),
            "--regex",
            args.regex,
            "--width",
            str(args.width),
            "--alpha",
            str(args.alpha),
            "--sigma",
            str(args.sigma),
            "--data_type",
            args.data_type,
            "--out",
            out.as_posix(),
        ]
        if args.swap:
            cmd += ["--swap"]
        if args.opt_enc:
            cmd += ["--opt-enc"]
        if args.dry_run:
            print(" ".join(cmd))
        else:
            subprocess.run(" ".join(cmd), shell=True)


if __name__ == "__main__":
    main()
