from collections import defaultdict
from pathlib import Path

from matplotlib import pyplot as plt
from matplotlib.colors import to_hex
import pandas as pd

from aof.utils.metrics import BD_RATE


colors = [
    (255, 75, 0),
    (3, 175, 122),
    # (0, 90, 255),
    (77, 96, 255),
    (255, 128, 130),
    (246, 170, 0),
    (153, 0, 153),
    (128, 64, 0),
    (255, 241, 0),
    (255, 0, 0),
    (0, 0, 255),
    (0, 255, 0),
]
# list of str
color_codes = [to_hex((c[0] / 255, c[1] / 255, c[2] / 255)) for c in colors]
markers = [
    "s",
    "x",
    "o",
    "^",
    "v",
    "|",
    "h",
    "D",
    "d",
    "p",
    "P",
    "X",
    "H",
    "*",
]

plt.rcParams["xtick.major.width"] = 2.0
plt.rcParams["ytick.major.width"] = 2.0
plt.rcParams["font.size"] = 14
plt.rcParams["axes.linewidth"] = 2.0
plt.rcParams["lines.linewidth"] = 2.0
plt.rcParams["lines.markersize"] = 8.0


def get_rdcurve(
    csv_paths: list,
    index=None,
) -> dict:
    rdcurve = dict(bpp_m=list(), bpp=list(), psnr=list(), loss=list())
    for i, csv_path in enumerate(csv_paths):
        df = pd.read_csv(csv_path, index_col=0)
        df["mse"] = 10 ** (-df["psnr"] / 10)
        if index is None:
            score = df.mean()
        else:
            score = df.iloc[index]
        if "bpp_m" in score:
            rdcurve["bpp_m"].append(score["bpp_m"])
        rdcurve["bpp"].append(score["bpp"])
        rdcurve["psnr"].append(score["psnr"])
    return rdcurve


def plot_rdcurve(rdcurve_dict, save_path):
    for i, (name, rdcurve) in enumerate(rdcurve_dict.items()):
        plt.plot(
            rdcurve["bpp"],
            rdcurve["psnr"],
            label=name,
            marker=markers[i],
            color=color_codes[i],
        )

    plt.legend()
    plt.grid()
    plt.xlabel("Rate (BPP)")
    plt.ylabel("PSNR (dB)")
    plt.savefig(save_path, bbox_inches="tight")
    plt.clf()


def main():
    outdir = Path("results/figures/")
    outdir.mkdir(parents=True, exist_ok=True)

    domains = (
        "vector",
        "comic",
        "line",
        "natural",
    )

    # 1. Rate-distortion curve
    table_dict = defaultdict(list)
    for domain in domains:
        tpl_dict = {
            "Ours": "results/ours/wacnn/q{}/{}_2nd.csv",
            "Zou et al., ISM 21": "results/zou-ism21/wacnn/q{}/{}_2nd.csv",
            "Yang et al., NeurIPS 20": "results/ours/wacnn/q{}/{}_1st.csv",
            "Baseline": "results/ours/wacnn/q{}/{}_0th.csv",
            "VVC": "results/VVC/q{}/{}.csv",
            "Lam et al., MM 20": "results/lam-mm20/wacnn/q{}/{}_2nd.csv",
            "Rozendaal et al., ICLR 21": "results/rozendaal-iclr21/wacnn/q{}/{}_1.csv",
        }
        rdcurve_dict = dict()
        for name, tpl in tpl_dict.items():
            if name == "VVC":
                qualities = list(range(24, 45, 3))
            else:
                qualities = list(range(1, 7))
            paths = [tpl.format(q, domain) for q in qualities]
            rdcurve_dict[name] = get_rdcurve(paths)

        plot_rdcurve(rdcurve_dict, outdir / f"{domain}.pdf")

        for name, rdcurve in rdcurve_dict.items():
            bdrate = BD_RATE(
                rdcurve_dict["VVC"]["bpp"],
                rdcurve_dict["VVC"]["psnr"],
                rdcurve["bpp"],
                rdcurve["psnr"],
            )
            table_dict[name].append(bdrate)

    df = pd.DataFrame.from_dict(table_dict).T
    df.columns = domains
    df["Average"] = df.mean(axis=1)
    print(df)
    print(df.round(2).to_latex())


if __name__ == "__main__":
    main()
