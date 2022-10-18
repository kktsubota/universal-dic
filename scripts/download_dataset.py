import argparse
from pathlib import Path
import random
import requests
import time
import tqdm

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default="data/")
    args = parser.parse_args()

    dataset_names = ("kodak", "bam")
    for dataset_name in dataset_names:
        df = pd.read_csv(f"datasets/{dataset_name}_url.csv")

        outdir: Path = args.out / dataset_name
        outdir.mkdir(parents=True, exist_ok=True)
        for name, url in tqdm.tqdm(df.values):
            ext = Path(url).suffix
            path = args.out / dataset_name / f"{name}{ext}"
            if path.exists():
                continue

            time.sleep(0.5 + random.random())
            r = requests.get(url, allow_redirects=True)
            with path.open("wb") as f:
                f.write(r.content)


if __name__ == "__main__":
    main()
