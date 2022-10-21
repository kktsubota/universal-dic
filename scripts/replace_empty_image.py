from pathlib import Path
import random
import time

import pandas as pd
import requests
import tqdm


DATASET_ROOT: Path = Path("data/bam")
NEW_URL_LIST: str = "datasets/bam_new_url.csv"

# The file size of empty images whose resolution is 600x343
# e.g., 15642096.png (https://mir-s3-cdn-cf.behance.net/project_modules/disp/bf992815642096.56038dc9bf59c.png)
EMPTY_IMAGE_SIZE: int = 3727


def main():
    # empty image names
    names: set = set()
    for path in DATASET_ROOT.iterdir():
        file_size: int = path.stat().st_size

        if file_size == EMPTY_IMAGE_SIZE:
            names.add(path.stem)

    # get new URLs of the empty images to replace
    df = pd.read_csv(NEW_URL_LIST)
    names = {int(name) for name in names}
    df = df[df["id"].isin(names)]

    # download new images
    for name, url in tqdm.tqdm(df.values):
        ext: str = Path(url).suffix
        path = DATASET_ROOT / f"{name}{ext}"

        time.sleep(0.5 + random.random())
        r = requests.get(url, allow_redirects=True)
        with path.open("wb") as f:
            f.write(r.content)
        
        names.remove(name)

    # test if all the empty images are replaced
    print("file names of empty images:", names)
    assert len(names) == 0, f"The number of empty images is expected to be 0, but actual: {len(names)}."


if __name__ == "__main__":
    main()
