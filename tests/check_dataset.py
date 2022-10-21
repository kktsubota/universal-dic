import random
import time

import requests
import pandas as pd
import tqdm


URL_LIST: str = "datasets/bam_url.csv"
NEW_URL_LIST: str = "datasets/bam_new_url.csv"

# The file size of empty images whose resolution is 600x343
# e.g., 15642096.png (https://mir-s3-cdn-cf.behance.net/project_modules/disp/bf992815642096.56038dc9bf59c.png)
EMPTY_IMAGE_SIZE: int = 3727

def main():
    df = pd.read_csv(URL_LIST)
    df_new = pd.read_csv(NEW_URL_LIST)

    # merge two tables
    indices = df["id"].isin(df_new["id"])
    df.loc[indices, "url"] = df_new["url"].values

    names = list()
    for name, url in tqdm.tqdm(df.values):
        time.sleep(0.5 + random.random())
        r = requests.get(url, allow_redirects=True)

        if len(r.content) == EMPTY_IMAGE_SIZE:
            names.append(name)

    print("file names of empty images:", names)
    assert len(names) == 0, f"The number of empty images is expected to be 0, but actual: {len(names)}."


if __name__ == "__main__":
    main()
