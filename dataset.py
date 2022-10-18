from pathlib import Path


root_dict = {
    "comic": "./data/bam/",
    "line": "./data/bam/",
    "natural": "./data/kodak/",
    "vector": "./data/bam/",
    # add
    "graphite": "./data/bam/",
    "oilpaint": "./data/bam/",
    "watercolor": "./data/bam/",
}

file_dict = {
    "line": "datasets/line-drawing.txt",
    "comic": "datasets/comic.txt",
    "natural": "datasets/natural.txt",
    "vector": "datasets/vector-art.txt",
    "graphite": "datasets/graphite.txt",
    "oilpaint": "datasets/oilpaint.txt",
    "watercolor": "datasets/watercolor.txt",
}


def get_paths(name: str):
    root = Path(root_dict[name])
    paths = list()
    with open(file_dict[name]) as f:
        paths = [root / line.rstrip() for line in f]
    return paths
