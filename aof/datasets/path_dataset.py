from PIL import Image
import torch

from aof.utils.image import read_image


class PathDataset(torch.utils.data.Dataset):
    def __init__(self, paths: list, transform=None) -> None:
        super().__init__()
        self.paths = paths
        self.transform = transform

    def __getitem__(self, index: int) -> torch.Tensor:
        path = self.paths[index]

        # alpha channel is converted to white
        img: Image.Image = read_image(path)

        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self) -> int:
        return len(self.paths)
