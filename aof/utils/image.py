from pathlib import Path

import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio
import torch
from torchvision import transforms
from torch.nn import functional as F


# modified from https://github.com/InterDigitalInc/CompressAI/blob/master/examples/codec.py
# Copyright (c) 2021-2022 InterDigital Communications, Inc Licensed under BSD 3-Clause Clear License.
def pad(
    x: torch.Tensor, p: int = 2 ** 6, mode: str = "constant", center: bool = True
) -> torch.Tensor:
    assert mode in {"constant", "reflect", "replicate", "circular"}
    h, w = x.size(2), x.size(3)
    H = (h + p - 1) // p * p
    W = (w + p - 1) // p * p
    if center:
        padding_left = (W - w) // 2
        padding_right = W - w - padding_left
        padding_top = (H - h) // 2
        padding_bottom = H - h - padding_top
    else:
        padding_left = 0
        padding_right = W - w
        padding_top = 0
        padding_bottom = H - h

    if mode == "constant":
        kwargs = {"mode": mode, "value": 0}
    else:
        kwargs = {"mode": mode}

    return F.pad(
        x, (padding_left, padding_right, padding_top, padding_bottom), **kwargs
    )


# modified from https://github.com/InterDigitalInc/CompressAI/blob/master/examples/codec.py
# Copyright (c) 2021-2022 InterDigital Communications, Inc Licensed under BSD 3-Clause Clear License.
def crop(x: torch.Tensor, size: tuple, center: bool = True) -> torch.Tensor:
    H, W = x.size(2), x.size(3)
    h, w = size
    if center:
        padding_left = (W - w) // 2
        padding_right = W - w - padding_left
        padding_top = (H - h) // 2
        padding_bottom = H - h - padding_top
    else:
        padding_left = 0
        padding_right = W - w
        padding_top = 0
        padding_bottom = H - h

    return F.pad(
        x,
        (-padding_left, -padding_right, -padding_top, -padding_bottom),
        mode="constant",
        value=0,
    )


def read_image(path: Path, color_bg: tuple = (255, 255, 255)) -> Image.Image:
    # color_bg: white
    with Image.open(path) as f:
        f.load()
        if f.mode in {"RGBA", "P", "LA"}:
            if f.mode in {"P", "LA"}:
                f = f.convert("RGBA")
            img = Image.new("RGB", f.size, color_bg)
            img.paste(f, mask=f.split()[3])
        elif f.mode == "RGB":
            img = f
        elif f.mode == "L":
            img = f.convert("RGB")
        else:
            raise NotImplementedError(f.mode, path)
    return img


def calc_psnr(img_1: Image.Image, img_2: Image.Image):
    return peak_signal_noise_ratio(
        np.asarray(img_1.convert("RGB")), np.asarray(img_2.convert("RGB"))
    )


def calc_msssim(img_1: Image.Image, img_2: Image.Image) -> float:
    from pytorch_msssim import MS_SSIM

    x_1 = transforms.ToTensor()(img_1)[None]
    x_2 = transforms.ToTensor()(img_2)[None]

    return MS_SSIM(data_range=1)(x_1, x_2).item()


def calc_bpp(bits, shape):
    return bits / shape[0] / shape[1]
