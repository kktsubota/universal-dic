import math

import torch
from torch import nn

# modified from https://github.com/InterDigitalInc/CompressAI/blob/master/examples/train.py
# Copyright (c) 2021-2022 InterDigital Communications, Inc Licensed under BSD 3-Clause Clear License.
class RateDistortionLoss(nn.Module):
    def __init__(self, lmbda=None, beta=None):
        super().__init__()
        self.mse = nn.MSELoss()
        assert lmbda is None or beta is None
        self.lmbda = lmbda
        self.beta = beta

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        out["mse_loss"] = self.mse(output["x_hat"], target)
        if self.lmbda is not None:
            out["loss"] = self.lmbda * 255 ** 2 * out["mse_loss"] + out["bpp_loss"]
        else:
            out["loss"] = out["mse_loss"] + self.beta * out["bpp_loss"]

        return out


# modified from https://github.com/InterDigitalInc/CompressAI/blob/master/examples/train.py
# Copyright (c) 2021-2022 InterDigital Communications, Inc Licensed under BSD 3-Clause Clear License.
class RateDistortionModelLoss(nn.Module):
    # modified from RateDistortionLoss
    def __init__(self, lmbda: float = 1e-2) -> None:
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda

    def forward(self, output: dict, target: torch.Tensor) -> dict:
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = 0

        for likelihood in output["likelihoods"].values():
            if likelihood is not None:
                out["bpp_loss"] += torch.log(likelihood).sum() / (
                    -math.log(2) * num_pixels
                )

        for likelihood in output["m_likelihoods"].values():
            if likelihood is not None:
                out["bpp_loss"] += likelihood.log().sum() / (-math.log(2) * num_pixels)

        out["mse_loss"] = self.mse(output["x_hat"], target)
        out["loss"] = self.lmbda * 255 ** 2 * out["mse_loss"] + out["bpp_loss"]

        return out
