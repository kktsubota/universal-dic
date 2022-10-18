import torch
from torch import distributions as D


class SpikeAndSlabCDF:
    def __init__(
        self, width: float = 5e-3, sigma: float = 5e-2, alpha: float = 1000
    ) -> None:
        self.alpha = alpha

        mean = torch.tensor(0.0)
        self.slab = D.Normal(mean, torch.tensor(sigma))
        if width != 0:
            self.spike = D.Normal(mean, torch.tensor(width / 6))
        else:
            self.spike = None

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        cdf_slab = self.slab.cdf(x)
        if self.spike is None:
            return cdf_slab
        else:
            cdf_spike = self.spike.cdf(x)
            return (cdf_slab + self.alpha * cdf_spike) / (1 + self.alpha)


class LogisticCDF:
    """CDF of logistic distribution with scale=scale and loc=loc."""

    def __init__(self, scale: float, loc: float = 0.0) -> None:
        self.loc = loc
        self.scale = scale

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 + 0.5 * torch.tanh((x - self.loc) / self.scale / 2)
