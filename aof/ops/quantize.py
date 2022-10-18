import torch


# modified from https://github.com/mandt-lab/improving-inference-for-neural-image-compression/blob/main/sga.py#L110-L121
# Copyright (c) 2020 mandt-lab Licensed under The MIT License
def quantize_sga(y: torch.Tensor, tau: float, medians=None, eps: float = 1e-5):
    # use Gumbel Softmax implemented in tfp.distributions.RelaxedOneHotCategorical

    # (N, C, H, W)
    if medians is not None:
        y -= medians
    y_floor = torch.floor(y)
    y_ceil = torch.ceil(y)
    # (N, C, H, W, 2)
    y_bds = torch.stack([y_floor, y_ceil], dim=-1)
    # (N, C, H, W, 2)
    ry_logits = torch.stack(
        [
            -torch.atanh(torch.clamp(y - y_floor, -1 + eps, 1 - eps)) / tau,
            -torch.atanh(torch.clamp(y_ceil - y, -1 + eps, 1 - eps)) / tau,
        ],
        axis=-1,
    )
    # last dim are logits for DOWN or UP
    ry_dist = torch.distributions.RelaxedOneHotCategorical(tau, logits=ry_logits)
    ry_sample = ry_dist.rsample()
    outputs = torch.sum(y_bds * ry_sample, dim=-1)
    if medians is not None:
        outputs += medians
    return outputs
