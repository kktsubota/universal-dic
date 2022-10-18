import torch


# modified from https://github.com/InterDigitalInc/CompressAI/blob/master/examples/train.py
# Copyright (c) 2021-2022 InterDigital Communications, Inc Licensed under BSD 3-Clause Clear License.
def configure_optimizers(net, lr: float, aux_lr: float):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""

    parameters = {
        n
        for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    }
    aux_parameters = {
        n
        for n, p in net.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    }

    # Make sure we don't have an intersection of parameters
    params_dict = dict(net.named_parameters())
    inter_params = parameters & aux_parameters
    union_params = parameters | aux_parameters

    assert len(inter_params) == 0
    assert len(union_params) - len(params_dict.keys()) == 0

    optimizer = torch.optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
        lr=lr,
    )
    aux_optimizer = torch.optim.Adam(
        (params_dict[n] for n in sorted(aux_parameters)),
        lr=aux_lr,
    )
    return optimizer, aux_optimizer
