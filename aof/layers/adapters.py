import torch
from torch import nn


class ZeroLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)


@torch.no_grad()
def init_adapter_layer(adapter_layer: nn.Module):
    if isinstance(adapter_layer, nn.Conv2d):
        # adapter_layer.weight.fill_(0.0)
        adapter_layer.weight.normal_(0.0, 0.02)

        if adapter_layer.bias is not None:
            adapter_layer.bias.fill_(0.0)


def define_adapter(
    in_ch: int,
    out_ch: int,
    dim_adapter: int,
    stride: int = 1,
    groups: int = 1,
    bias: bool = False,
) -> nn.Module:
    """define residual adapters for the decoder

    Args:
        in_ch (int):
        out_ch (int):
        dim_adapter (int): the intermediate dimension of the adapter
        stride (int, optional): Defaults to 1.
        groups (int, optional): Defaults to 1. if the groups=in_ch,
            the adapter becomes a channel-wise multiplication layer used in [1].
        bias (bool, optional): Defaults to False.

    Returns:
        nn.Module: the adapter layer

    References:
    [1] https://openaccess.thecvf.com/content/CVPR2022/papers/Li_Cross-Domain_Few-Shot_Learning_With_Task-Specific_Adapters_CVPR_2022_paper.pdf
    """
    if dim_adapter == 0:
        if stride == 1:
            return ZeroLayer()

        elif stride == -1:
            return nn.Sequential(
                nn.Conv2d(
                    in_ch,
                    out_ch * 4,
                    kernel_size=1,
                    stride=1,
                    bias=bias,
                    groups=groups,
                ),
                nn.PixelShuffle(2),
                # above operation is only used for computing the shape.
                ZeroLayer(),
            )
        else:
            return nn.Sequential(
                nn.Conv2d(
                    in_ch,
                    out_ch,
                    kernel_size=1,
                    stride=stride,
                    bias=bias,
                    groups=groups,
                ),
                # above operation is only used for computing the shape.
                ZeroLayer(),
            )

    elif dim_adapter < 0:
        if stride == -1:
            # this implementation of subpixel conv. is done by ours.
            # original impl. by Cheng uses 3x3 conv.
            return nn.Sequential(
                nn.Conv2d(
                    in_ch,
                    out_ch * 4,
                    kernel_size=1,
                    stride=1,
                    bias=bias,
                    groups=groups,
                ),
                nn.PixelShuffle(2),
            )
        else:
            return nn.Conv2d(
                in_ch, out_ch, kernel_size=1, stride=stride, bias=bias, groups=groups
            )

    else:
        if stride == -1:
            # this implementation of subpixel conv. is done by ours.
            # original impl. by Cheng uses 3x3 conv.
            return nn.Sequential(
                nn.Conv2d(
                    in_ch,
                    dim_adapter * 4,
                    kernel_size=1,
                    bias=bias,
                    stride=stride,
                    groups=groups,
                ),
                nn.PixelShuffle(2),
                nn.Conv2d(dim_adapter, out_ch, kernel_size=1, bias=bias, groups=groups),
            )
        else:
            return nn.Sequential(
                nn.Conv2d(
                    in_ch,
                    dim_adapter,
                    kernel_size=1,
                    bias=bias,
                    stride=stride,
                    groups=groups,
                ),
                nn.Conv2d(dim_adapter, out_ch, kernel_size=1, bias=bias, groups=groups),
            )
