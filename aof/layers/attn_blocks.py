from compressai.layers import conv3x3
import torch
import torch.nn as nn
from timm.models.layers import to_2tuple

from .adapters import define_adapter, init_adapter_layer

from .win_attn import WinBasedAttention, WindowAttention


# copy from https://github.com/Googolxx/STF/blob/b923265869269347faf2e848341582e9017c4f81/compressai/layers/layers.py
# Copyright (c) 2022 @Googolxx Licensed under Apache License 2.0
def conv1x1(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    """1x1 convolution."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride)


# copy from https://github.com/Googolxx/STF/blob/b923265869269347faf2e848341582e9017c4f81/compressai/layers/layers.py
# Copyright (c) 2022 @Googolxx Licensed under Apache License 2.0
class Win_noShift_Attention(nn.Module):
    """Window-based self-attention module."""

    def __init__(self, dim, num_heads=8, window_size=8, shift_size=0):
        super().__init__()
        N = dim

        class ResidualUnit(nn.Module):
            """Simple residual unit."""

            def __init__(self):
                super().__init__()
                self.conv = nn.Sequential(
                    conv1x1(N, N // 2),
                    nn.GELU(),
                    conv3x3(N // 2, N // 2),
                    nn.GELU(),
                    conv1x1(N // 2, N),
                )
                self.relu = nn.GELU()

            def forward(self, x):
                identity = x
                out = self.conv(x)
                out += identity
                out = self.relu(out)
                return out

        self.conv_a = nn.Sequential(ResidualUnit(), ResidualUnit(), ResidualUnit())

        self.conv_b = nn.Sequential(
            WinBasedAttention(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=shift_size,
            ),
            ResidualUnit(),
            ResidualUnit(),
            ResidualUnit(),
            conv1x1(N, N),
        )

    def forward(self, x):
        identity = x
        a = self.conv_a(x)
        b = self.conv_b(x)
        out = a * torch.sigmoid(b)
        out += identity
        return out


class WindowAttentionAdapter(WindowAttention):
    def __init__(
        self,
        dim=192,
        window_size=(8, 8),
        num_heads=8,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0,
        proj_drop=0,
        dim_adapter=0,
        groups=1,
    ):
        super().__init__(
            dim, window_size, num_heads, qkv_bias, qk_scale, attn_drop, proj_drop
        )
        self.adapter = define_adapter(
            dim, dim, dim_adapter=dim_adapter, groups=groups, bias=False
        )
        self.adapter.apply(init_adapter_layer)

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        y = super().forward(x, mask)
        # cf. VL-Adapter without GeLU activation
        return y + self.adapter(y)


class WinBaseAttentionAdapter(WinBasedAttention):
    def __init__(
        self,
        dim=192,
        num_heads=8,
        window_size=8,
        shift_size=0,
        qkv_bias=True,
        qk_scale=None,
        drop=0,
        attn_drop=0,
        drop_path=0,
        dim_adapter=0,
        groups=1,
        position="attn",
    ):
        super().__init__(
            dim,
            num_heads,
            window_size,
            shift_size,
            qkv_bias,
            qk_scale,
            drop,
            attn_drop,
            drop_path,
        )
        self.position = position
        if self.position == "attn":
            self.adapter = define_adapter(
                dim, dim, dim_adapter=dim_adapter, groups=groups, bias=False
            )
            self.adapter.apply(init_adapter_layer)
        elif self.position == "attnattn":
            self.attn = WindowAttentionAdapter(
                dim,
                window_size=to_2tuple(self.window_size),
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
                dim_adapter=dim_adapter,
                groups=groups,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.position == "attn":
            shortcut = x
            y = super().forward(x)
            return self.adapter(y - shortcut) + shortcut
        else:
            return super().forward(x)


class Win_noShift_Attention_Adapter(Win_noShift_Attention):
    def __init__(
        self,
        dim,
        num_heads=8,
        window_size=8,
        shift_size=0,
        dim_adapter: int = 1,
        groups: int = 1,
        position: str = "last",
    ):
        """Win_noShift_Attention with adapters.

        Args:
            dim_adapter (int): dimension for the intermediate feature of the adapter.
            groups (int): number of groups for the adapter. Default: 1. if groups=dim, the adapter become channel-wise multiplication.
        """
        super().__init__(dim, num_heads, window_size, shift_size)
        self.position = position
        if self.position == "last":
            self.adapter = define_adapter(
                dim, dim, dim_adapter=dim_adapter, groups=groups, bias=False
            )
            self.adapter.apply(init_adapter_layer)
        elif self.position in {"attn", "attnattn"}:
            self.conv_b[0] = WinBaseAttentionAdapter(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=shift_size,
                dim_adapter=dim_adapter,
                groups=groups,
                position=self.position,
            )

    def forward(self, x):
        identity = x
        a = self.conv_a(x)
        b = self.conv_b(x)
        out = a * torch.sigmoid(b)

        if self.position == "last":
            # modify output by adapters
            out = out + self.adapter(out)

        out += identity
        return out
