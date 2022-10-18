from compressai.layers import (
    ResidualBlock,
    ResidualBlockUpsample,
)
import torch

from .adapters import define_adapter, init_adapter_layer


class ResidualBlockAdapter(ResidualBlock):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        dim_adapter=0,
        groups: int = 1,
        connection: str = "serial",
    ):
        super().__init__(in_ch, out_ch)
        self.connection: str = connection

        if isinstance(dim_adapter, int):
            dim_1, dim_2 = dim_adapter, dim_adapter
        else:
            dim_1, dim_2 = dim_adapter
        self.adapter_1 = define_adapter(in_ch, out_ch, dim_1, groups=groups)
        self.adapter_2 = define_adapter(out_ch, out_ch, dim_2, groups=groups)

        self.adapter_1.apply(init_adapter_layer)
        self.adapter_2.apply(init_adapter_layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        if self.connection == "parallel":
            out = self.conv1(x) + self.adapter_1(x)
        elif self.connection == "serial":
            out = self.conv1(x)
            out = out + self.adapter_1(out)
        elif self.connection == "serial wo id":
            out = self.conv1(x)
            out = self.adapter_1(out)
        else:
            raise NotImplementedError

        out = self.leaky_relu(out)
        if self.connection == "parallel":
            out = self.conv2(out) + self.adapter_2(out)
        elif self.connection == "serial":
            out = self.conv2(out)
            out = out + self.adapter_2(out)
        elif self.connection == "serial wo id":
            out = self.conv2(out)
            out = self.adapter_2(out)
        else:
            raise NotImplementedError
        out = self.leaky_relu(out)

        if self.skip is not None:
            identity = self.skip(x)

        out = out + identity
        return out


class ResidualBlockUpsampleAdapter(ResidualBlockUpsample):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        upsample: int = 2,
        dim_adapter=0,
        groups: int = 1,
        connection: str = "serial",
    ):
        super().__init__(in_ch, out_ch, upsample)
        self.connection: str = connection

        if isinstance(dim_adapter, int):
            dim_1, dim_2 = dim_adapter, dim_adapter
        else:
            dim_1, dim_2 = dim_adapter

        stride_adpt = -1 if connection == "parallel" else 1
        self.adapter_1 = define_adapter(
            in_ch, out_ch, dim_1, groups=groups, stride=stride_adpt
        )
        self.adapter_2 = define_adapter(out_ch, out_ch, dim_2, groups=groups)

        self.adapter_1.apply(init_adapter_layer)
        self.adapter_2.apply(init_adapter_layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        if self.connection == "parallel":
            out = self.subpel_conv(x) + self.adapter_1(x)
        elif self.connection == "serial":
            out = self.subpel_conv(x)
            out = out + self.adapter_1(out)
        elif self.connection == "serial wo id":
            out = self.subpel_conv(x)
            out = self.adapter_1(out)
        else:
            raise NotImplementedError
        out = self.leaky_relu(out)
        if self.connection == "parallel":
            out = self.conv(out) + self.adapter_2(out)
        elif self.connection == "serial":
            out = self.conv(out)
            out = out + self.adapter_2(out)
        elif self.connection == "serial no id":
            out = self.conv(out)
            out = self.adapter_2(out)
        else:
            raise NotImplementedError
        out = self.igdn(out)
        identity = self.upsample(x)
        out = out + identity
        return out
