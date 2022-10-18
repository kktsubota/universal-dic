from compressai.layers import (
    AttentionBlock,
    subpel_conv3x3,
)
from compressai.models import Cheng2020Attention, CompressionModel
from compressai.models.utils import update_registered_buffers
from torch import nn

from aof.layers.res_blocks import (
    ResidualBlockAdapter,
    ResidualBlockUpsampleAdapter,
)


class Cheng2020AttnAdapter(Cheng2020Attention):
    def __init__(
        self,
        N=192,
        dim_1=[0, 0, 0, 0, 0, 0, 0],
        dim_2=None,
        groups: int = 1,
        connection: str = "serial",
        **kwargs
    ):
        super().__init__(N, **kwargs)
        if dim_2 is None:
            dim_2 = dim_1

        self.g_s = nn.Sequential(
            AttentionBlock(N),
            ResidualBlockAdapter(N, N, (dim_1[0], dim_2[0]), groups, connection),
            ResidualBlockUpsampleAdapter(
                N, N, 2, (dim_1[1], dim_2[1]), groups, connection
            ),
            ResidualBlockAdapter(N, N, (dim_1[2], dim_2[2]), groups, connection),
            ResidualBlockUpsampleAdapter(
                N, N, 2, (dim_1[3], dim_2[3]), groups, connection
            ),
            AttentionBlock(N),
            ResidualBlockAdapter(N, N, (dim_1[4], dim_2[4]), groups, connection),
            ResidualBlockUpsampleAdapter(
                N, N, 2, (dim_1[5], dim_2[5]), groups, connection
            ),
            ResidualBlockAdapter(N, N, (dim_1[6], dim_2[6]), groups, connection),
            subpel_conv3x3(N, 3, 2),
        )

    def load_state_dict(self, state_dict, strict: bool = True):
        # Dynamically update the entropy bottleneck buffers related to the CDFs
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        update_registered_buffers(
            self.entropy_bottleneck,
            "entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )
        super(CompressionModel, self).load_state_dict(state_dict, strict=strict)
