import warnings

from compressai.models import (
    ScaleHyperprior,
    MeanScaleHyperprior,
    JointAutoregressiveHierarchicalPriors,
)
from compressai.zoo import (
    mbt2018_mean,
    mbt2018,
    cheng2020_attn,
)
from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai.ops import ste_round
import torch
from torch import nn
from torch.nn import functional as F

from aof.models.wacnn import WACNN


def get_model(model_name: str, quality: int, model_path=None) -> nn.Module:
    if model_name == "mbt2018-mean":
        model = mbt2018_mean(quality=quality, pretrained=model_path is None)

    elif model_name == "mbt2018":
        model = mbt2018(quality=quality, pretrained=model_path is None)

    elif model_name == "cheng2020-attn":
        model = cheng2020_attn(quality=quality, pretrained=model_path is None)

    elif model_name == "wacnn":
        model = WACNN()
        # equivalent with pretrained=True in other models
        if model_path is None:
            pretrained_model_path = f"weights/q{quality}/model.pth"
            state_dict = torch.load(pretrained_model_path)
            model.load_state_dict(state_dict)
            model.update()

    else:
        raise NotImplementedError

    if model_path is not None:
        model.load_state_dict(torch.load(model_path))
    return model


def forward_enc(self: nn.Module, x: torch.Tensor) -> dict:
    y = self.g_a(x)
    if type(self) == ScaleHyperprior:
        z = self.h_a(torch.abs(y))

    elif type(self) == MeanScaleHyperprior:
        z = self.h_a(y)

    elif isinstance(self, JointAutoregressiveHierarchicalPriors):
        z = self.h_a(y)

    elif isinstance(self, WACNN):
        z = self.h_a(y)

    else:
        raise NotImplementedError
    return {
        "y": y,
        "z": z,
    }


def forward_dec(
    self: nn.Module, y: torch.Tensor, z: torch.Tensor, recompute_y_hat: bool = True
) -> dict:
    if type(self) == ScaleHyperprior:
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        scales_hat = self.h_s(z_hat)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat)
    elif type(self) == MeanScaleHyperprior:
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
    elif isinstance(self, JointAutoregressiveHierarchicalPriors):
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        params = self.h_s(z_hat)

        y_hat = self.gaussian_conditional.quantize(
            y, "noise" if self.training else "dequantize"
        )
        ctx_params = self.context_prediction(y_hat)
        gaussian_params = self.entropy_parameters(
            torch.cat((params, ctx_params), dim=1)
        )
        scales_hat, means_hat = gaussian_params.chunk(2, 1)

        # NOTE: In usual, we do not need to use means and scales
        # because y_hat is quantized with additive uniform noise.
        # Here, we recompute y_hat because we used SGA quantization.
        if recompute_y_hat:
            y_hat, y_likelihoods = self.gaussian_conditional(
                y, scales_hat, means=means_hat
            )
        else:
            _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)

    elif isinstance(self, WACNN):
        y_shape = y.shape[2:]
        _, z_likelihoods = self.entropy_bottleneck(z)

        z_offset = self.entropy_bottleneck._get_medians()
        z_tmp = z - z_offset
        z_hat = ste_round(z_tmp) + z_offset

        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices = []
        y_likelihood = []

        for slice_index, y_slice in enumerate(y_slices):
            support_slices = (
                y_hat_slices
                if self.max_support_slices < 0
                else y_hat_slices[: self.max_support_slices]
            )
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, : y_shape[0], : y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, : y_shape[0], : y_shape[1]]

            # original implementation
            # _, y_slice_likelihood = self.gaussian_conditional(y_slice, scale, mu)
            # y_hat_slice = ste_round(y_slice - mu) + mu

            # add .clone() to y_slice to avoid runtime error
            y_hat_slice, y_slice_likelihood = self.gaussian_conditional(
                y_slice.clone(), scale, mu
            )
            y_likelihood.append(y_slice_likelihood)

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)
        y_likelihoods = torch.cat(y_likelihood, dim=1)

    else:
        raise NotImplementedError

    x_hat = self.g_s(y_hat)
    x_hat = x_hat.clamp(0, 1)
    return {
        "y_hat": y_hat,
        "z_hat": z_hat,
        "x_hat": x_hat,
        "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
    }


def encode_latent(self, y: torch.Tensor, z: torch.Tensor) -> dict:
    if type(self) == ScaleHyperprior:
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        scales_hat = self.h_s(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes)

    elif type(self) == MeanScaleHyperprior:
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes, means=means_hat)

    elif isinstance(self, JointAutoregressiveHierarchicalPriors):
        if next(self.parameters()).device != torch.device("cpu"):
            warnings.warn(
                "Inference on GPU is not recommended for the autoregressive "
                "models (the entropy coder is run sequentially on CPU)."
            )
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        params = self.h_s(z_hat)

        s = 4  # scaling factor between z and y
        kernel_size = 5  # context prediction kernel size
        padding = (kernel_size - 1) // 2

        y_height = z_hat.size(2) * s
        y_width = z_hat.size(3) * s

        y_hat = F.pad(y, (padding, padding, padding, padding))

        y_strings = []
        for i in range(y.size(0)):
            string = self._compress_ar(
                y_hat[i : i + 1],
                params[i : i + 1],
                y_height,
                y_width,
                kernel_size,
                padding,
            )
            y_strings.append(string)

    elif isinstance(self, WACNN):
        y_shape = y.shape[2:]

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices = []
        y_scales = []
        y_means = []

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []
        y_strings = []

        for slice_index, y_slice in enumerate(y_slices):
            support_slices = (
                y_hat_slices
                if self.max_support_slices < 0
                else y_hat_slices[: self.max_support_slices]
            )

            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, : y_shape[0], : y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, : y_shape[0], : y_shape[1]]

            index = self.gaussian_conditional.build_indexes(scale)
            y_q_slice = self.gaussian_conditional.quantize(y_slice, "symbols", mu)
            y_hat_slice = y_q_slice + mu

            symbols_list.extend(y_q_slice.reshape(-1).tolist())
            indexes_list.extend(index.reshape(-1).tolist())

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)
            y_scales.append(scale)
            y_means.append(mu)

        encoder.encode_with_indexes(
            symbols_list, indexes_list, cdf, cdf_lengths, offsets
        )
        y_string = encoder.flush()
        y_strings.append(y_string)

        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    else:
        raise NotImplementedError

    return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}


def decode_latent(self: nn.Module, strings, shape) -> torch.Tensor:
    if type(self) == ScaleHyperprior:
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        scales_hat = self.h_s(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(strings[0], indexes, z_hat.dtype)
    elif type(self) == MeanScaleHyperprior:
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(
            strings[0], indexes, means=means_hat
        )
    elif isinstance(self, JointAutoregressiveHierarchicalPriors):
        assert isinstance(strings, list) and len(strings) == 2

        if next(self.parameters()).device != torch.device("cpu"):
            warnings.warn(
                "Inference on GPU is not recommended for the autoregressive "
                "models (the entropy coder is run sequentially on CPU)."
            )

        # FIXME: we don't respect the default entropy coder and directly call the
        # range ANS decoder

        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        params = self.h_s(z_hat)

        s = 4  # scaling factor between z and y
        kernel_size = 5  # context prediction kernel size
        padding = (kernel_size - 1) // 2

        y_height = z_hat.size(2) * s
        y_width = z_hat.size(3) * s

        # initialize y_hat to zeros, and pad it so we can directly work with
        # sub-tensors of size (N, C, kernel size, kernel_size)
        y_hat = torch.zeros(
            (z_hat.size(0), self.M, y_height + 2 * padding, y_width + 2 * padding),
            device=z_hat.device,
        )

        for i, y_string in enumerate(strings[0]):
            self._decompress_ar(
                y_string,
                y_hat[i : i + 1],
                params[i : i + 1],
                y_height,
                y_width,
                kernel_size,
                padding,
            )

        y_hat = F.pad(y_hat, (-padding, -padding, -padding, -padding))

    elif isinstance(self, WACNN):
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_shape = [z_hat.shape[2] * 4, z_hat.shape[3] * 4]

        y_string = strings[0][0]

        y_hat_slices = []
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        decoder = RansDecoder()
        decoder.set_stream(y_string)

        for slice_index in range(self.num_slices):
            support_slices = (
                y_hat_slices
                if self.max_support_slices < 0
                else y_hat_slices[: self.max_support_slices]
            )
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, : y_shape[0], : y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, : y_shape[0], : y_shape[1]]

            index = self.gaussian_conditional.build_indexes(scale)

            rv = decoder.decode_stream(
                index.reshape(-1).tolist(), cdf, cdf_lengths, offsets
            )
            rv = torch.Tensor(rv).reshape(1, -1, y_shape[0], y_shape[1])
            y_hat_slice = self.gaussian_conditional.dequantize(rv, mu)

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)

    else:
        raise NotImplementedError

    return y_hat
