# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
import torch.nn as nn
from torch import Tensor


class RoPE(torch.nn.Module):
    """Implementation of RoPE.

    The module holds the coefficients in a separate tensor that is re-used
    across the entire model.
    """

    def __init__(self, dim: int, max_seq_len: int, base: float = 10000.0):
        """Constructs a RoPE module

        Args:
            dim (int): Feature dimension that is embedded.
            max_seq_len (int): Maximum sequence length. This parameter limits
                the sequence length the whole Transformer model can process.
            base (float, optional): Base frequency. Defaults to 10000.0.
        """
        super().__init__()
        self.max_seq_len = max_seq_len
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        inv_freq = inv_freq[: dim // 2]
        self.seq_len_cached = 0

        t = torch.arange(self.max_seq_len).type_as(inv_freq)
        freqs = torch.outer(t, inv_freq).float()
        self.cos_param = nn.Parameter(
            freqs.cos()[None, :, None, :], requires_grad=False
        )
        self.sin_param = nn.Parameter(
            freqs.sin()[None, :, None, :], requires_grad=False
        )

    def forward(self, x: Tensor, offset: int = 0) -> Tensor:
        """Add embedding to tensor.

        Args:
            x (Tensor): Activation tensor.
            offset (int, optional): Offset since start of sequence. Defaults
                to 0.

        Returns:
            Tensor: Tensor with RoPE applied.
        """
        assert offset + x.shape[1] < self.max_seq_len

        cos_param = self.cos_param[:, offset : offset + x.shape[1], :, :]
        sin_param = self.sin_param[:, offset : offset + x.shape[1], :, :]

        x_0 = cos_param * x[..., ::2] - sin_param * x[..., 1::2]
        x_1 = sin_param * x[..., ::2] + cos_param * x[..., 1::2]
        x_out = torch.concat((x_0[..., None], x_1[..., None]), dim=-1)
        return x_out.reshape(*x_out.shape[:-2], -1).type_as(x)
