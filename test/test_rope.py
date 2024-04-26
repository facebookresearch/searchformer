# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import pytest


@pytest.mark.parametrize(
    "batch_size,steps,head_dim,n_heads,offset",
    [
        (1, 2, 4, 1, 0),
        (1, 5, 4, 1, 0),
        (1, 5, 4, 1, 1),
        (1, 5, 4, 1, 2),
        (1, 5, 4, 1, 3),
    ],
)
def test_rope_offset(
    batch_size: int, steps: int, head_dim: int, n_heads: int, offset: int
):
    """Insert a test for Rope offsets."""
    import torch

    from searchformer.transformer.rotary import RoPE

    rope = RoPE(dim=head_dim, max_seq_len=steps + 10)
    x = torch.ones((batch_size, steps, n_heads, head_dim))
    x_rope = rope(x)
    assert x_rope.shape == x.shape

    x_offset = torch.ones((batch_size, steps - offset, n_heads, head_dim))
    x_rope_offset = rope(x_offset, offset=offset)

    assert (x_rope[:, offset:] == x_rope_offset).all()
