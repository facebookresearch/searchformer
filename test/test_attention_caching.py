# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import pytest


@pytest.mark.parametrize(
    "batch_size,steps,head_dim,n_heads,dim,use_rope",
    [
        (1, 2, 4, 1, 6, False),
        (1, 2, 4, 1, 6, True),
        (1, 5, 4, 1, 6, False),
        (1, 5, 4, 1, 6, True),
        (2, 5, 4, 1, 6, False),
        (2, 5, 4, 1, 6, True),
        (2, 5, 4, 2, 6, False),
        (2, 5, 4, 2, 6, True),
    ],
)
def test_inference_causal_attention(
    batch_size: int,
    steps: int,
    head_dim: int,
    n_heads: int,
    dim: int,
    use_rope: bool,
):
    from typing import Optional

    import torch

    from searchformer.transformer.model import Attention, KVCache
    from searchformer.transformer.rotary import RoPE

    max_seq_len = (steps + 100) // 100 * 100
    q = torch.rand((batch_size, steps, dim))
    k = torch.rand((batch_size, steps, dim))
    v = torch.rand((batch_size, steps, dim))
    rope: Optional[RoPE] = None
    if use_rope:
        rope = RoPE(dim=head_dim, max_seq_len=max_seq_len)

    attn = Attention(
        dim=dim,
        head_dim=head_dim,
        n_heads=n_heads,
        dropout=0.0,
        is_causal=True,
        rope=rope,
    )
    attn.eval()
    attn_output = attn(queries=q, keys=k, values=v)
    assert attn_output.shape == (batch_size, steps, dim)

    kv_cache = KVCache(
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        n_heads=n_heads,
        head_dim=head_dim,
    )
    for i in range(steps):
        attn_cached_output_i = attn(
            queries=q[:, i : i + 1, :],
            keys=k[:, i : i + 1, :],
            values=v[:, i : i + 1, :],
            cache=kv_cache,
        )
        assert attn_cached_output_i.shape == (batch_size, 1, dim)
        pred_diff = attn_output[:, i, :] - attn_cached_output_i[:, 0, :]
        assert pred_diff.abs().max() < 5e-7


@pytest.mark.parametrize(
    "batch_size,steps,head_dim,n_heads,dim,prompt_len",
    [
        (1, 2, 4, 1, 6, 7),
        (1, 5, 4, 1, 6, 7),
        (2, 2, 4, 1, 6, 7),
        (2, 5, 4, 1, 6, 7),
        (1, 5, 4, 2, 6, 7),
        (2, 5, 4, 2, 6, 7),
    ],
)
def test_inference_feature_conditioned_attention(
    batch_size: int,
    steps: int,
    head_dim: int,
    n_heads: int,
    dim: int,
    prompt_len: int,
):
    import torch

    from searchformer.transformer.model import Attention

    features = torch.rand((batch_size, prompt_len, dim))
    feature_mask = torch.ones((batch_size, 1, steps, prompt_len))
    q = torch.rand((batch_size, steps, dim))

    attn = Attention(
        dim=dim,
        head_dim=head_dim,
        n_heads=n_heads,
        dropout=0.0,
        is_causal=False,
    )
    attn.eval()
    attn_output = attn(
        queries=q,
        keys=features,
        values=features,
        mask=feature_mask,
    )
    assert attn_output.shape == (batch_size, steps, dim)

    for i in range(steps):
        attn_output_i = attn(
            queries=q[:, i : i + 1, :],
            keys=features,
            values=features,
            mask=feature_mask[:, :, i : i + 1, :],
        )
        assert attn_output_i.shape == (batch_size, 1, dim)
        pred_diff = attn_output_i[:, 0, :] - attn_output[:, i, :]
        assert pred_diff.abs().max() < 1e-6


@pytest.mark.parametrize(
    "batch_size,steps,n_heads,n_layers,dim,prompt_len",
    [
        (1, 2, 1, 1, 8, 7),
        (1, 2, 1, 1, 8, 7),
        (2, 2, 1, 1, 8, 7),
        (2, 2, 1, 1, 8, 7),
        (1, 5, 1, 1, 8, 7),
        (1, 5, 1, 1, 8, 7),
        (2, 5, 1, 1, 8, 7),
        (2, 5, 1, 1, 8, 7),
        (1, 5, 2, 1, 8, 7),
        (1, 5, 2, 1, 8, 7),
        (2, 5, 2, 1, 8, 7),
        (2, 5, 2, 1, 8, 7),
    ],
)
def test_inference_feature_conditioned_attention(
    batch_size: int,
    steps: int,
    n_heads: int,
    n_layers: int,
    dim: int,
    prompt_len: int,
):
    from typing import Optional

    import torch

    from searchformer.transformer.model import (
        FeatureConditionedBlock,
        HyperParams,
        KVCache,
    )
    from searchformer.transformer.rotary import RoPE

    hp = HyperParams(
        dim=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        dropout=0.0,
        multiple_of=2,
        vocab_size=14,
        output_size=14,
    )
    head_dim = hp.dim // hp.n_heads
    max_seq_len = (steps + 100) // 100 * 100
    rope = RoPE(dim=head_dim, max_seq_len=max_seq_len)
    block = FeatureConditionedBlock(hp, 0, rope)

    features = torch.rand((batch_size, prompt_len, dim))
    mask_features = torch.ones((batch_size, 1, steps, prompt_len))
    input = torch.rand((batch_size, steps, dim))
    output = block(
        input=input,
        features=features,
        mask_features=mask_features,
    )
    assert output.shape == (batch_size, steps, dim)

    kv_cache = KVCache(
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        n_heads=n_heads,
        head_dim=head_dim,
    )
    for i in range(steps):
        output_i = block(
            input=input[:, i : i + 1, :],
            features=features,
            mask_features=mask_features[:, :, i : i + 1, :],
            cache=kv_cache,
        )
        assert output_i.shape == (batch_size, 1, dim)
        pred_diff = output_i[:, 0, :] - output[:, i, :]
        assert pred_diff.abs().max() < 1e-6


@pytest.mark.parametrize(
    "batch_size,steps,n_heads,n_layers,dim,prompt_len",
    [
        (1, 2, 1, 1, 8, 7),
        (1, 2, 1, 1, 8, 7),
        (2, 2, 1, 1, 8, 7),
        (2, 2, 1, 1, 8, 7),
        (1, 5, 1, 1, 8, 7),
        (1, 5, 1, 1, 8, 7),
        (2, 5, 1, 1, 8, 7),
        (2, 5, 1, 1, 8, 7),
        (1, 5, 2, 1, 8, 7),
        (1, 5, 2, 1, 8, 7),
        (2, 5, 2, 1, 8, 7),
        (2, 5, 2, 1, 8, 7),
    ],
)
def test_inference_decoder(
    batch_size: int,
    steps: int,
    n_heads: int,
    n_layers: int,
    dim: int,
    prompt_len: int,
):
    import torch

    from searchformer.transformer.model import Decoder, HyperParams, KVCache
    from searchformer.transformer.rotary import RoPE

    hp = HyperParams(
        dim=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        dropout=0.0,
        multiple_of=2,
        vocab_size=14,
        output_size=14,
    )
    embedding = torch.nn.Embedding(hp.vocab_size, hp.dim)
    head_dim = hp.dim // hp.n_heads
    max_seq_len = (steps + 100) // 100 * 100
    rope = RoPE(dim=head_dim, max_seq_len=max_seq_len)
    decoder = Decoder(hp, embedding, rope)

    features = torch.rand((batch_size, prompt_len, dim))
    mask_features = torch.ones((batch_size, prompt_len))
    tokens = torch.zeros((batch_size, steps + 1)).long()
    logits = torch.zeros((batch_size, steps, hp.output_size))
    for i in range(steps):
        output = decoder(
            tokens=tokens[:, : i + 1],
            features=features,
            feature_mask=mask_features,
        )
        logits[:, i, :] = output[:, -1, :]
        tokens[:, i + 1] = output[:, -1, :].argmax(-1)

    cache = []
    for _ in range(hp.n_layers):
        cache.append(
            KVCache(
                batch_size=batch_size,
                max_seq_len=max_seq_len,
                n_heads=n_heads,
                head_dim=head_dim,
            )
        )
    tokens_cached = torch.zeros((batch_size, steps + 1)).long()
    logits_cached = torch.zeros((batch_size, steps, hp.output_size))
    for i in range(steps):
        output = decoder(
            tokens=tokens[:, i : i + 1],
            features=features,
            feature_mask=mask_features,
            cache=cache,
        )
        logits_cached[:, i, :] = output[:, -1, :]
        tokens_cached[:, i + 1] = output[:, -1, :].argmax(-1)

    assert (logits - logits_cached).abs().max() < 1e-6
    assert (tokens == tokens_cached).all()


@pytest.mark.parametrize(
    "n_heads,n_layers,dim,prompt_len",
    [
        (1, 1, 8, 7),
        (2, 1, 8, 7),
        (2, 3, 8, 7),
    ],
)
def test_rollout(
    n_heads: int,
    n_layers: int,
    dim: int,
    prompt_len: int,
):
    import torch

    from searchformer.transformer.model import EncoderDecoder, HyperParams

    hp_enc = HyperParams(
        dim=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        dropout=0.0,
        multiple_of=2,
        vocab_size=14,
        output_size=dim,
    )
    hp_dec = HyperParams(
        dim=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        dropout=0.0,
        multiple_of=2,
        vocab_size=14,
        output_size=14,
    )
    for _ in range(5):
        enc_dec = EncoderDecoder(hp_enc, hp_dec)
        prompt = torch.zeros(prompt_len).long()
        rollout_naive = enc_dec.rollout_naive(prompt, 0, 1, 100)
        rollout_cache = enc_dec.rollout(prompt, 0, 1, 100)
        assert (rollout_naive == rollout_cache).all()
