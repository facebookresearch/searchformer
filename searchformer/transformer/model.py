# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import json
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

from .rotary import RoPE
from .utils import sample_greedy

# from torch.profiler import record_function


class RMSLayerNorm(nn.Module):
    """Implements RMS layer normalization."""

    def __init__(
        self,
        normalized_shape: Tuple[int],
        eps: float = 1e-5,
        elementwise_affine: bool = True,
    ):
        super().__init__()
        assert type(normalized_shape) is tuple
        self.normalized_shape = torch.Size(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(Tensor(*normalized_shape))
        else:
            self.register_parameter("weight", None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.elementwise_affine:
            nn.init.ones_(self.weight)

    def forward(self, input: Tensor) -> Tensor:
        # layer norm should always be calculated in float32
        dims = tuple(i for i in range(-1, -len(self.normalized_shape) - 1, -1))
        variance = input.to(torch.float32).pow(2).mean(dims, keepdim=True)
        input = input * torch.rsqrt(variance + self.eps)

        if self.weight is None:
            return input

        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            input = input.to(self.weight.dtype)

        return self.weight * input

    def extra_repr(self) -> str:
        repr_str = f"normalized_shape={self.normalized_shape}"
        repr_str += f"eps={self.eps}"
        repr_str += f"elementwise_affine={self.elementwise_affine}"
        return repr_str


def repeat_kv(x: Tensor, n_rep: int) -> Tensor:
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


@dataclass
class HyperParams:
    """Dataclass holding architecture hyper parameters."""

    name: str = "default"
    dim: int = 1536
    n_layers: int = 48
    n_heads: int = 24
    dropout: float = 0.0
    multiple_of: int = 256
    norm_eps: float = 1e-5
    norm_affine: bool = True
    max_seq_len: int = 2**16
    vocab_size: int = -1  # defined later by tokenizer
    output_size: int = -1  # if -1 use vocab_size
    rope_freq: float = 10000.0

    @staticmethod
    def from_name(
        name: str, vocab_size: int, output_size: Optional[int] = None
    ) -> "HyperParams":
        if output_size is None:
            output_size = vocab_size
        args = {
            "enc-xl-2": dict(
                name="enc-xl-2",
                dim=1152,
                n_layers=16,
                n_heads=12,
                vocab_size=vocab_size,
                output_size=output_size,
            ),
            "dec-xl-2": dict(
                name="dec-xl-2",
                dim=1152,
                n_layers=16,
                n_heads=12,
                vocab_size=vocab_size,
                output_size=output_size,
            ),
            "enc-l": dict(
                name="enc-l",
                dim=768,
                n_layers=9,
                n_heads=4,
                vocab_size=vocab_size,
                output_size=output_size,
            ),
            "dec-l": dict(
                name="dec-l",
                dim=768,
                n_layers=9,
                n_heads=4,
                vocab_size=vocab_size,
                output_size=output_size,
            ),
            "enc-m-s": dict(
                name="enc-m-s",
                dim=384,
                n_layers=8,
                n_heads=4,
                vocab_size=vocab_size,
                output_size=output_size,
            ),
            "dec-m-s": dict(
                name="dec-m-s",
                dim=384,
                n_layers=8,
                n_heads=4,
                vocab_size=vocab_size,
                output_size=output_size,
            ),
            "enc-s": dict(
                name="enc-s",
                dim=192,
                n_layers=6,
                n_heads=3,
                vocab_size=vocab_size,
                output_size=output_size,
            ),
            "dec-s": dict(
                name="dec-s",
                dim=192,
                n_layers=6,
                n_heads=3,
                vocab_size=vocab_size,
                output_size=output_size,
            ),
        }[name]
        return HyperParams(**args)  # type: ignore

    def to_dict(self) -> Dict[str, Any]:
        return dict(
            name=self.name,
            dim=self.dim,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            dropout=self.dropout,
            multiple_of=self.multiple_of,
            norm_eps=self.norm_eps,
            norm_affine=self.norm_affine,
            vocab_size=self.vocab_size,
            output_size=self.output_size,
        )

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "HyperParams":
        return HyperParams(**d)

    @staticmethod
    def from_json(filename: str, override: Dict[str, Any]) -> "HyperParams":
        with open(filename, "r") as f:
            kvargs = {**json.load(f), **override}
            return HyperParams(**kvargs)


class KVCache:
    """Key-value cache used for faster inference."""

    def __init__(
        self,
        batch_size: int,
        max_seq_len: int,
        n_heads: int,
        head_dim: int,
    ):
        cache_shape = (batch_size, max_seq_len, n_heads, head_dim)
        self.k_cache = torch.zeros(cache_shape).float()
        self.v_cache = torch.zeros(cache_shape).float()
        self._curr_index = 0

    def __len__(self) -> int:
        return self._curr_index

    def __call__(self, k: Tensor, v: Tensor) -> Tuple[Tensor, Tensor]:
        assert k.shape == self.k_cache[:, :1, :, :].shape
        assert v.shape == self.v_cache[:, :1, :, :].shape

        self.k_cache[:, self._curr_index, :, :] = k[:, 0, :, :].detach()
        self.v_cache[:, self._curr_index, :, :] = v[:, 0, :, :].detach()

        self._curr_index += 1

        k_cache_res = self.k_cache[:, : self._curr_index, :, :]
        v_cache_res = self.v_cache[:, : self._curr_index, :, :]
        return k_cache_res, v_cache_res

    def pin_memory(self) -> "KVCache":
        self.k_cache = self.k_cache.pin_memory()
        self.v_cache = self.v_cache.pin_memory()
        return self

    def to(self, *params, **kvargs) -> "KVCache":
        self.k_cache = self.k_cache.to(*params, **kvargs)
        self.v_cache = self.v_cache.to(*params, **kvargs)
        return self

    def cuda(self) -> "KVCache":
        self.k_cache = self.k_cache.cuda()
        self.v_cache = self.v_cache.cuda()
        return self

    def cpu(self) -> "KVCache":
        self.k_cache = self.k_cache.cpu()
        self.v_cache = self.v_cache.cpu()
        return self


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        head_dim: int,
        n_heads: int,
        dropout: float,
        is_causal: bool = False,
        rope: Optional[RoPE] = None,
    ):
        """Constructs the attention module.

        Args:
            dim (int): Dimension of inputs.
            head_dim (int): Number of heads.
            n_heads (int): Dimension of each head.
            dropout (float): Dropout rate used on attention matrix.
            is_causal (bool): Flag to set if attention is causal.
            rope (Optional[RoPE]): RoPE embedding object. If None, then rope is
                not used.
        """
        super().__init__()

        self.n_heads = n_heads
        self.head_dim = head_dim
        self.dropout = dropout
        self.is_causal = is_causal
        self.rope = rope

        self.wq = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.n_heads * self.head_dim, dim, bias=False)

    def forward(
        self,
        queries: Tensor,
        keys: Tensor,
        values: Tensor,
        mask: Optional[Tensor] = None,
        cache: Optional[KVCache] = None,
    ) -> Tensor:
        """Forward pass through attention layer.

        Args:
            queries (Tensor): Shape `[batch_size, seq_len, feat_dim]`
            keys (Tensor): Shape `[batch_size, seq_len, feat_dim]`
            values (Tensor): Shape `[batch_size, seq_len, feat_dim]`
            mask (Optional[Tensor], optional): Shape
                `[batch_size, 1, seq_len, seq_len]`. Defaults to None.
            cache (Optional[KVCache], optional): Defaults to None.

        Returns:
            Tensor: _description_
        """
        assert mask is None or cache is None

        xq = self.wq(queries)  # (batch_size, seq_len, n_heads * head_dim)
        xk = self.wk(keys)  # (batch_size, seq_len, n_heads * head_dim)
        xv = self.wv(values)  # (batch_size, seq_len, n_heads * head_dim)

        bs = xq.shape[0]
        q_len = xq.shape[1]
        k_len = xk.shape[1]
        v_len = xv.shape[1]

        xq = xq.view(bs, q_len, self.n_heads, self.head_dim)
        xk = xk.view(bs, k_len, self.n_heads, self.head_dim)
        xv = xv.view(bs, v_len, self.n_heads, self.head_dim)

        if self.rope is not None:
            offset = 0
            if cache is not None:
                offset = len(cache)
            xq = self.rope(xq, offset=offset)
            xk = self.rope(xk, offset=offset)

        if cache is not None:
            xk, xv = cache(xk, xv)
            assert xq.shape[1] == 1

        xq = xq.transpose(1, 2)  # (bs, n_heads, q_len, head_dim)
        xk = xk.transpose(1, 2)  # (bs, n_heads, k_len, head_dim)
        xv = xv.transpose(1, 2)  # (bs, n_heads, v_len, head_dim)

        output = F.scaled_dot_product_attention(
            query=xq,
            key=xk,
            value=xv,
            attn_mask=mask,
            dropout_p=self.dropout,
            is_causal=self.is_causal and cache is None,
        )
        output = output.transpose(1, 2)  # (bs, q_len, n_heads, head_dim)
        output = output.contiguous().view(bs, q_len, -1)

        return self.wo(output)  # (batch_size, seq_len, n_heads * head_dim)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int = 256,
        dropout: float = 0.0,
    ):
        """Constructs a feed forward block.

        The hidden layer dimension the multiple of the value `multiple_of` that
        is closest to the `hidden_dim` value.

        Args:
            dim (int): Dimension of inputs.
            hidden_dim (int): Hidden layer dimension.
            multiple_of (int): Multiplier used for hidden layer size
                calculation.
            dropout (float): Dropout rate applied to layer.
            non_linearity (str): Non-linearity used in layer. Can be relu, gelu,
                or swiglu.
        """
        super().__init__()
        self.dropout = dropout

        # Hidden layer dim calculation.
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = hidden_dim + multiple_of - 1
        hidden_dim = multiple_of * (hidden_dim // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, input: Tensor) -> Tensor:
        hidden = self.w1(input)
        hidden = F.silu(hidden) * self.w3(input)
        return self.w2(hidden)


class SelfAttentionBlock(nn.Module):
    def __init__(
        self,
        hp: HyperParams,
        layer_id: int,
        rope: RoPE,
        is_causal: bool = False,
    ):
        """Constructs a self-attention block.

        Args:
            hp (HyperParams): Architecture hyper parameters.
            layer_id (int): Layer id.
            rope (RoPE): RoPE module used for position embeddings.
            is_causal (bool, optional): If True, then uses causal attention
                maps, otherwise the full attention map is constructed.
                Defaults to False.
        """
        super().__init__()
        logging.debug(f"Creating block: n_heads={hp.n_heads}, dim={hp.dim}.")
        assert hp.n_heads is not None
        assert hp.dim % hp.n_heads == 0

        self.hp = hp
        self.head_dim = hp.dim // hp.n_heads
        self.layer_id = layer_id
        self.attention = Attention(
            dim=hp.dim,
            head_dim=self.head_dim,
            n_heads=hp.n_heads,
            dropout=hp.dropout,
            rope=rope,
            is_causal=is_causal,
        )
        self.feed_forward = FeedForward(
            dim=hp.dim,
            hidden_dim=4 * hp.dim,
            multiple_of=hp.multiple_of,
            dropout=hp.dropout,
        )
        self.attention_norm = RMSLayerNorm(
            normalized_shape=(hp.dim,),
            eps=hp.norm_eps,
            elementwise_affine=hp.norm_affine,
        )
        self.ffn_norm = RMSLayerNorm(
            normalized_shape=(hp.dim,),
            eps=hp.norm_eps,
            elementwise_affine=hp.norm_affine,
        )

    def forward(
        self,
        input: Tensor,
    ) -> Tensor:
        """Forward pass through self-attention block.

        Args:
            input (Tensor): Feature tensor of shape `[batch_size, seq_len, feat_dim]`

        Returns:
            Tensor: Output tensor of shape `[batch_size, seq_len, feat_dim]`
        """
        input_norm = self.attention_norm(input)
        h = self.attention(input_norm, input_norm, input_norm)
        h = h + input
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class FeatureConditionedBlock(nn.Module):
    """Decoder block used for the encoder-decoder architecture.

    This block accepts the encoder's output feature tensor as input.
    """

    def __init__(self, hp: HyperParams, layer_id: int, rope: RoPE):
        """Constructs feature-conditions network block used in the decoder
        part of the architecture.

        Args:
            hp (HyperParams): Architecture hyper parameter.
            layer_id (int): Layer id.
            rope (RoPE): RoPE module used for position embeddings.
        """
        super().__init__()
        logging.debug(f"Creating block: n_heads={hp.n_heads}, dim={hp.dim}.")
        assert hp.n_heads is not None
        assert hp.dim % hp.n_heads == 0

        self.hp = hp
        self.head_dim = hp.dim // hp.n_heads
        self.layer_id = layer_id

        self.attention = Attention(
            dim=hp.dim,
            head_dim=self.head_dim,
            n_heads=hp.n_heads,
            dropout=hp.dropout,
            rope=rope,
            is_causal=True,
        )
        self.attention_feat = Attention(
            dim=hp.dim,
            head_dim=self.head_dim,
            n_heads=hp.n_heads,
            dropout=hp.dropout,
            is_causal=False,
            # rope=rope,
        )
        self.feed_forward = FeedForward(
            dim=hp.dim,
            hidden_dim=4 * hp.dim,
            multiple_of=hp.multiple_of,
            dropout=hp.dropout,
        )
        self.attention_norm = RMSLayerNorm(
            normalized_shape=(hp.dim,),
            eps=hp.norm_eps,
            elementwise_affine=hp.norm_affine,
        )
        self.attention_feat_norm = RMSLayerNorm(
            normalized_shape=(hp.dim,),
            eps=hp.norm_eps,
            elementwise_affine=hp.norm_affine,
        )
        self.ffn_norm = RMSLayerNorm(
            normalized_shape=(hp.dim,),
            eps=hp.norm_eps,
            elementwise_affine=hp.norm_affine,
        )

    def forward(
        self,
        input: Tensor,
        features: Tensor,
        mask_features: Optional[Tensor] = None,
        cache: Optional[KVCache] = None,
    ) -> Tensor:
        """Forward pass through feature conditioned transformer block.

        Args:
            input (Tensor): Input feature tensor (either coming from another
                decoder block or from module embedding tokens in feature
                space.)
            features (Tensor): Encoder network output of shape
                `[batch_size, max_prompt_len, feat_dim]`.
            mask_features (Optional[Tensor], optional): Encoder network
                output length mask of shapce `[batch_size, max_prompt_len]`.
                This tensor specifies the actual length of each prompt
                sequence contained in the mini batch. Defaults to None.
            cache (Optional[KVCache], optional): KV cache module used during
                inference. Defaults to None.

        Returns:
            Tensor: Output features.
        """
        input_norm = self.attention_norm(input)
        h = self.attention(
            queries=input_norm,
            keys=input_norm,
            values=input_norm,
            cache=cache,
        )
        h = h + input
        h_enc = self.attention_feat(
            queries=self.attention_feat_norm(h),
            keys=features,
            values=features,
            mask=mask_features,
        )
        h = h + h_enc
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


def sequence_mask_to_attention_mask(mask: Tensor) -> Tensor:
    bs, max_seq_len = mask.shape
    seq_len = mask.sum(-1)
    mask_attn = torch.full(
        size=(bs, 1, max_seq_len, max_seq_len),
        fill_value=float(0.0),
        device=mask.device,
    )
    for i, j in enumerate(seq_len):
        mask_attn[i, 0, :j, :j] = 1.0
    return mask_attn


class Transformer(nn.Module):
    """Decoder-style transformer model."""

    def __init__(
        self,
        embedding: nn.Module,
        rope: RoPE,
        layers: nn.ModuleList,
        output: nn.Module,
    ):
        """Constructs a decoder-style transformer model.

        Args:
            embedding (nn.Module): Embedding layer processing token sequences
                as input.
            rope (RoPE): RoPE module used for applying position embeddings.
                The same module instance is used across the entire
                architecture.
            layers (nn.ModuleList): List of transformer block layers.
            output (nn.Module): Output layer mapping features to logits. This
                layer outputs a tensor of shape
                `[batch_size, seq_len, vocab_size]`.
        """
        super().__init__()
        self.embedding = embedding
        self.rope = rope
        self.layers = layers
        self.output = output

    def forward(
        self,
        tokens: Tensor,
        features: Optional[Tensor] = None,
        feature_mask: Optional[Tensor] = None,
        cache: Optional[List[KVCache]] = None,
    ) -> Tensor:
        """Forward pass through transformer architecture.

        Args:
            tokens (Tensor): Tokens tensor of shape `[batch_size, seq_len]`.
            features (Optional[Tensor], optional): Feature tensor of shape
                `[batch_size, seq_len, feat_dim]`. Defaults to None
                (no conditioning).
            feature_mask (Optional[Tensor], optional): _description_. Sequence
                length mask tensor of features of shape
                `[batch_size, seq_len]`. Defaults to None (no
                conditioning).

        Returns:
            Tensor: Logit tensor of shape `[batch_size, seq_len, vocab_size]`.
        """
        h = self.embedding(tokens)  # (batch_size, seq_len, feat_dim)
        if feature_mask is not None:
            feature_mask = feature_mask.float()
            feature_mask = 1.0 - feature_mask[:, None, None, :]
            feature_mask[feature_mask == 1.0] = float("-inf")  # type: ignore

        if features is None:
            for layer in self.layers:
                h = layer(input=h)
        else:
            cache_list: Sequence[Union[None, KVCache]] = []
            if cache is None:
                cache_list = [None] * len(self.layers)
            else:
                cache_list = cache
            for layer, kv_cache in zip(self.layers, cache_list):
                h = layer(
                    input=h,
                    features=features,
                    mask_features=feature_mask,
                    cache=kv_cache,
                )
        logits = self.output(h).float()
        return logits


class Encoder(Transformer):
    """Encoder network."""

    def __init__(self, hp: HyperParams, embedding: nn.Module, rope: RoPE):
        layers = nn.ModuleList(
            SelfAttentionBlock(hp=hp, layer_id=id, rope=rope)
            for id in range(hp.n_layers)
        )
        output = nn.Sequential(
            RMSLayerNorm(
                normalized_shape=(hp.dim,),
                eps=hp.norm_eps,
                elementwise_affine=hp.norm_affine,
            ),
            nn.Linear(hp.dim, hp.output_size, bias=False),
        )
        super().__init__(
            embedding=embedding,
            rope=rope,
            layers=layers,
            output=output,
        )


class Decoder(Transformer):
    """Decoder network."""

    def __init__(self, hp: HyperParams, embedding: nn.Module, rope: RoPE):
        layers = nn.ModuleList(
            FeatureConditionedBlock(hp, layer_id=id, rope=rope)
            for id in range(hp.n_layers)
        )
        output = nn.Sequential(
            RMSLayerNorm(
                normalized_shape=(hp.dim,),
                eps=hp.norm_eps,
                elementwise_affine=hp.norm_affine,
            ),
            nn.Linear(hp.dim, hp.output_size, bias=False),
        )
        super().__init__(
            embedding=embedding,
            rope=rope,
            layers=layers,
            output=output,
        )


class EncoderDecoder(nn.Module):
    """Encoder-decoder network used in all experiments."""

    def __init__(self, enc_hp: HyperParams, dec_hp: HyperParams):
        """Constructs the encoder-decoder network for all experiments.

        Args:
            enc_hp (HyperParams): Encoder archictecture hyper parameters.
            dec_hp (HyperParams): Decoder archictecture hyper parameters.
        """
        super().__init__()
        embedding_enc = nn.Embedding(enc_hp.vocab_size, enc_hp.dim)
        embedding_dec = nn.Embedding(dec_hp.vocab_size, dec_hp.dim)
        rope_enc = RoPE(
            dim=enc_hp.dim // enc_hp.n_heads,
            max_seq_len=enc_hp.max_seq_len,
            base=enc_hp.rope_freq,
        )
        rope_dec = RoPE(
            dim=dec_hp.dim // dec_hp.n_heads,
            max_seq_len=dec_hp.max_seq_len,
            base=dec_hp.rope_freq,
        )
        self.encoder = Encoder(enc_hp, embedding_enc, rope_enc)
        self.decoder = Decoder(dec_hp, embedding_dec, rope_dec)

    def forward(
        self,
        prompt: Tensor,
        prompt_mask: Tensor,
        trace: Tensor,
    ) -> Tensor:
        features = self.encoder(prompt)
        output = self.decoder(trace, features, prompt_mask)
        return output

    @torch.inference_mode()
    def rollout(
        self,
        prompt: Tensor,
        bos_idx: int,
        eos_idx: int,
        max_rollout_len: int,
        sample_fn: Optional[
            Callable[
                [
                    Tensor,
                ],
                Tensor,
            ]
        ] = None,
        num_samples: int = 1,
        prefix_seq: Optional[List[int]] = None,
    ) -> Tensor:
        """Performs a cached rollout. `prefix_seq` is used to cue rollout generation
        with more than a `bos` token. If None, no additional cue is given.

        Args:
            prompt (Tensor): Prompt tensor.
            bos_idx (int): Bos token index.
            eos_idx (int): Eos token index.
            max_rollout_len (int): Maximum rollout length.
            sample_fn (Optional[ Callable[ [ Tensor, ], Tensor, ] ], optional):
                Function used to sample from logits. Defaults to None.
            num_samples (int, optional): Number of token sequences that are generated
                for the provided prompt. Defaults to 1.
            prefix_seq (Optional[List[int]], optional): Any prefix sequence that is
                feed into the decoder to start sequence generation. Defaults to None.

        Returns:
            Tensor: Long int tensor containing the generated sequences.
        """
        if sample_fn is None:
            sample_fn = sample_greedy
        if prefix_seq is None:
            prefix_seq = []

        prompt = prompt.reshape(1, -1).repeat(num_samples, 1)
        prompt_mask = torch.ones_like(prompt)
        trace = torch.ones((num_samples, max_rollout_len)).long() * bos_idx
        for i, tok_idx in enumerate(prefix_seq):
            trace[:, i + 1] = tok_idx
        trace = trace.to(prompt.device)

        features = self.encoder(prompt)
        cache = []
        for layer in self.decoder.layers:
            assert isinstance(layer, FeatureConditionedBlock)
            cache.append(
                KVCache(
                    batch_size=prompt.shape[0],
                    max_seq_len=max_rollout_len,
                    n_heads=layer.hp.n_heads,
                    head_dim=layer.head_dim,
                ).to(prompt.device)
            )

        i = 0
        for i in range(1, max_rollout_len):
            logits = self.decoder(
                tokens=trace[:, i - 1 : i],
                features=features,
                feature_mask=prompt_mask,
                cache=cache,
            )
            if i > len(prefix_seq):
                next_token_idx = sample_fn(logits[:, -1])
                trace[:, i] = next_token_idx
            logits = None

            eos_mask = (trace == eos_idx).any(-1)
            if i % 200 == 0 and i > 0:
                num_eos = eos_mask.sum().item()
                logging.debug(f"Rollout {i} steps, {num_eos} seq. complete.")
            if eos_mask.all():
                break
        logging.info(f"Rollout length: {i + 1}")
        return trace[:, : i + 1]

    @torch.inference_mode()
    def rollout_naive(
        self,
        prompt: Tensor,
        bos_idx: int,
        eos_idx: int,
        max_rollout_len: int,
        sample_fn: Optional[
            Callable[
                [
                    Tensor,
                ],
                Tensor,
            ]
        ] = None,
    ) -> Tensor:
        """Naive implementation of sequence generation that does not use key-value caching.

        Args:
            prompt (Tensor): Prompt tensor.
            bos_idx (int): Bos token index.
            eos_idx (int): Eos token index.
            max_rollout_len (int): Maximum rollout length.
            sample_fn (Optional[ Callable[ [ Tensor, ], Tensor, ] ], optional):
                Function used to sample from logits. Defaults to None.

        Returns:
            Tensor: Long int tensor containing the generated sequences.
        """
        if sample_fn is None:
            sample_fn = sample_greedy

        prompt = prompt.reshape(1, -1)
        prompt_mask = torch.ones_like(prompt)
        trace = torch.ones((1, max_rollout_len)).long() * bos_idx
        trace = trace.to(prompt.device)

        features = self.encoder(prompt)
        i = 0
        for i in range(1, max_rollout_len):
            logits = self.decoder(trace[:, :i], features, prompt_mask)
            next_token_idx = sample_fn(logits[0, -1])
            trace[:, i] = next_token_idx
            logits = None
            if next_token_idx == eos_idx:
                break
        return trace[:, : i + 1].reshape(-1)


@dataclass
class EncoderDecoderConfig:
    """Dataclass holding all architecture hyper parameter."""

    encoder: HyperParams
    decoder: HyperParams

    @staticmethod
    def from_name(
        enc_name: str, dec_name: str, vocab_size: int
    ) -> "EncoderDecoderConfig":
        dec_conf = HyperParams.from_name(dec_name, vocab_size)
        enc_conf = HyperParams.from_name(enc_name, vocab_size, dec_conf.dim)
        return EncoderDecoderConfig(encoder=enc_conf, decoder=dec_conf)

    def to_dict(self) -> Dict[str, Any]:
        return dict(
            encoder=self.encoder.to_dict(),
            decoder=self.decoder.to_dict(),
        )

    @staticmethod
    def from_dict(d: Dict[str, Any]):
        return EncoderDecoderConfig(
            encoder=HyperParams.from_dict(d["encoder"]),
            decoder=HyperParams.from_dict(d["decoder"]),
        )

    def construct_model(self) -> EncoderDecoder:
        return EncoderDecoder(self.encoder, self.decoder)
