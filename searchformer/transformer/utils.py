# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import functools
from typing import Any, Callable, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def num_model_parameters(model: nn.Module) -> int:
    """Counts the number of model parameters.

    Args:
        model (nn.Module): Model.

    Returns:
        int: Number of parameters.
    """
    num_params = 0
    for p in model.parameters():
        num_params += p.numel()
    return num_params


def sample_greedy(logits: Tensor) -> Tensor:
    return logits.argmax(-1)


def sample_probability(logits: Tensor, temp: float = 1.0) -> Tensor:
    probs = F.softmax(temp * logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)[:, 0]
    return next_token


def load_sampler(
    name: str,
    **kvargs: Dict[str, Any],
) -> Callable[[Tensor], Tensor]:
    """Loads the sampler used for generating next tokens.

    Args:
        name (str): Sampler name, can be `greedy` or `probability`.

    Returns:
        Callable[ [ Tensor, ], Tensor, ]: Function mapping logits to next
            token indices.
    """
    sample_fn = {
        "greedy": sample_greedy,
        "probability": sample_probability,
    }[name]
    if len(kvargs) > 0:
        return functools.partial(sample_fn, **kvargs)  # type: ignore
    else:
        return sample_fn  # type: ignore
