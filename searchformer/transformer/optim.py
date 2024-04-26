# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import functools
import math
from dataclasses import dataclass
from typing import Any, Dict, Tuple

from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR, LRScheduler


@dataclass
class OptimConfig:
    """Dataclass holding optimizer config."""

    lr: float
    lr_schedule: str
    train_steps: int
    warmup: int = 2000
    beta_0: float = 0.9
    beta_1: float = 0.99
    cycle_length: float = 1.0
    cosine_theta: float = 1.0
    lr_min_ratio: float = 0.1

    def __post_init__(self):
        assert self.lr_schedule in ["constant", "cosine"]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lr": self.lr,
            "lr_schedule": self.lr_schedule,
            "train_steps": self.train_steps,
            "warmup": self.warmup,
            "beta_0": self.beta_0,
            "beta_1": self.beta_1,
            "cycle_length": self.cycle_length,
            "cosine_theta": self.cosine_theta,
            "lr_min_ratio": self.lr_min_ratio,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "OptimConfig":
        return OptimConfig(**d)


def lr_constant(step: int, warmup: int) -> float:
    if step < warmup:
        return float(step) / float(warmup)
    else:
        return 1.0


def lr_cosine(
    step: int,
    warmup: int,
    n_steps: int,
    cycle_length: float,
    theta: float,
    min_ratio: float,
) -> float:
    if step <= warmup:
        lr = float(step) / warmup
    elif step <= n_steps:
        s = float(step - warmup) / (n_steps - warmup)
        lr = min_ratio + 0.5 * (1 - min_ratio) * (
            math.cos(math.pi * s**theta / cycle_length) + 1
        )
    else:
        lr = min_ratio
    return lr


def build_optimizer(
    model: nn.Module,
    config: OptimConfig,
) -> Tuple[AdamW, LRScheduler]:
    """Builds optimizer for provided nextwork based on optimizer config.

    Args:
        model (nn.Module): Model that is optimized.
        config (OptimConfig): Instance of optimizer config.

    Raises:
        ValueError: Raised if `config.lr_schedule` is neither set to
            `constant` or `cosine`.

    Returns:
        Tuple[AdamW, LRScheduler]: Instance of AdamW optimizer and instance of
            LRScheduler used for scheduling the learning rate.
    """
    optimizer = AdamW(
        params=model.parameters(),
        lr=config.lr,
        betas=(config.beta_0, config.beta_1),
    )
    if config.lr_schedule == "constant":
        schedule_fn = functools.partial(
            lr_constant,
            warmup=config.warmup,
        )
    elif config.lr_schedule == "cosine":
        schedule_fn = functools.partial(
            lr_cosine,
            warmup=config.warmup,
            n_steps=config.train_steps,
            cycle_length=config.cycle_length,
            theta=config.cosine_theta,
            min_ratio=config.lr_min_ratio,
        )
    else:
        raise ValueError(
            f"Invalid learning rate schedule {config.lr_schedule}.",
        )
    scheduler = LambdaLR(optimizer, schedule_fn)
    return optimizer, scheduler
