# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import datetime
import functools
import io
import json
import logging
import math
import os
import random
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import click
import gridfs
import pandas as pd
import torch
import torch.nn as nn
from pymongo.collection import Collection
from torch import Tensor
from torch.distributed import (
    barrier,
    destroy_process_group,
    get_rank,
    get_world_size,
    init_process_group,
)
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from .trace import AStarTrace, DictTokenizer, TokenizedDataset
from .transformer import (
    EncoderDecoder,
    EncoderDecoderConfig,
    OptimConfig,
    build_optimizer,
)
from .utils import mongodb_client, repeat_iterator, setup_logging_ddp


class AStarTraceIterableDataset(IterableDataset):
    def __init__(
        self,
        name: str,
        num_sequences: Optional[int] = None,
        reasoning_range: Optional[Tuple[int, int]] = None,
        shuffle: bool = False,
        use_test: bool = False,
        load_batch_size: int = 10000,
        plan_only: bool = False,
    ):
        """Constructs an object to load and iterate over training sequences
        stored in MongoDB.

        Args:
            name (str): Dataset name.
            num_sequences (Optional[int], optional): Number of sequences to
                iterate over. If None, iterate over all sequences stored in
                MongoDB. Defaults to None.
            reasoning_range (Optional[Tuple[int, int]], optional): Only
                generate sequences whose `reasoning` sequnece falls into the
                specified range. If None, do not filter sequences. Defaults
                to None.
            shuffle (bool, optional): Randomize iteration order. Defaults
                to False.
            use_test (bool, optional): If True, load from test set, otherwise
                load from training set. Defaults to False.
            load_batch_size (int, optional): Batch size used for bulk-loading
                over network from MongoDB. Defaults to 10000.
            plan_only (bool, optional): If True, only include solution plan
                and exclude execution trace. If False, include execution trace
                and plan. Defaults to False.
        """
        self.shuffle = shuffle
        self.use_test = use_test
        self.load_batch_size = load_batch_size
        self.dataset = TokenizedDataset(name)
        self.tokenizer = DictTokenizer(self.dataset.vocabulary)
        self.plan_only = plan_only

        if not use_test and reasoning_range is None:
            self.ids = self.dataset.train_ids
            logging.debug(f"Found {len(self.ids)} train sequence ids.")
        elif not use_test and reasoning_range is not None:
            self.ids = self.dataset.train_ids_within_range(*reasoning_range)
            logging.debug(
                f"Found {len(self.ids)} train sequence ids "
                + f"in range {reasoning_range}."
            )
        elif use_test and reasoning_range is None:
            self.ids = self.dataset.test_ids
            logging.debug(f"Found {len(self.ids)} test sequence ids.")
        elif use_test and reasoning_range is not None:
            self.ids = self.dataset.test_ids_within_range(*reasoning_range)
            logging.debug(
                f"Found {len(self.ids)} test sequence ids "
                + f"in range {reasoning_range}."
            )

        self.ids.sort()
        if num_sequences is not None:
            self.ids = self.ids[:num_sequences]
        logging.debug(f"Using {len(self.ids)} seq ids in worker")
        logging.debug(f"use_test={use_test}")

        rank = get_rank()
        world_size = get_world_size()
        slice_size = math.ceil(len(self.ids) / world_size)

        rank_world_str = f"rank={rank}, world_size={world_size}"
        logging.debug(f"AStarTraceIterableDataset: {rank_world_str}")
        self.ids = self.ids[rank * slice_size : (rank + 1) * slice_size]

    def __iter__(self) -> Iterator[AStarTrace]:
        worker_info = get_worker_info()
        ids_wk = self.ids
        if worker_info is not None:
            per_worker = math.ceil(len(ids_wk) / worker_info.num_workers)
            worker_id = worker_info.id
            it_start = worker_id * per_worker
            ids_wk = ids_wk[it_start : it_start + per_worker]

        if not self.use_test:
            batch_loader = self.dataset.train_it(ids_wk, self.load_batch_size)
        else:
            batch_loader = self.dataset.test_it(ids_wk, self.load_batch_size)
        for batch in batch_loader:
            tensor_list = self.tokenizer.tokenize_batch(batch, self.plan_only)
            if self.shuffle:
                random.shuffle(tensor_list)
            for trace in tensor_list:
                yield trace


def pad_and_mask_sequences(
    batch: Sequence[Tensor],
    max_seq_len: Optional[int] = None,
) -> Tuple[Tensor, Tensor]:
    """Concatenates a sequence of different length tensors to a padded tensor
    and length mask.

    Args:
        batch (Sequence[Tensor]): List of 1D tensors.
        max_seq_len (Optional[int], optional): Length of padded tensor.
            Defaults to None.

    Returns:
        Tuple[Tensor, Tensor]:
            - Padded tensor of shape `[n, l]`.
            - Length mask of 0 and 1 values.
    """
    if max_seq_len is None:
        seq_len = map(lambda seq: seq.shape[0], batch)
        max_seq_len = functools.reduce(max, seq_len)
    tokens = torch.zeros((len(batch), max_seq_len)).long()
    mask = torch.zeros_like(tokens)
    for i, seq in enumerate(batch):
        tokens[i, : len(seq)] = seq
        mask[i, : len(seq)] = 1
    return tokens, mask


@dataclass
class BatchedAStarTrace:
    """Batched A* trace tensors used for training."""

    prompt: Tensor
    prompt_mask: Tensor
    trace_plan: Tensor
    trace_mask: Tensor
    plan_mask: Tensor

    def pin_memory(self) -> "BatchedAStarTrace":
        self.prompt = self.prompt.pin_memory()
        self.prompt_mask = self.prompt_mask.pin_memory()
        self.trace_plan = self.trace_plan.pin_memory()
        self.trace_mask = self.trace_mask.pin_memory()
        self.plan_mask = self.plan_mask.pin_memory()
        return self

    def to(self, rank: int) -> "BatchedAStarTrace":
        self.prompt = self.prompt.to(f"cuda:{rank}")
        self.prompt_mask = self.prompt_mask.to(f"cuda:{rank}")
        self.trace_plan = self.trace_plan.to(f"cuda:{rank}")
        self.trace_mask = self.trace_mask.to(f"cuda:{rank}")
        self.plan_mask = self.plan_mask.to(f"cuda:{rank}")
        return self

    def cuda(self) -> "BatchedAStarTrace":
        self.prompt = self.prompt.cuda()
        self.prompt_mask = self.prompt_mask.cuda()
        self.trace_plan = self.trace_plan.cuda()
        self.trace_mask = self.trace_mask.cuda()
        self.plan_mask = self.plan_mask.cuda()
        return self

    def cpu(self) -> "BatchedAStarTrace":
        self.prompt = self.prompt.cpu()
        self.prompt_mask = self.prompt_mask.cpu()
        self.trace_plan = self.trace_plan.cpu()
        self.trace_mask = self.trace_mask.cpu()
        self.plan_mask = self.plan_mask.cpu()
        return self

    def __len__(self) -> int:
        return self.prompt.shape[0]

    @staticmethod
    def from_sequence(batch: Sequence[AStarTrace]) -> "BatchedAStarTrace":
        """Constructs a batched tensor from a sequence of `AStarTrace`
        data classes.

        Args:
            batch (Sequence[AStarTrace]): List of A* trace dataclasses.

        Returns:
            BatchedAStarTrace: Batched A* trace dataclass.
        """
        prompt_seq = [b.prompt for b in batch]
        trace_plan_seq = [b.trace_plan for b in batch]
        prompt, prompt_mask = pad_and_mask_sequences(prompt_seq)
        trace_plan, plan_mask = pad_and_mask_sequences(trace_plan_seq)
        trace_mask = torch.zeros_like(plan_mask)
        for i, l in enumerate(map(lambda b: b.plan_start, batch)):
            trace_mask[i, :l] = 1
            plan_mask[i, :l] = 0
        return BatchedAStarTrace(
            prompt=prompt,
            prompt_mask=prompt_mask,
            trace_plan=trace_plan,
            trace_mask=trace_mask,
            plan_mask=plan_mask,
        )


@dataclass
class DataConfig:
    """Dataclass holding data loader configuration parameters."""

    train_name: str
    test_name: str
    batch_size: int
    plan_only: bool = False
    num_train_sequences: Optional[int] = None
    num_test_sequences: Optional[int] = 100000
    load_batch_size: int = 10000
    num_workers: int = 2
    min_reasoning_len: Optional[int] = None
    max_reasoning_len: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        config_as_dict = {
            "train_name": self.train_name,
            "test_name": self.test_name,
            "batch_size": self.batch_size,
            "plan_only": self.plan_only,
            "num_train_sequences": self.num_train_sequences,
            "num_test_sequences": self.num_test_sequences,
            "load_batch_size": self.load_batch_size,
            "num_workers": self.num_workers,
        }
        if self.min_reasoning_len is not None:
            config_as_dict["min_reasoning_len"] = self.min_reasoning_len
        if self.max_reasoning_len is not None:
            config_as_dict["max_reasoning_len"] = self.max_reasoning_len
        return config_as_dict

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "DataConfig":
        return DataConfig(**d)

    @property
    def reasoning_range(self) -> Optional[Tuple[int, int]]:
        if self.min_reasoning_len is not None:
            if self.max_reasoning_len is not None:
                return (self.min_reasoning_len, self.max_reasoning_len)
        return None


class NextTokenPredictionLoss(nn.Module):
    """Next token prediction loss module. This loss module is used for training
    the encoder-decoder architecture.
    """

    def __init__(self, model: EncoderDecoder):
        """Constructs loss module.

        Args:
            model (EncoderDecoder): Encoder-decoder network module that is
                optimized.
        """
        super().__init__()
        self.model = model
        self.loss_obj = nn.CrossEntropyLoss(reduction="none")

    def forward(
        self,
        batch: BatchedAStarTrace,
    ) -> Tuple[Tensor, Dict[str, float]]:
        logits = self.model(
            prompt=batch.prompt,
            prompt_mask=batch.prompt_mask,
            trace=batch.trace_plan[:, :-1],
        )
        logits_1 = logits.reshape(-1, logits.shape[-1])

        loss_1 = self.loss_obj(logits_1, batch.trace_plan[:, 1:].reshape(-1))

        loss_mat = loss_1.reshape(*batch.trace_plan[:, 1:].shape)
        tok_eq = (logits.argmax(-1) == batch.trace_plan[:, 1:]).float()

        loss_plan = (loss_mat * batch.plan_mask[:, 1:]).sum(-1)
        loss_plan /= batch.plan_mask[:, 1:].sum(-1)
        tok_eq_plan = (tok_eq * batch.plan_mask[:, 1:]).sum(-1)
        acc_plan = (tok_eq_plan == batch.plan_mask[:, 1:].sum(-1)).float()

        trace_seq_len = batch.trace_mask[:, 1:].sum(-1)
        if torch.any(trace_seq_len > 0):
            loss_trace = (loss_mat * batch.trace_mask[:, 1:]).sum(-1)
            loss_trace /= trace_seq_len
            tok_eq_trace = (tok_eq * batch.trace_mask[:, 1:]).sum(-1)
            acc_trace = tok_eq_trace == batch.trace_mask[:, 1:].sum(-1)
            acc_trace = acc_trace.float()
        else:
            loss_trace = torch.zeros_like(loss_plan)
            acc_trace = torch.zeros_like(acc_plan)

        mask = batch.trace_mask[:, 1:] + batch.plan_mask[:, 1:]
        assert mask.max() == 1.0
        loss = (loss_mat * mask).sum(-1)
        loss /= mask.sum(-1)
        acc_objective = ((tok_eq * mask).sum(-1) == mask.sum(-1)).float()

        loss_log = {
            "loss.objective": loss.mean().item(),
            "loss.trace": loss_trace.mean().item(),
            "loss.plan": loss_plan.mean().item(),
            "accuracy.trace": acc_trace.mean().item(),
            "accuracy.plan": acc_plan.mean().item(),
            "accuracy.objective": acc_objective.mean().item(),
        }
        return loss.mean(), loss_log


@dataclass
class TrainConfig:
    """Training hyper-parameter configuration."""

    run_id: str
    data: DataConfig
    encoder: str
    decoder: str
    optimizer: OptimConfig
    log_interval: int
    eval_interval: int
    start_checkpoint: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        doc_dict = {
            "_id": self.run_id,
            "data": self.data.to_dict(),
            "encoder": self.encoder,
            "decoder": self.decoder,
            "optimizer": self.optimizer.to_dict(),
            "log_interval": self.log_interval,
            "eval_interval": self.eval_interval,
        }
        if self.start_checkpoint is not None:
            doc_dict["start_checkpoint"] = self.start_checkpoint
        return doc_dict

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "TrainConfig":
        return TrainConfig(
            run_id=d["_id"],
            data=DataConfig.from_dict(d["data"]),
            encoder=d["encoder"],
            decoder=d["decoder"],
            optimizer=OptimConfig.from_dict(d["optimizer"]),
            log_interval=d["log_interval"],
            eval_interval=d["eval_interval"],
            start_checkpoint=d.get("start_checkpoint", None),
        )

    @staticmethod
    def from_args(
        run_id: str,
        train_name: str,
        test_name: str,
        encoder: str,
        decoder: str,
        plan_only: bool = False,
        batch_size: int = 16,
        num_train_sequences: int = 1000,
        num_test_sequences: int = 100000,
        lr: float = 1e-4,
        lr_schedule: str = "constant",
        train_steps: int = 5000,
        log_interval: int = 100,
        eval_interval: int = 500,
        min_reasoning_len: Optional[int] = None,
        max_reasoning_len: Optional[int] = None,
        start_checkpoint: Optional[str] = None,
    ) -> "TrainConfig":
        return TrainConfig(
            run_id=run_id,
            data=DataConfig(
                train_name=train_name,
                test_name=test_name,
                batch_size=batch_size,
                num_train_sequences=num_train_sequences,
                num_test_sequences=num_test_sequences,
                plan_only=plan_only,
                min_reasoning_len=min_reasoning_len,
                max_reasoning_len=max_reasoning_len,
            ),
            encoder=encoder,
            decoder=decoder,
            optimizer=OptimConfig(
                lr=lr,
                lr_schedule=lr_schedule,
                train_steps=train_steps,
            ),
            log_interval=log_interval,
            eval_interval=eval_interval,
            start_checkpoint=start_checkpoint,
        )


def dataframe_from_log_collection(collection: Collection) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    for res in collection.find():
        res["timestamp"] = res["_id"].generation_time
        res.pop("_id")
        if "rank" in res.keys():
            res.pop("rank")
        if "num_sequences" in res.keys():
            res.pop("num_sequences")
        if "lr" in res.keys():
            # lr_list = res["lr"]
            # for i, lr in enumerate(lr_list):
            #     res[f"lr_{i}"] = lr
            res.pop("lr")
        records.append(res)
    df = pd.DataFrame(records)
    if len(df) > 0:
        df = df.groupby(["step"]).mean().reset_index()
    return df


class TrainLogger:
    """Logger class to log training statistics to MongoDB."""

    def __init__(self, rank: Optional[int] = None):
        """Constructs logger object for training logger of a specific rank.

        Args:
            rank (Optional[int], optional): Rank index raning from `0` to
                `n-1`, where `n` is the total number of workers.
                Defaults to None.
        """
        self._rank = rank
        self._values: Dict[str, float] = {}
        self._count = 0
        self._num_sequences = 0

    def log(self, d: Dict[str, float], num_sequences: int):
        if len(self._values) == 0:
            for k, v in d.items():
                assert type(v) is float
                self._values[k] = v
        else:
            k_exist = set(self._values.keys())
            k_new = set(d.keys())
            assert (
                len(k_exist.difference(k_new)) == 0
            ), "Cannot change keys during logging."

            for k, v in d.items():
                assert type(v) is float
                self._values[k] = self._values[k] + v

        self._count += 1
        self._num_sequences += num_sequences

    def get_log_dict_and_reset(self, step: int) -> Dict[str, Any]:
        values = {k: v / self._count for k, v in self._values.items()}
        metas = {"step": step, "num_sequences": self._num_sequences}
        if self._rank is not None:
            metas["rank"] = self._rank
        for k in self._values.keys():
            self._values[k] = 0.0
        self._count = 0
        self._num_sequences = 0
        return {"meta": metas, "value": values}


def _id_to_ts(d: Dict[str, Any]) -> Dict[str, Any]:
    d["_ts"] = d["_id"].generation_time
    d.pop("_id")
    return d


def filter_logs_most_recent(logs: pd.DataFrame) -> pd.DataFrame:
    most_recent = (
        logs[["meta.step", "meta.rank", "_ts"]]
        .groupby(["meta.step", "meta.rank"])
        .max()
        .reset_index()
    )
    logs = pd.merge(
        most_recent, logs, on=["_ts", "meta.step", "meta.rank"], how="left"
    )  # type: ignore
    return logs.copy()


def agg_logs_across_ranks(logs: pd.DataFrame) -> pd.DataFrame:
    logs.drop(columns=["_ts", "meta.rank"], inplace=True)  # type: ignore
    seq_per_step = (
        logs[["meta.step", "meta.num_sequences"]]
        .groupby("meta.step")
        .sum()
        .reset_index()
    )
    seq_per_step.rename(
        columns={"meta.num_sequences": "seq_total"},
        inplace=True,
    )

    logs = pd.merge(
        logs,
        seq_per_step,
        on="meta.step",
        how="left",
    )  # type: ignore
    meta_num_seq = logs["meta.num_sequences"]
    rank_weight = meta_num_seq / logs["seq_total"]  # type: ignore
    logs.drop(
        columns=["meta.num_sequences", "seq_total"],
        inplace=True,
    )  # type: ignore
    for col in filter(lambda c: c.startswith("value."), logs.columns):
        logs[col] = logs[col] * rank_weight
    logs = logs.groupby("meta.step").sum().reset_index()
    rename_dict = {"meta.step": "step"}
    for col in filter(lambda c: c.startswith("value."), logs.columns):
        rename_dict[col] = col.split("value.")[1]
    logs.rename(columns=rename_dict, inplace=True)  # type: ignore
    return logs


class TrainRunData:
    """Class used to access training data logs from MongoDB."""

    def __init__(self):
        self.client = mongodb_client()
        self.db = self.client["trainDB"]

    @functools.cached_property
    def config_collection(self) -> Collection:
        return self.db["config"]

    def drop_configs(self):
        self.db.drop_collection("config")

    @property
    def run_ids(self) -> List[str]:
        res_it = self.config_collection.find({}, {"_id": 1})
        return [d["_id"] for d in res_it]

    def run_exists(self, run_id: str) -> bool:
        return self.config_collection.find_one({"_id": run_id}) is not None

    def add_config(self, config: TrainConfig):
        self.config_collection.insert_one(config.to_dict())

    def load_configs_by_id(self, regex: str = "*") -> pd.DataFrame:
        config_res = self.config_collection.find({"_id": {"$regex": regex}})
        return pd.json_normalize(config_res)  # type: ignore

    def load_sweep_config(
        self,
        regex: str = "*",
        remove_common_configs: bool = True,
    ) -> pd.DataFrame:
        configs = self.load_configs_by_id(regex)
        if remove_common_configs:
            nuniq = configs.nunique()
            nuniq[nuniq == 1].index
            configs.drop(
                columns=nuniq[nuniq == 1].index,
                inplace=True,
            )  # type: ignore
        return configs

    def log_train_collection(self, run_id: str) -> Collection:
        return self.db[f"log.{run_id}.train"]

    def log_test_collection(self, run_id: str) -> Collection:
        return self.db[f"log.{run_id}.test"]

    def log_train(self, run_id: str, log_dict: Dict[str, Any]):
        self.log_train_collection(run_id).insert_one(log_dict)

    def log_test(self, run_id: str, log_dict: Dict[str, Any]):
        self.log_test_collection(run_id).insert_one(log_dict)

    def get_train_log(self, run_id: str) -> Optional[pd.DataFrame]:
        coll = self.log_train_collection(run_id)
        logs = pd.json_normalize(map(_id_to_ts, coll.find()))  # type: ignore
        if len(logs) == 0:
            return None
        logs = filter_logs_most_recent(logs)
        logs = agg_logs_across_ranks(logs)
        return logs

    def get_test_log(self, run_id: str) -> Optional[pd.DataFrame]:
        coll = self.log_test_collection(run_id)
        logs = pd.json_normalize(map(_id_to_ts, coll.find()))  # type: ignore
        if len(logs) == 0:
            return None
        logs = filter_logs_most_recent(logs)
        logs = agg_logs_across_ranks(logs)
        return logs

    def drop_all(self):
        self.client.drop_database("train-db")

    def drop(self, run_id: str):
        # self.db.drop_collection(self.checkpoint_collection(id))
        self.db.drop_collection(self.log_train_collection(run_id))
        self.db.drop_collection(self.log_test_collection(run_id))
        self.config_collection.delete_one({"_id": run_id})

    def load_train_logs_from_id(self, run_id: str) -> pd.DataFrame:
        return dataframe_from_log_collection(self.log_train_collection(run_id))

    def load_test_logs_from_id(self, run_id: str) -> pd.DataFrame:
        return dataframe_from_log_collection(self.log_test_collection(run_id))

    def bulk_load_train_logs(self, id_list: List[str]) -> pd.DataFrame:
        df_list: List[pd.DataFrame] = []
        for run_id in id_list:
            df = self.get_train_log(run_id)
            if df is None:
                logging.warning(f"Cannot load train logs for {run_id}")
                continue
            df["_id"] = run_id
            df_list.append(df)
        return pd.concat(df_list)

    def bulk_load_test_logs(self, id_list: List[str]) -> pd.DataFrame:
        df_list: List[pd.DataFrame] = []
        for run_id in id_list:
            df = self.get_test_log(run_id)
            if df is None:
                logging.warning(f"Cannot load test logs for {run_id}")
                continue
            df["_id"] = run_id
            df_list.append(df)
        return pd.concat(df_list)


def _checkpoint_id_valid(checkpoint_id: str) -> bool:
    p = re.compile("^[a-zA-Z0-9-_]*$")
    m = p.match(checkpoint_id)
    if m is None:
        return False
    else:
        start, end = m.span()
        if start == 0 and end == len(checkpoint_id):
            return True
        else:
            return False


@dataclass
class Checkpoint:
    """Dataclass used to hold a checkpoint.

    A checkpoint holds training run ID, a hyper-parameter config dictionary,
    the model parameters, and optimizer parameters.
    """

    checkpoint_id: str
    step: int
    config: Dict[str, Any]
    model: Dict[str, Any]
    optimizer: Dict[str, Any]

    def __post_init__(self):
        assert _checkpoint_id_valid(self.checkpoint_id)

    def to_dict(self) -> Dict[str, Any]:
        return dict(
            checkpoint_id=self.checkpoint_id,
            step=self.step,
            config=self.config,
            model=self.model,
            optimizer=self.optimizer,
        )

    @property
    def config_obj(self) -> TrainConfig:
        return TrainConfig.from_dict(self.config)

    @property
    def model_only_state_dict(self) -> Dict[str, Any]:
        model_state_dict: Dict[str, Any] = {}
        for k, v in self.model.items():
            if k.startswith("module.model"):
                k_model = k.replace("module.model.", "")
                model_state_dict[k_model] = v
        return model_state_dict

    def to_buffer(self) -> io.BytesIO:
        buffer = io.BytesIO()
        torch.save(self.to_dict(), buffer)
        buffer.seek(0)
        return buffer

    @staticmethod
    def from_buffer(buffer: io.BytesIO) -> "Checkpoint":
        state_dict = torch.load(buffer, map_location=torch.device("cpu"))
        return Checkpoint(**state_dict)

    def to_file(self, filename: str):
        torch.save(self.to_dict(), filename)

    @staticmethod
    def from_file(filename: str) -> "Checkpoint":
        state_dict = torch.load(filename, map_location=torch.device("cpu"))
        return Checkpoint(**state_dict)


CHECKPOINT_DB_NAME = "ckptDB"


class CheckpointDataset:
    """Class to write and load checkpoints from MonogDB."""

    def __init__(self):
        self.client = mongodb_client()
        self.db = self.client[CHECKPOINT_DB_NAME]

    def get_fs(self, checkpoint_id: str) -> gridfs.GridFS:
        assert _checkpoint_id_valid(checkpoint_id)
        return gridfs.GridFS(self.db, collection=f"fs.{checkpoint_id}")

    def list_checkpoint_id(self) -> List[str]:
        coll_names = self.db.list_collection_names()
        coll_names_files = filter(lambda s: s.endswith("files"), coll_names)
        ckpt_id_it = map(lambda s: s.split(".")[1], coll_names_files)
        return list(ckpt_id_it)

    def drop(self, checkpoint_id: str):
        assert _checkpoint_id_valid(checkpoint_id)
        self.db.drop_collection(f"fs.{checkpoint_id}.files")
        self.db.drop_collection(f"fs.{checkpoint_id}.chunks")

    def drop_all(self):
        self.client.drop_database(CHECKPOINT_DB_NAME)

    def add(self, ckpt: Checkpoint):
        buffer = ckpt.to_buffer()
        self.get_fs(ckpt.checkpoint_id).put(
            buffer.read(),
            _id=ckpt.checkpoint_id,
        )

    def remove(self, checkpoint_id: str):
        assert _checkpoint_id_valid(checkpoint_id)
        self.get_fs(checkpoint_id).delete(file_id=checkpoint_id)

    def has_checkpoint(self, checkpoint_id: str) -> bool:
        assert _checkpoint_id_valid(checkpoint_id)
        return self.get_fs(checkpoint_id).exists(document_or_id=checkpoint_id)

    def load(self, checkpoint_id: str) -> Checkpoint:
        assert self.has_checkpoint(
            checkpoint_id
        ), f"No checkpoint exists for id={checkpoint_id}"
        ckpt = self.get_fs(checkpoint_id).get(file_id=checkpoint_id).read()
        buffer = io.BytesIO(ckpt)
        return Checkpoint.from_buffer(buffer)


@click.group()
def main():
    pass


@main.command()
def list_all_checkpoints():
    """List all stored checkpoints."""
    ckpt_dataset = CheckpointDataset()
    ckpt_id_list = ckpt_dataset.list_checkpoint_id()
    ckpt_id_list.sort()
    print("\n".join(ckpt_id_list))


def _export_checkpoint(checkpoint_id: str, export_dir: Optional[str] = None):
    ckpt_dataset = CheckpointDataset()
    ckpt = ckpt_dataset.load(checkpoint_id)
    fn = f"{checkpoint_id}.ckpt"
    if export_dir is not None:
        fn = os.path.join(export_dir, fn)
    logging.info(f"Saving checkpoint to file {fn}")
    ckpt.to_file(fn)
    logging.info("Finished export.")


@main.command()
@click.option(
    "--checkpoint-id",
    type=str,
    help="Checkpoint id which is the same as run_id.",
)
@click.option(
    "--export-dir",
    type=None,
    help="Export directory",
)
def export_checkpoint(checkpoint_id: str, export_dir: Optional[str]):
    """Export a checkpoint stored in MongoDB to a file."""
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(levelname)s - %(asctime)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    _export_checkpoint(checkpoint_id, export_dir=export_dir)


@main.command()
@click.option(
    "--num-worker",
    type=int,
    default=None,
    help="Number of parallel export workers.",
)
@click.option(
    "--export-dir",
    type=None,
    help="Export directory",
)
def bulk_export_checkpoint(
    num_worker: Optional[int],
    export_dir: Optional[str],
):
    """Bulk export of all checkpoints into current working directory."""
    import multiprocessing as mp

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(levelname)s - %(asctime)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    ckpt_dataset = CheckpointDataset()
    ckpt_id_list = ckpt_dataset.list_checkpoint_id()
    export_fn = functools.partial(_export_checkpoint, export_dir=export_dir)
    with mp.Pool(num_worker) as p:
        p.map(export_fn, ckpt_id_list)

    logging.info("Finished exporting all checkpoints.")


def _import_checkpoint(filename: str):
    ckpt = Checkpoint.from_file(filename)
    logging.info(f"Importing checkpoint with id {ckpt.checkpoint_id}")
    ckpt_dataset = CheckpointDataset()
    ckpt_dataset.add(ckpt)
    logging.info("Finished import.")


@main.command()
@click.option("--filename", type=str, help="Checkpoint filename.")
def import_checkpoint(filename: str):
    """Import a checkpoint into MongoDB from file."""
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(levelname)s - %(asctime)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    _import_checkpoint(filename)


@main.command()
@click.option(
    "--num-worker",
    type=int,
    default=None,
    help="Number of parallel export workers.",
)
@click.option(
    "--import-dir",
    type=None,
    help="Import directory",
)
def bulk_import_checkpoint(num_worker: Optional[int], import_dir: str):
    """Bulk import checkpoints from directory."""
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(levelname)s - %(asctime)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    import multiprocessing as mp
    from glob import glob

    fn_list = glob(os.path.join(import_dir, "*.ckpt"))
    with mp.Pool(num_worker) as p:
        p.map(_import_checkpoint, fn_list)


class TrainRun:
    """Class implementing a training run of an encoder-decoder model."""

    def __init__(self, config: TrainConfig):
        self.config = config
        self.ckpt_data = CheckpointDataset()
        self.rank = get_rank()
        self.world_size = get_world_size()

        vocab_size = self.test_dataset.tokenizer.vocab_size
        model_config = EncoderDecoderConfig.from_name(
            enc_name=self.config.encoder,
            dec_name=self.config.decoder,
            vocab_size=vocab_size,
        )
        torch.cuda.set_device(self.rank % torch.cuda.device_count())
        self.model = model_config.construct_model().cuda()
        self.loss = DDP(NextTokenPredictionLoss(self.model))
        self.optimizer, self.schedule = build_optimizer(
            self.model, self.config.optimizer
        )
        self.step = 0

    @functools.cached_property
    def train_dataset(self) -> AStarTraceIterableDataset:
        return AStarTraceIterableDataset(
            name=self.config.data.train_name,
            num_sequences=self.config.data.num_train_sequences,
            shuffle=True,
            use_test=False,
            load_batch_size=self.config.data.load_batch_size,
            plan_only=self.config.data.plan_only,
            reasoning_range=self.config.data.reasoning_range,
        )

    @functools.cached_property
    def test_dataset(self) -> AStarTraceIterableDataset:
        return AStarTraceIterableDataset(
            name=self.config.data.test_name,
            num_sequences=self.config.data.num_test_sequences,
            shuffle=False,
            use_test=True,
            load_batch_size=self.config.data.load_batch_size,
            plan_only=self.config.data.plan_only,
            reasoning_range=self.config.data.reasoning_range,
        )

    @functools.cached_property
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.batch_size,
            pin_memory=True,
            collate_fn=BatchedAStarTrace.from_sequence,
            num_workers=self.config.data.num_workers,
            persistent_workers=self.config.data.num_workers > 0,
        )

    @functools.cached_property
    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.config.data.batch_size,
            pin_memory=True,
            collate_fn=BatchedAStarTrace.from_sequence,
            num_workers=self.config.data.num_workers,
            persistent_workers=self.config.data.num_workers > 0,
        )

    def train_batch_iterator(self) -> Iterator[Tuple[int, Any]]:
        return repeat_iterator(
            self.train_dataloader,
            self.config.optimizer.train_steps,
        )

    def train_step(self, batch: Any) -> Dict[str, Any]:
        self.optimizer.zero_grad(set_to_none=True)
        batch_cuda = batch.cuda()
        loss_obj, loss_dict = self.loss(batch_cuda)
        loss_obj.backward()
        self.optimizer.step()
        self.schedule.step()
        loss_obj = None
        self.optimizer.zero_grad(set_to_none=True)
        return loss_dict

    def evaluate(self, run_data: TrainRunData):
        barrier()
        self.model.eval()
        barrier()
        test_logger = TrainLogger(self.rank)
        logging.info("Starting evaluation ...")
        for batch in self.test_dataloader:
            batch_cuda = batch.cuda()
            loss_obj, log_dict = self.loss(batch_cuda)
            test_logger.log(log_dict, len(batch))
            loss_obj = None

        logging.info("Completed evaluation.")
        run_data.log_test(
            self.config.run_id, test_logger.get_log_dict_and_reset(self.step)
        )
        barrier()
        self.model.train()
        barrier()

    def checkpoint(self):
        barrier()
        if self.rank == 0:
            logging.info("Checkpointing model ...")
            if self.ckpt_data.has_checkpoint(self.config.run_id):
                self.ckpt_data.remove(self.config.run_id)
            self.ckpt_data.add(
                Checkpoint(
                    checkpoint_id=self.config.run_id,
                    step=self.step,
                    config=self.config.to_dict(),
                    model=self.loss.state_dict(),
                    optimizer=self.optimizer.state_dict(),
                )
            )
        barrier()
        logging.info("Continuing after checkpoint")

    @property
    def has_checkpoint(self) -> bool:
        return self.ckpt_data.has_checkpoint(self.config.run_id)

    def reconstruct_from_checkpoint(self, run_id: Optional[str] = None):
        if run_id is None:
            run_id = self.config.run_id

        barrier()
        logging.debug(f"Reconstructing checkpoint for run {run_id}")
        ckpt = self.ckpt_data.load(run_id)
        self.step = ckpt.step
        self.loss.load_state_dict(ckpt.model)
        self.optimizer.load_state_dict(ckpt.optimizer)
        barrier()

    def train(self, run_data: TrainRunData):
        steps_to_go = self.config.optimizer.train_steps - self.step
        if steps_to_go == 0:
            logging.info("Run already complete. No further steps to train.")
            return

        logging.info("Starting training ...")
        train_logger = TrainLogger(self.rank)
        for batch in repeat_iterator(self.train_dataloader, steps_to_go):
            step_result = self.train_step(batch)
            train_logger.log(step_result, len(batch))
            self.step += 1

            if self.step % self.config.log_interval == 0:
                log_dict = train_logger.get_log_dict_and_reset(self.step)
                lr_list = self.schedule.get_last_lr()
                lr_dict = {str(i): lr for i, lr in enumerate(lr_list)}
                log_dict["value"]["lr"] = lr_dict
                run_data.log_train(self.config.run_id, log_dict)
                logging.info(
                    f"Completed {self.step} steps, "
                    + f"lr={self.schedule.get_last_lr()}"
                )

            if self.step % self.config.eval_interval == 0:
                self.evaluate(run_data)
                self.checkpoint()

        if self.step % self.config.eval_interval > 0:
            self.evaluate(run_data)
            self.checkpoint()


@main.command()
def list_all():
    """List all training runs."""
    run_ids = [str(i) for i in TrainRunData().run_ids]
    run_ids.sort()
    print("\n".join(run_ids))


def _drop_run_by_id(run_id: str):
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(levelname)s - %(asctime)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.info(f"Attempting to delete {run_id} ...")
    ckpt_data = CheckpointDataset()
    if ckpt_data.has_checkpoint(run_id):
        ckpt_data.remove(run_id)
        assert not ckpt_data.has_checkpoint(
            run_id,
        ), "Checkpoint delete did not work."
    else:
        logging.warning(f"No checkpoint found for {run_id}")

    TrainRunData().drop(run_id)
    logging.info(f"Deleted {run_id}")


@main.command()
@click.option("--run-id", type=str, help="Training run id.")
def drop_run(run_id: str):
    """Drop individual training run."""
    _drop_run_by_id(run_id)


@main.command()
@click.option(
    "--run-id",
    type=str,
    multiple=True,
    help="Training run id list.",
)
@click.option("--jobs", type=int, default=None, help="Number of workers.")
def bulk_drop_run(run_id: List[str], jobs: int):
    """Drop multiple training runs."""
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(levelname)s - %(asctime)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    run_id_str = " ".join(run_id)
    logging.info(f"Bulk dropping of runs with ids: {run_id_str}")

    import multiprocessing as mp

    with mp.Pool(processes=jobs) as p:
        p.map(_drop_run_by_id, run_id)


def _train(run_id: str, **args):
    """Executes DDP training worker."""
    init_process_group(backend="nccl", timeout=datetime.timedelta(hours=4))
    setup_logging_ddp()
    run_data = TrainRunData()
    run = TrainRun(TrainConfig.from_args(run_id=run_id, **args))
    logging.info(f"Run id: {run.config.run_id}")
    if run.rank == 0:
        logging.info(f"Args: {args}")

    # To determine if this is the first time the run is launched, check in
    # every worker if the run already exists. If not, then determine if a
    # different checkpoint should be used to reconstruct from.
    run_exists = run_data.run_exists(run.config.run_id)
    if not run_exists and "start_checkpoint" in args.keys():
        run.reconstruct_from_checkpoint(args["start_checkpoint"])
        # Set step to 0 to re-initialize training.
        run.step = 0

    # After every worker has checked if a different checkpoint should be used
    # for initialization, add the run into MongoDB if it is new.
    barrier()
    if not run_data.run_exists(run.config.run_id) and run.rank == 0:
        run_data.add_config(run.config)

    barrier()
    if run.has_checkpoint:
        run.reconstruct_from_checkpoint()
    elif run.step == 0:
        # Only run evaluation and checkpointing at the beginning of training.
        run.evaluate(run_data)
        run.checkpoint()
    logging.info(f"Starting at step {run.step}")
    run.train(run_data)

    destroy_process_group()
    logging.info("Done.")


@main.command()
@click.option("--run-id", type=str, help="Run id.")
@click.option(
    "--train-name",
    type=str,
    default="maze.7-by-7-deterministic.simple",
    help="Training token dataset name.",
)
@click.option(
    "--test-name",
    type=str,
    default="maze.7-by-7-deterministic.simple",
    help="Test token dataset name.",
)
@click.option(
    "--encoder",
    type=str,
    default="enc-s",
    help="Encoder architecture name.",
)
@click.option(
    "--decoder",
    type=str,
    default="dec-s",
    help="Decoder architecture name.",
)
@click.option(
    "--plan-only",
    is_flag=True,
    help="""Train solution-only model. If not used a search-augmented model is 
trained.""",
)
@click.option(
    "--batch-size",
    type=int,
    default=16,
    help="""Batch size to be used by DDP worker. For example, if `--batch-size=16` 
is used together with 2 DDP workers (training on 2 GPUs), then the model is 
trained with a batch-size of 2*16=32.
""",
)
@click.option(
    "--num-train-sequences",
    type=int,
    default=125000,
    help="Number of training sequences.",
)
@click.option(
    "--num-test-sequences",
    type=int,
    default=100000,
    help="Number of test sequences (only used for test logs).",
)
@click.option("--lr", type=float, default=1e-5, help="Learning rate.")
@click.option(
    "--lr-schedule",
    type=str,
    default="constant",
    help="Learning rate schedule. Can be set to `constant` or `cosine`.",
)
@click.option(
    "--train-steps",
    type=int,
    default=2000,
    help="Total number of training steps.",
)
@click.option("--log-interval", type=int, default=100, help="Log interval.")
@click.option(
    "--eval-interval",
    type=int,
    default=1000,
    help="""Evaluation and checkpoint interval. Evaluation evaluates the loss 
statistic on the test set.""",
)
@click.option(
    "--min-reasoning-len",
    type=int,
    default=None,
    help="Minimum reasoning token trace length.",
)
@click.option(
    "--max-reasoning-len",
    type=int,
    default=None,
    help="Mazimum reasoning token trace length.",
)
@click.option(
    "--start-checkpoint",
    type=str,
    default=None,
    help="Id of start checkpoint.",
)
def single(run_id: str, **args):
    """Start single DDP training run."""
    _train(run_id, **args)


@main.command()
@click.option("--run-id", type=str, help="Run id.")
@click.option("--index", type=int, default=0, help="Config index.")
@click.option(
    "--sweep",
    type=str,
    default="config/sweep/maze_size_comparison.json",
    help="Sweep config file.",
)
def sweep(run_id: str, index: int, sweep: str):
    """Start single DDP training run with provided sweep config file.

    This launches the run using the hyper-parameter config at the
    provided index file.
    """
    with open(sweep, "r") as f:
        sweep_config_list = json.load(f)
    _train(run_id, **sweep_config_list[index])


if __name__ == "__main__":
    # logging.basicConfig(
    #     level=logging.DEBUG,
    #     format="%(levelname)s - %(asctime)s - %(name)s - %(message)s",
    #     datefmt="%Y-%m-%d %H:%M:%S",
    # )
    main()
