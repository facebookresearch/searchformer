# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import functools
import logging
import math
import random
from dataclasses import dataclass
from itertools import chain
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import click
import pandas as pd
import pymongo
from bson.objectid import ObjectId
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
from torch import Tensor

from .trace import DictTokenizer, TokenizedDataset, TokenizedTrace
from .train import Checkpoint, CheckpointDataset
from .transformer.model import EncoderDecoderConfig
from .transformer.utils import load_sampler
from .utils import mongodb_client

ROLLOUT_DB_NAME = "rolloutDB"


class RolloutException(Exception):
    pass


@dataclass
class Rollout:
    id: int
    prompt: List[str]
    reasoning: List[str]
    plan: List[str]
    rollouts: List[List[str]]

    def to_doc(self) -> Dict[str, Any]:
        return {
            "_id": self.id,
            "prompt": self.prompt,
            "reasoning": self.reasoning,
            "plan": self.plan,
            "rollouts": self.rollouts,
        }

    @staticmethod
    def from_doc(d: Dict[str, Any]) -> "Rollout":
        return Rollout(
            id=d["_id"],
            prompt=d["prompt"],
            reasoning=d["reasoning"],
            plan=d["plan"],
            rollouts=d["rollouts"],
        )


@dataclass
class RolloutMetaData:
    id: int
    correct_sequence: Dict[str, Any]
    rollouts: List[Dict[str, Any]]

    def to_doc(self) -> Dict[str, Any]:
        return {
            "_id": self.id,
            "correct_sequence": self.correct_sequence,
            "rollouts": self.rollouts,
        }

    @staticmethod
    def from_doc(d: Dict[str, Any]) -> "RolloutMetaData":
        return RolloutMetaData(
            id=d["_id"],
            correct_sequence=d["correct_sequence"],
            rollouts=d["rollouts"],
        )


def _slice_id_from_collection(
    collection: Collection,
    rank: int = 0,
    world_size: int = 1,
) -> List[Any]:
    ids = [res["_id"] for res in collection.find({}, {"_id": 1})]
    ids.sort()
    slice_size = math.ceil(len(ids) / world_size)
    ids_slice = ids[rank * slice_size : (rank + 1) * slice_size]
    logging.debug(f"Loading {len(ids_slice)} ids.")
    return ids_slice


def _rollout_iterator(
    collection: Collection,
    ids: List[int],
    load_batch_size: int = 1,
) -> Iterator[Rollout]:
    for i in range(0, len(ids), load_batch_size):
        result = collection.find(
            {"_id": {"$in": ids[i : i + load_batch_size]}},
        )
        rollout_it = map(Rollout.from_doc, result)
        for rollout in rollout_it:
            yield rollout


@dataclass
class RolloutParameter:
    checkpoint_id: str
    dataset_name: str
    sampler_name: str
    sampler_kvargs: Dict[str, Any]
    rollout_len: int
    rollout_repeats: int
    prefix_len: int = 0
    # Stores -1 to indicate that the dataset is not truncated by
    # reasoning length trace.
    min_reasoning_len: int = -1
    max_reasoning_len: int = -1

    def to_doc(self) -> Dict[str, Any]:
        return dict(
            checkpoint_id=self.checkpoint_id,
            dataset_name=self.dataset_name,
            sampler_name=self.sampler_name,
            sampler_kvargs=self.sampler_kvargs,
            rollout_len=self.rollout_len,
            rollout_repeats=self.rollout_repeats,
            prefix_len=self.prefix_len,
            min_reasoning_len=self.min_reasoning_len,
            max_reasoning_len=self.max_reasoning_len,
        )

    @functools.cached_property
    def sample_fn(
        self,
    ) -> Callable[[Tensor], Tensor]:
        return load_sampler(self.sampler_name, **self.sampler_kvargs)

    @staticmethod
    def unique_keys() -> List[str]:
        return [
            "checkpoint_id",
            "dataset_name",
            "sampler_name",
            "sampler_kvargs",
            "rollout_len",
            "rollout_repeats",
            "prefix_len",
            "min_reasoning_len",
            "max_reasoning_len",
        ]

    @property
    def reasoning_range(self) -> Optional[Tuple[int, int]]:
        if self.min_reasoning_len >= 0 and self.max_reasoning_len >= 0:
            return (self.min_reasoning_len, self.max_reasoning_len)
        else:
            return None


@dataclass
class RolloutDataset:
    id: str
    params: RolloutParameter

    @functools.cached_property
    def client(self) -> MongoClient:
        return mongodb_client()

    @functools.cached_property
    def db(self) -> Database:
        return self.client[ROLLOUT_DB_NAME]

    @functools.cached_property
    def checkpoint(self) -> Checkpoint:
        return CheckpointDataset().load(self.params.checkpoint_id)

    @functools.cached_property
    def dataset(self) -> TokenizedDataset:
        return TokenizedDataset(self.params.dataset_name)

    @functools.cached_property
    def rollout_train_collection(self) -> Collection:
        return self.db[f"rollout.{self.id}.train"]

    @functools.cached_property
    def rollout_test_collection(self) -> Collection:
        return self.db[f"rollout.{self.id}.test"]

    def has_train_sequence(self, id: int) -> bool:
        return self.rollout_train_collection.find_one({"_id": id}) is not None

    def has_test_sequence(self, id: int) -> bool:
        return self.rollout_test_collection.find_one({"_id": id}) is not None

    def add_rollout_train(self, rollout: Rollout):
        logging.debug(f"Adding train rollout with id {rollout.id}")
        self.rollout_train_collection.insert_one(rollout.to_doc())

    def add_rollout_test(self, rollout: Rollout):
        logging.debug(f"Adding test rollout with id {rollout.id}")
        self.rollout_test_collection.insert_one(rollout.to_doc())

    def rollout_train_it(
        self,
        rank: int = 0,
        world_size: int = 1,
        load_batch_size: int = 1000,
        shuffle: bool = False,
    ) -> Iterator[Rollout]:
        ids = _slice_id_from_collection(
            self.rollout_train_collection, rank=rank, world_size=world_size
        )
        if shuffle:
            random.shuffle(ids)
        return _rollout_iterator(
            self.rollout_train_collection,
            ids,
            load_batch_size,
        )

    def rollout_test_it(
        self,
        rank: int = 0,
        world_size: int = 1,
        load_batch_size: int = 1000,
        shuffle: bool = False,
    ) -> Iterator[Rollout]:
        ids = _slice_id_from_collection(
            self.rollout_test_collection, rank=rank, world_size=world_size
        )
        if shuffle:
            random.shuffle(ids)
        return _rollout_iterator(
            self.rollout_test_collection,
            ids,
            load_batch_size,
        )


@dataclass
class RolloutEvaluation:
    has_eos: bool
    plan_syntax_correct: bool
    plan_correct_start: bool
    plan_correct_goal: bool
    plan_correct: bool
    plan_length: int
    trace_tokens: int

    def to_dict(self) -> Dict[str, Any]:
        return dict(
            has_eos=self.has_eos,
            plan_syntax_correct=self.plan_syntax_correct,
            plan_correct_start=self.plan_correct_start,
            plan_correct_goal=self.plan_correct_goal,
            plan_correct=self.plan_correct,
            plan_length=self.plan_length,
            trace_tokens=self.trace_tokens,
        )


@dataclass
class RolloutEvaluationBatch:
    id: int
    optimal_plan_length: int
    reasoning_length: int
    rollout: List[RolloutEvaluation]

    def to_doc(self) -> Dict[str, Any]:
        return {
            "_id": self.id,
            "rollout": [r.to_dict() for r in self.rollout],
            "optimal_plan_length": self.optimal_plan_length,
            "reasoning_length": self.reasoning_length,
        }

    @staticmethod
    def from_doc(d: Dict[str, Any]) -> "RolloutEvaluationBatch":
        return RolloutEvaluationBatch(
            id=d["_id"],
            optimal_plan_length=d["optimal_plan_length"],
            reasoning_length=d["reasoning_length"],
            rollout=[RolloutEvaluation(**e) for e in d["rollout"]],
        )

    def to_dataframe(self) -> pd.DataFrame:
        df = pd.json_normalize(  # type: ignore
            [{"index": i, **e.to_dict()} for i, e in enumerate(self.rollout)]
        )
        df["_id"] = self.id
        df["optimal_plan_length"] = self.optimal_plan_length
        df["reasoning_length"] = self.reasoning_length
        return df


class RolloutDataStore:
    @functools.cached_property
    def client(self) -> MongoClient:
        return mongodb_client()

    @functools.cached_property
    def db(self) -> Database:
        return self.client[ROLLOUT_DB_NAME]

    @functools.cached_property
    def dataset_collection(self) -> Collection:
        data_coll = self.db["dataset"]
        uniq_keys = RolloutParameter.unique_keys()
        data_coll.create_index({k: 1 for k in uniq_keys}, unique=True)
        return data_coll

    def drop_all(self):
        self.client.drop_database(ROLLOUT_DB_NAME)

    def drop_dataset(self, dataset: RolloutDataset):
        self.db.drop_collection(dataset.rollout_train_collection)
        self.db.drop_collection(dataset.rollout_test_collection)
        self.dataset_collection.delete_one({"_id": ObjectId(dataset.id)})

    def load_by_id(self, id: str) -> RolloutDataset:
        res = self.dataset_collection.find_one({"_id": ObjectId(id)})
        assert type(res) is dict
        args = {k: v for k, v in res.items() if k != "_id"}
        return RolloutDataset(
            id=str(res["_id"]),
            params=RolloutParameter(**args),
        )

    def create_or_load(self, params: RolloutParameter) -> RolloutDataset:
        """Creates a new rollout datasets if one for the provided parameters
        does not exist. If it exists, then returns the created rollout dataset.

        Args:
            rollout_params (RolloutParameter): Parameters for rollout dataset.

        Returns:
            RolloutDataset: Dataset object
        """
        try:
            res_ins = self.dataset_collection.insert_one(params.to_doc())
            return RolloutDataset(id=str(res_ins.inserted_id), params=params)
        except pymongo.errors.DuplicateKeyError:  # type: ignore
            res_find = self.dataset_collection.find_one(filter=params.to_doc())
            assert res_find is not None and type(res_find) is dict
            return RolloutDataset(id=str(res_find["_id"]), params=params)

    def list_all(self) -> pd.DataFrame:
        return pd.json_normalize(self.dataset_collection.find())  # type: ignore


class RolloutWorker:
    def __init__(
        self,
        params: RolloutParameter,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.rank = rank
        self.world_size = world_size
        self.params = params

        self.rollouts = RolloutDataStore().create_or_load(params)
        self.tokenizer = DictTokenizer(self.rollouts.dataset.vocabulary)
        model_config = EncoderDecoderConfig.from_name(
            enc_name=self.rollouts.checkpoint.config_obj.encoder,
            dec_name=self.rollouts.checkpoint.config_obj.decoder,
            vocab_size=self.tokenizer.vocab_size,
        )
        model = model_config.construct_model()
        model.load_state_dict(self.rollouts.checkpoint.model_only_state_dict)
        self.model = model.cuda()
        self.model.eval()

    def rollout_from_prompt_tensor(
        self,
        prompt: Tensor,
        prefix_seq: Optional[List[str]] = None,
        num_samples: int = 1,
    ) -> List[List[str]]:
        if prefix_seq is None:
            prefix_seq = []
        prefix_seq_idx = self.tokenizer.encode(prefix_seq)

        trace_batch = self.model.rollout(
            prompt=prompt,
            bos_idx=self.tokenizer.bos,
            eos_idx=self.tokenizer.eos,
            max_rollout_len=self.rollouts.params.rollout_len,
            sample_fn=self.rollouts.params.sample_fn,
            num_samples=num_samples,
            prefix_seq=prefix_seq_idx,
        )
        trace_tok_list: List[List[str]] = []
        for trace in trace_batch:
            trace_tok = self.tokenizer.decode(trace.tolist())
            if "eos" in trace_tok:
                trace_tok = trace_tok[: trace_tok.index("eos") + 1]
            trace_tok_list.append(trace_tok)
        return trace_tok_list

    def rollout_trace(
        self,
        trace: TokenizedTrace,
        batch_size: int = 1,
    ) -> Rollout:
        prompt = self.tokenizer.tokenize_prompt(trace).cuda()
        rollout_list: List[List[str]] = []

        corr_rollout = trace.reasoning + trace.plan
        num_batch = math.ceil(
            self.rollouts.params.rollout_repeats / batch_size,
        )
        for _ in range(num_batch):
            rollout_list += self.rollout_from_prompt_tensor(
                prompt=prompt,
                prefix_seq=corr_rollout[: self.params.prefix_len],
                num_samples=batch_size,
            )
        return Rollout(
            id=trace.id,
            prompt=trace.prompt,
            reasoning=trace.reasoning,
            plan=trace.plan,
            rollouts=rollout_list,
        )

    def __slice_of_ids(self, ids: List[Any], num_sequences: int) -> List[Any]:
        ids.sort()
        ids = ids[:num_sequences]
        slice_size = math.ceil(len(ids) / self.world_size)
        ids_slice = ids[self.rank * slice_size : (self.rank + 1) * slice_size]
        logging.debug(f"Slice ids: {ids_slice}")
        return ids_slice

    def get_train_ids(self, num_sequences: int) -> List[int]:
        if self.params.reasoning_range is None:
            ids = self.rollouts.dataset.train_ids
        else:
            ids = self.rollouts.dataset.train_ids_within_range(
                *self.params.reasoning_range
            )
        ids = self.__slice_of_ids(ids, num_sequences)
        ids = list(
            filter(lambda id: not self.rollouts.has_train_sequence(id), ids),
        )
        return ids

    def get_test_ids(self, num_sequences: int) -> List[int]:
        if self.params.reasoning_range is None:
            ids = self.rollouts.dataset.test_ids
        else:
            ids = self.rollouts.dataset.test_ids_within_range(
                *self.params.reasoning_range
            )
        ids = self.__slice_of_ids(ids, num_sequences)
        ids = list(
            filter(lambda id: not self.rollouts.has_test_sequence(id), ids),
        )
        return ids

    def rollout_train(self, num_sequences: int, batch_size: int = 1):
        ids = self.get_train_ids(num_sequences)
        traces_bat = self.rollouts.dataset.train_it(ids, batch_size=1000)
        for i, trace in enumerate(chain.from_iterable(traces_bat)):
            logging.debug(f"Rolling out sequence {i + 1}.")
            rollout = self.rollout_trace(trace, batch_size=batch_size)
            self.rollouts.add_rollout_train(rollout)

    def rollout_test(self, num_sequences: int, batch_size: int = 1):
        ids = self.get_test_ids(num_sequences)
        traces_bat = self.rollouts.dataset.test_it(ids, batch_size=1000)
        for i, trace in enumerate(chain.from_iterable(traces_bat)):
            logging.debug(f"Rolling out sequence {i + 1}.")
            rollout = self.rollout_trace(trace, batch_size=batch_size)
            self.rollouts.add_rollout_test(rollout)


@click.group()
def main():
    pass


@main.command()
@click.option("--rollout-id", type=str, help="Rollout dataset id.")
def drop(rollout_id: str):
    """Drop rollout dataset from MongoDB."""
    logging.debug(f"Deleting rollout dataset {rollout_id}")
    data_store = RolloutDataStore()
    dataset = data_store.load_by_id(rollout_id)
    data_store.drop_dataset(dataset)
    logging.debug("Finished.")


def rollout(
    params: RolloutParameter,
    test_sequences: int = 100000,
    train_sequences: int = 1000000,
    include_train: bool = False,
    rank: int = 0,
    world_size: int = 1,
    batch_size: int = 4,
):
    """Generate rollout dataset.

    Args:
        params (RolloutParameter): Rollout dataset parameters. These parameters
            uniquely identify a particular rollout dataset. If the dataset
            already existis, this function will attempt to write additional
            data unless responses were already recorded in the dataset.
        test_sequences (int, optional): Number of test prompts to be included
            in dataset. Defaults to 100000.
        train_sequences (int, optional): Number of training prompts to be
            included in dataset. Defaults to 1000000.
        include_train (bool, optional): If True, evaluate training prompts. If
            False only evaluate test prompts and ignore `train_sequences`
            argument. Defaults to False.
        rank (int, optional): Worker index. Defaults to 0.
        world_size (int, optional): Number of workers. Defaults to 1.
        batch_size (int, optional): Batch size used for generating responses.
            Defaults to 4.
    """
    logging.debug(f"Launching worker: rank={rank}, world_size={world_size}")
    worker = RolloutWorker(params=params, rank=rank, world_size=world_size)
    worker.rollout_test(
        num_sequences=test_sequences,
        batch_size=batch_size,
    )
    if include_train:
        worker.rollout_train(
            num_sequences=train_sequences,
            batch_size=batch_size,
        )
    logging.debug("Finished rollout.")


@main.command()
@click.option(
    "--checkpoint-id",
    type=str,
    help="ID of checkpoint that is to be evaluated.",
)
@click.option(
    "--dataset-name",
    type=str,
    help="Token dataset to be used to evalute model.",
)
@click.option(
    "--rollout-len",
    type=int,
    help="Maximum number of tokens generated per prompt.",
)
@click.option(
    "--rollout-repeats",
    type=int,
    help="Number of responses generated per prompt.",
)
@click.option(
    "--test-sequences",
    type=int,
    default=100000,
    help="Number of test prompts included in rollout dataset.",
)
@click.option(
    "--train-sequences",
    type=int,
    default=1000000,
    help="Number of training prompts included in rollout dataset.",
)
@click.option(
    "--include-train",
    is_flag=True,
    help="""If used, rollouts are also generated for training prompts. 
If not set, the `--train-sequences` option has no effect.""",
)
@click.option("--rank", type=int, help="Rollout worker index.")
@click.option("--world-size", type=int, help="Number of rollout workers.")
@click.option(
    "--batch-size",
    type=int,
    default=4,
    help="Number of sequences that are generated in parallel on GPU.",
)
@click.option(
    "--prefix-len",
    type=int,
    default=0,
    help="Prefix length for generating response.",
)
@click.option(
    "--min-reasoning-len",
    type=int,
    default=-1,
    help="""Only use prompts where the A* execution trace included in the 
token dataset has at least `--min-reasoning-len` tokens.""",
)
@click.option(
    "--max-reasoning-len",
    type=int,
    default=-1,
    help="""Only use prompts where the A* execution trace included in the 
token dataset has at most `--max-reasoning-len` tokens.""",
)
def greedy(
    checkpoint_id: str,
    dataset_name: str,
    rollout_len: int,
    rollout_repeats: int,
    test_sequences: int,
    train_sequences: int,
    include_train: bool,
    rank: int,
    world_size: int,
    batch_size: int,
    prefix_len: int,
    min_reasoning_len: int,
    max_reasoning_len: int,
):
    """Launch rollout worker with greedy next token selection.

    The options `--min-reasoning-len` and `--max-reasoning-len` are set
    to constrain the dataset slice included this rollout dataset.
    """
    logging.info(f"Rolling out checkpoint {checkpoint_id}")
    params = RolloutParameter(
        checkpoint_id=checkpoint_id,
        dataset_name=dataset_name,
        sampler_name="greedy",
        sampler_kvargs=dict(),
        rollout_len=rollout_len,
        rollout_repeats=rollout_repeats,
        prefix_len=prefix_len,
        min_reasoning_len=min_reasoning_len,
        max_reasoning_len=max_reasoning_len,
    )
    rollout(
        params=params,
        test_sequences=test_sequences,
        train_sequences=train_sequences,
        include_train=include_train,
        rank=rank,
        world_size=world_size,
        batch_size=batch_size,
    )


@main.command()
@click.option(
    "--checkpoint-id",
    type=str,
    help="ID of checkpoint that is to be evaluated.",
)
@click.option(
    "--dataset-name",
    type=str,
    help="Token dataset to be used to evalute model.",
)
@click.option(
    "--rollout-len",
    type=int,
    help="Maximum number of tokens generated per prompt.",
)
@click.option(
    "--rollout-repeats",
    type=int,
    help="Number of responses generated per prompt.",
)
@click.option(
    "--test-sequences",
    type=int,
    default=100000,
    help="Number of test prompts included in rollout dataset.",
)
@click.option(
    "--train-sequences",
    type=int,
    default=1000000,
    help="Number of training prompts included in rollout dataset.",
)
@click.option(
    "--include-train",
    is_flag=True,
    help="""If used, rollouts are also generated for training prompts. 
If not set, the `--train-sequences` option has no effect.""",
)
@click.option("--rank", type=int, help="Rollout worker index.")
@click.option("--world-size", type=int, help="Number of rollout workers.")
@click.option(
    "--batch-size",
    type=int,
    default=4,
    help="Number of sequences that are generated in parallel on GPU.",
)
@click.option(
    "--prefix-len",
    type=int,
    default=0,
    help="Prefix length for generating response.",
)
@click.option(
    "--min-reasoning-len",
    type=int,
    default=-1,
    help="""Only use prompts where the A* execution trace included in the 
token dataset has at least `--min-reasoning-len` tokens.""",
)
@click.option(
    "--max-reasoning-len",
    type=int,
    default=-1,
    help="""Only use prompts where the A* execution trace included in the 
token dataset has at most `--max-reasoning-len` tokens.""",
)
def probability(
    checkpoint_id: str,
    dataset_name: str,
    rollout_len: int,
    rollout_repeats: int,
    test_sequences: int,
    train_sequences: int,
    include_train: bool,
    rank: int,
    world_size: int,
    batch_size: int,
    prefix_len: int,
    min_reasoning_len: int,
    max_reasoning_len: int,
):
    """Launch rollout worker that selects tokens from the predicted next-token
    probability distribution.

    The options `--min-reasoning-len` and `--max-reasoning-len` are set
    to constrain the dataset slice included this rollout dataset.
    """
    logging.info(f"Rolling out checkpoint {checkpoint_id}")
    params = RolloutParameter(
        checkpoint_id=checkpoint_id,  # type: ignore
        dataset_name=dataset_name,
        sampler_name="probability",
        sampler_kvargs=dict(),
        rollout_len=rollout_len,
        rollout_repeats=rollout_repeats,
        prefix_len=prefix_len,
        min_reasoning_len=min_reasoning_len,
        max_reasoning_len=max_reasoning_len,
    )
    rollout(
        params=params,
        test_sequences=test_sequences,
        train_sequences=train_sequences,
        include_train=include_train,
        rank=rank,
        world_size=world_size,
        batch_size=batch_size,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(levelname)s - %(asctime)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()
