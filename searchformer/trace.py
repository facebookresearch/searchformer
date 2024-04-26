# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import functools
import logging
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Sequence, Set

import click
import pandas as pd
import pymongo
from pymongo.collection import Collection
from torch import Tensor

from .utils import mongodb_client


@dataclass
class TokenizedTrace:
    """Dataclass holding a tokenized trace used for training.

    * `prompt`: Contains the task prompt sequence.
    * `reasoning`: Contains the execution trace for computing a plan.
    * `plan`: Contains the optimal solution plan.
    """

    id: int
    prompt: List[str]
    reasoning: List[str]
    plan: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "_id": self.id,
            "prompt": self.prompt,
            "reasoning": self.reasoning,
            "plan": self.plan,
        }

    def to_stats_dict(self) -> Dict[str, Any]:
        return {
            "_id": self.id,
            "prompt_len": len(self.prompt),
            "reasoning_len": len(self.reasoning),
            "plan_len": len(self.plan),
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "TokenizedTrace":
        if "_id" in d.keys():
            return TokenizedTrace(
                id=d["_id"],
                prompt=d["prompt"],
                reasoning=d["reasoning"],
                plan=d["plan"],
            )
        else:
            return TokenizedTrace(**d)


class CannotTokenizeException(Exception):
    pass


class Tokenizer:
    """Super-class used to implement a tokenizer mapping A* execution traces
    and tasks to a `TokenizedTrace`.
    """

    def __init__(self, vocabulary: Iterable[str]):
        vocabulary_list = list(vocabulary)
        vocabulary_list.sort()
        self.__vocabulary = tuple(vocabulary_list)

    @property
    def vocabulary(self) -> Set[str]:
        return set(self.__vocabulary)

    @abstractmethod
    def tokenize(self, trace: Any, is_test: bool = False) -> TokenizedTrace:
        """Tokenize trace.

        Args:
            trace (Any): _description_
            is_test (bool, optional): _description_. Defaults to False.

        Throws:
            CannotTokenizeException: If the provided trace cannot be tokenized.

        Returns:
            TokenizedTrace: _description_
        """
        pass

    def __call__(self, trace: Any, is_test: bool = False) -> TokenizedTrace:
        """Tokenize trace.

        Args:
            trace (Any): _description_
            is_test (bool, optional): _description_. Defaults to False.

        Throws:
            CannotTokenizeException: If the provided trace cannot be tokenized.

        Returns:
            TokenizedTrace: _description_
        """
        tok_trace = self.tokenize(trace, is_test)
        vocabulary = self.vocabulary
        assert vocabulary.issuperset(
            tok_trace.prompt
        ), "Tokenized prompt outside vocabulary."
        assert vocabulary.issuperset(
            tok_trace.reasoning
        ), "Tokenized reasoning trace outside vocabulary."
        assert vocabulary.issuperset(
            tok_trace.plan
        ), "Tokenized plan outside vocabulary."
        return tok_trace


def _tok_seq_iterator(
    collection: Collection, ids: List[int], batch_size: int = 1
) -> Iterator[Iterable[TokenizedTrace]]:
    logging.debug(f"Iterating over {len(ids)} ids.")
    for i in range(0, len(ids), batch_size):
        result = collection.find({"_id": {"$in": ids[i : i + batch_size]}})
        trace_it = map(TokenizedTrace.from_dict, result)
        yield list(trace_it)


def _load_ids(collection: Collection) -> List[Any]:
    logging.debug(f"Loading all ids from {collection} ...")
    id_list = [d["_id"] for d in collection.find({}, {"_id": 1})]
    logging.debug("Finished loading.")
    return id_list


def _load_ids_within_reasoning_len_range(
    trace_meta_collection: Collection, min_len: int, max_len: int
) -> List[Any]:
    """Loads all ids from the provided trace meta data collection and filters
    them by the `reasoning_len` field.

    Args:
        trace_collection (Collection): trace meta data collection.
        min_len (int): Minimum trace length.
        max_len (int): Maximum trace length.

    Returns:
        List[Any]: _description_
    """
    res_it = trace_meta_collection.find(
        {
            "$and": [
                {"reasoning_len": {"$gte": min_len}},
                {"reasoning_len": {"$lt": max_len}},
            ],
        },
        {"_id": 1},
    )
    return [r["_id"] for r in res_it]


def max_seq_len_from_collection(collection: Collection) -> Dict[str, Any]:
    """Returns prompt, reasoning, and plan sequence lengths.

    Args:
        collection (Collection): Collection containing `TokenizedTrace`
            documents.

    Returns:
        Dict[str, Any]: Resulting statistics.
    """
    res_it = collection.aggregate(
        [
            {
                "$facet": {
                    "max_prompt_len": [
                        {
                            "$group": {
                                "_id": None,
                                "max": {"$max": "$prompt_len"},
                            },
                        }
                    ],
                    "max_reasoning_len": [
                        {
                            "$group": {
                                "_id": None,
                                "max": {"$max": "$reasoning_len"},
                            },
                        }
                    ],
                    "max_plan_len": [
                        {
                            "$group": {
                                "_id": None,
                                "max": {"$max": "$plan_len"},
                            },
                        }
                    ],
                }
            }
        ]
    )
    res = next(iter(res_it))
    return {k: v[0]["max"] for k, v in res.items()}


TOK_TRACE_DB_NAME = "tokenSeqDB"


class TokenizedDataset:
    """Used to read and write to a tokenized dataset with a particular name."""

    def __init__(self, name: str):
        self.name = name
        self.client = mongodb_client()
        self.db = self.client[TOK_TRACE_DB_NAME]

    @functools.cached_property
    def vocabulary_collection(self) -> Collection:
        return self.db["vocabulary"]

    def exists(self) -> bool:
        result = self.vocabulary_collection.find_one({"_id": self.name})
        return result is not None

    def add_vocabulary(self, vocabulary: Set[str]):
        try:
            self.vocabulary_collection.insert_one(
                {"_id": self.name, "vocabulary": list(set(vocabulary))}
            )
        except pymongo.errors.DuplicateKeyError:
            logging.warning(
                f"Cannot set vocabulary for existing dataset {self.name}",
            )

    @property
    def vocabulary(self) -> List[str]:
        """Vocabulary used in this dataset. The `prompt`, `reasoning`, and
        `plan` sequences are expressed using the same vobulary.

        Returns:
            List[str]: Vocabulary as a list of strings. Each string is a token.
        """
        res = self.vocabulary_collection.find_one({"_id": self.name})
        assert res is not None, f"Cannot find vocabulary for {self.name}"
        return res["vocabulary"]

    @functools.cached_property
    def log_collection(self) -> Collection:
        return self.db[f"{self.name}.log"]

    def claim_id(self, id: int) -> bool:
        """Returns true if the id is claimed by the worker. This function is
        not thread safe and can only be called by a single worker.

        Args:
            id (int): Sequence id.

        Returns:
            bool: True if claimed, false if not.
        """
        try:
            self.log_collection.insert_one({"_id": id, "state": "incomplete"})
        except pymongo.errors.DuplicateKeyError:
            res = self.log_collection.find_one({"_id": id})
            assert type(res) is dict
            if res.get("state", "incomplete") == "complete":
                return False
        return True

    def finish_id(self, id: int):
        self.log_collection.update_one(
            {"_id": {"$eq": id}}, {"$set": {"state": "complete"}}
        )

    @functools.cached_property
    def train_seq_collection(self) -> Collection:
        return self.db[f"{self.name}.seq.train"]

    @functools.cached_property
    def test_seq_collection(self) -> Collection:
        return self.db[f"{self.name}.seq.test"]

    @functools.cached_property
    def train_meta_collection(self) -> Collection:
        return self.db[f"{self.name}.meta.train"]

    @functools.cached_property
    def test_meta_collection(self) -> Collection:
        return self.db[f"{self.name}.meta.test"]

    def add_train_trace(self, trace: TokenizedTrace):
        self.train_seq_collection.insert_one(trace.to_dict())
        self.train_meta_collection.insert_one(trace.to_stats_dict())

    def drop_train_trace(self, id_list: List[int]):
        self.train_seq_collection.delete_many({"_id": {"$in": id_list}})
        self.train_meta_collection.delete_many({"_id": {"$in": id_list}})

    def add_test_trace(self, trace: TokenizedTrace):
        self.test_seq_collection.insert_one(trace.to_dict())
        self.test_meta_collection.insert_one(trace.to_stats_dict())

    def drop_test_trace(self, id_list: List[int]):
        self.test_seq_collection.delete_many({"_id": {"$in": id_list}})
        self.test_meta_collection.delete_many({"_id": {"$in": id_list}})

    def drop(self):
        """Drop the tokenized dataset from MongoDB."""
        self.log_collection.drop()
        self.train_seq_collection.drop()
        self.test_seq_collection.drop()
        self.train_meta_collection.drop()
        self.test_meta_collection.drop()
        self.vocabulary_collection.delete_one({"_id": self.name})

    @property
    def train_ids(self) -> List[int]:
        return [int(i) for i in _load_ids(self.train_meta_collection)]

    @property
    def test_ids(self) -> List[int]:
        return [int(i) for i in _load_ids(self.test_meta_collection)]

    def train_ids_within_range(self, min_len: int, max_len: int) -> List[int]:
        id_list = _load_ids_within_reasoning_len_range(
            self.train_meta_collection, min_len, max_len
        )
        return [int(i) for i in id_list]

    def test_ids_within_range(self, min_len: int, max_len: int) -> List[int]:
        id_list = _load_ids_within_reasoning_len_range(
            self.test_meta_collection, min_len, max_len
        )
        return [int(i) for i in id_list]

    def train_it(
        self, ids: List[int], batch_size: int = 1
    ) -> Iterator[Iterable[TokenizedTrace]]:
        return _tok_seq_iterator(self.train_seq_collection, ids, batch_size)

    def test_it(
        self, ids: List[int], batch_size: int = 1
    ) -> Iterator[Iterable[TokenizedTrace]]:
        return _tok_seq_iterator(self.test_seq_collection, ids, batch_size)

    def max_seq_lens_train(self) -> Dict[str, Any]:
        return max_seq_len_from_collection(self.train_meta_collection)

    def max_seq_lens_test(self) -> Dict[str, Any]:
        return max_seq_len_from_collection(self.test_meta_collection)

    def max_seq_lens_df(self) -> pd.DataFrame:
        train_lens = self.max_seq_lens_train()
        train_lens["set"] = "train"
        test_lens = self.max_seq_lens_test()
        test_lens["set"] = "test"
        return pd.DataFrame([train_lens, test_lens])


def find_tokenized_datasets() -> List[str]:
    """List all tokenized dataset names.

    Returns:
        List[str]: Tokenized dataset names.
    """
    client = mongodb_client()
    res_it = client[TOK_TRACE_DB_NAME].vocabulary.find({}, {"_id": 1})
    id_list = [res["_id"] for res in res_it]
    id_list.sort()
    return id_list


@dataclass
class AStarTrace:
    """Tokenized A* trace dataclass holding tensors used for training."""

    prompt: Tensor
    trace_plan: Tensor
    plan_start: int


class DictTokenizer:
    def __init__(self, vocabulary: List[str]):
        """Constructs a tokenizer mapping a sequence of string tokens into a
        sequence of indices.

        Args:
            vocabulary (List[str]): Token vocabulary.
        """
        vocabulary.sort()
        vocabulary = ["bos", "eos"] + vocabulary
        logging.info(f"Vocabulary size: {len(vocabulary)}")
        self.token_to_idx: Dict[str, int] = {}
        self.idx_to_token: Dict[int, str] = {}
        for i, token in enumerate(vocabulary):
            self.token_to_idx[token] = i
            self.idx_to_token[i] = token

    @property
    def bos(self) -> int:
        return self.token_to_idx["bos"]

    @property
    def eos(self) -> int:
        return self.token_to_idx["eos"]

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_idx)

    def encode(self, tokens: Sequence[str]) -> List[int]:
        idxs: List[int] = []
        for token in tokens:
            idxs.append(self.token_to_idx[token])
        return idxs

    def decode(self, idxs: Sequence[int]) -> List[str]:
        tokens: List[str] = []
        for i in idxs:
            tokens.append(self.idx_to_token[i])
        return tokens

    def tokenize(
        self,
        trace: TokenizedTrace,
        plan_only: bool = False,
    ) -> AStarTrace:
        prompt_tok = ["bos", *trace.prompt, "eos"]
        prompt_idx = self.encode(prompt_tok)

        if plan_only:
            trace_tok = ["bos", *trace.plan, "eos"]
            plan_start = 1
        else:
            trace_tok = ["bos", *trace.reasoning, *trace.plan, "eos"]
            plan_start = len(trace.reasoning) + 1
        trace_idx = self.encode(trace_tok)
        return AStarTrace(
            prompt=Tensor(prompt_idx).long(),
            trace_plan=Tensor(trace_idx).long(),
            plan_start=plan_start,
        )

    def tokenize_batch(
        self, trace_it: Iterable[TokenizedTrace], plan_only: bool = False
    ) -> List[AStarTrace]:
        tok_list: List[AStarTrace] = []
        for trace in trace_it:
            tok_list.append(self.tokenize(trace, plan_only=plan_only))
        return tok_list

    def tokenize_prompt(self, trace: TokenizedTrace) -> Tensor:
        prompt_tok = ["bos", *trace.prompt, "eos"]
        prompt_idx = self.encode(prompt_tok)
        return Tensor(prompt_idx).long()


@click.group()
def main():
    pass


@main.command()
@click.option("--name", type=str, help="Dataset name.")
def drop_dataset(name: str):
    """Deletes token dataset by name."""
    TokenizedDataset(name).drop()


@main.command()
def list_token_datasets():
    """Lists all token datasets."""
    vocab_coll = mongodb_client()[TOK_TRACE_DB_NAME]["vocabulary"]
    id_it = map(lambda d: d["_id"], vocab_coll.find({}, {"_id": 1}))
    print("\n".join(id_it))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(levelname)s - %(asctime)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()
