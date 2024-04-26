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
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
)

import click
import pymongo
from pymongo.collection import Collection

from .astar import AStarCannotSolveTaskException, AStarState, TraceStep, astar
from .rollout import (
    Rollout,
    RolloutDataStore,
    RolloutEvaluation,
    RolloutEvaluationBatch,
)
from .trace import (
    CannotTokenizeException,
    TokenizedDataset,
    TokenizedTrace,
    Tokenizer,
)
from .utils import mongodb_client


@dataclass
class GridPos:
    x: int
    y: int

    def __hash__(self) -> int:
        assert self.x < 100000
        return int(self.x + 100000 * self.y)

    def to_idx(self, width: int) -> int:
        return self.x + self.y * width

    @staticmethod
    def from_idx(idx: int, width: int) -> "GridPos":
        return GridPos(x=idx % width, y=idx // width)

    def to_dict(self) -> dict:
        return {"x": self.x, "y": self.y}

    @staticmethod
    def from_dict(pos_dict: Dict[str, int]) -> "GridPos":
        return GridPos(x=int(pos_dict["x"]), y=int(pos_dict["y"]))

    @property
    def up(self) -> "GridPos":
        return GridPos(self.x, self.y + 1)

    @property
    def down(self) -> "GridPos":
        return GridPos(self.x, self.y - 1)

    @property
    def left(self) -> "GridPos":
        return GridPos(self.x - 1, self.y)

    @property
    def right(self) -> "GridPos":
        return GridPos(self.x + 1, self.y)

    def __eq__(self, other: object) -> bool:
        assert isinstance(other, GridPos)
        return self.x == other.x and self.y == other.y

    def distance(self, other: "GridPos") -> int:
        x_dist = abs(self.x - other.x)
        y_dist = abs(self.y - other.y)
        return x_dist + y_dist


@dataclass
class MazeSpec:
    """Dataclass to specify a Maze planning task."""

    width: int
    height: int
    start: GridPos
    goal: GridPos
    walls: Sequence[GridPos]

    @property
    def int_seq(self) -> Tuple[int, ...]:
        return (
            self.width,
            self.height,
            self.start.to_idx(self.width),
            self.goal.to_idx(self.width),
            *self.wall_int_seq,
        )

    @property
    def wall_int_seq(self) -> Tuple[int, ...]:
        wall_idxs = [p.to_idx(self.width) for p in self.walls]
        wall_idxs.sort()
        return tuple(wall_idxs)

    @property
    def id(self) -> str:
        return "-".join(map(str, self.int_seq))

    @property
    def spec_hash(self) -> int:
        return hash("-".join(map(str, self.int_seq)))

    @property
    def maze_hash(self) -> int:
        return hash("-".join(map(str, self.wall_int_seq)))

    def __hash__(self) -> int:
        return self.spec_hash

    @staticmethod
    def from_int_seq(int_seq: Sequence[int]) -> "MazeSpec":
        width, height, idx_start, idx_goal = int_seq[:4]
        return MazeSpec(
            width=width,
            height=height,
            start=GridPos.from_idx(idx_start, width),
            goal=GridPos.from_idx(idx_goal, width),
            walls=[GridPos.from_idx(i, width) for i in int_seq[4:]],
        )

    def is_pos_in_spec(self, pos: GridPos) -> bool:
        x_pos_range = pos.x >= 0 and pos.x < self.width
        y_pos_range = pos.y >= 0 and pos.y < self.height
        is_wall = False
        for wall in self.walls:
            is_wall = is_wall or (wall == pos)
        return x_pos_range and y_pos_range and not is_wall

    def is_pos_in_wall(self, pos: GridPos) -> bool:
        for wall in self.walls:
            if pos == wall:
                return True
        return False

    def neighborhood(self, pos: GridPos) -> Sequence[GridPos]:
        neigh: List[GridPos] = []
        up_pos = pos.up
        if self.is_pos_in_spec(up_pos):
            neigh.append(up_pos)
        down_pos = pos.down
        if self.is_pos_in_spec(down_pos):
            neigh.append(down_pos)
        left_pos = pos.left
        if self.is_pos_in_spec(left_pos):
            neigh.append(left_pos)
        right_pos = pos.right
        if self.is_pos_in_spec(right_pos):
            neigh.append(right_pos)
        return neigh

    def to_dict(self) -> dict:
        return {
            "width": self.width,
            "height": self.height,
            "start": self.start.to_dict(),
            "goal": self.goal.to_dict(),
            "walls": [w.to_dict() for w in self.walls],
        }

    @staticmethod
    def from_dict(spec_dict: Dict[str, Any]) -> "MazeSpec":
        return MazeSpec(
            width=int(spec_dict["width"]),
            height=int(spec_dict["height"]),
            start=GridPos.from_dict(spec_dict["start"]),
            goal=GridPos.from_dict(spec_dict["goal"]),
            walls=[GridPos.from_dict(w) for w in spec_dict["walls"]],
        )

    @property
    def boundary_walls(self) -> Sequence[GridPos]:
        w, h = self.width, self.height
        top = [GridPos(x, h) for x in range(-1, w + 1)]
        bottom = [GridPos(x, -1) for x in range(-1, w + 1)]
        left = [GridPos(-1, y) for y in range(-1, h + 1)]
        right = [GridPos(w, y) for y in range(-1, h + 1)]
        return top + bottom + left + right


class AStarMazeState(AStarState):
    """A* state implementation for maze planning tasks."""

    def __init__(
        self,
        maze_spec: MazeSpec,
        position: Optional[GridPos] = None,
        cost_from_start: float = 0.0,
        parent: Optional["AStarMazeState"] = None,
        deterministic: bool = True,
    ):
        super().__init__(parent, deterministic=deterministic)
        if position is None:
            position = maze_spec.start
        self.position = position
        self.maze_spec = maze_spec
        self._cost_from_start = cost_from_start

    @property
    def state(self) -> Dict[str, Any]:
        return {"x": self.position.x, "y": self.position.y}

    def _get_children(self) -> List["AStarState"]:
        child_list: List["AStarState"] = []
        for child_pos in self.maze_spec.neighborhood(self.position):
            child_list.append(
                AStarMazeState(
                    maze_spec=self.maze_spec,
                    position=child_pos,
                    cost_from_start=self.cost_from_start + 1,
                    parent=self,
                    deterministic=self.deterministic,
                )
            )
        # random.shuffle(children)
        return child_list

    @property
    def heuristic(self) -> float:
        return float(self.position.distance(self.maze_spec.goal))

    @property
    def cost_from_start(self) -> float:
        return self._cost_from_start

    def __hash__(self) -> int:
        return hash(self.position)

    @property
    def is_goal(self) -> bool:
        return self.position == self.maze_spec.goal


@dataclass
class MazeTrace:
    """Data class holding an execution trace of a Maze planning task.

    This class is used for storing execution trace (before tokenization) in
    MongoDB. The hashing function implemented in this class determines
    uniqueness in the constructed dataset. This value is used as a
    document id by MongoDB to prevent insertion of duplicate tasks.
    """

    spec: MazeSpec
    trace: Sequence[TraceStep]

    def __hash__(self) -> int:
        return self.spec.__hash__()

    # @staticmethod
    # def from_grid_spec(
    #     spec: MazeSpec, deterministic: bool = True
    # ) -> "MazeTrace":
    #     trace = astar(AStarMazeState(spec), deterministic=deterministic)
    #     return MazeTrace(spec=spec, trace=trace)

    @staticmethod
    def from_int_seq(
        int_seq: Sequence[int],
        deterministic: bool = True,
    ) -> "MazeTrace":
        spec = MazeSpec.from_int_seq(int_seq)
        trace = astar(AStarMazeState(spec, deterministic=deterministic))
        return MazeTrace(spec=spec, trace=trace)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "spec": self.spec.to_dict(),
            "trace": [t.to_dict() for t in self.trace],
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "MazeTrace":
        return MazeTrace(
            spec=MazeSpec.from_dict(d["spec"]),
            trace=[TraceStep.from_dict(t_d) for t_d in d["trace"]],
        )

    @property
    def plan(self) -> Sequence[TraceStep]:
        return [s for s in self.trace if s.action == "plan"]

    @property
    def plan_start_correct(self) -> bool:
        return GridPos(**self.plan[0].state) == self.spec.start

    @property
    def plan_reaches_goal(self) -> bool:
        return GridPos(**self.plan[-1].state) == self.spec.goal

    @property
    def plan_enters_wall(self) -> bool:
        pos_it = map(lambda n: GridPos(**n.state), self.plan)
        return any(map(self.spec.is_pos_in_wall, pos_it))

    @property
    def plan_steps_distances_correct(self) -> bool:
        plan = self.plan
        for p1, p2 in zip(plan[:-1], plan[1:]):
            if GridPos(**p1.state).distance(GridPos(**p2.state)) != 1:
                return False
        return True

    @property
    def reasoning_trace(self) -> Sequence[TraceStep]:
        return [s for s in self.trace if s.action != "plan"]


MAZE_DB_NAME = "mazeDB"


class MazeTraceDataset:
    """Class for storing and reading Maze execution traces to MongoDB."""

    def __init__(self, name: str):
        self.name = name
        self.client = mongodb_client()
        self.db = self.client[MAZE_DB_NAME]

    @functools.cached_property
    def trace_collection(self) -> Collection:
        return self.db[f"{self.name}.trace"]

    @functools.cached_property
    def index_collection(self) -> Collection:
        return self.db[f"{self.name}.index"]

    def drop(self):
        self.db.drop_collection(self.trace_collection)
        self.db.drop_collection(self.index_collection)

    def generate(
        self,
        is_test: bool,
        width: int,
        height: int,
        deterministic: bool,
        num_attempts: int = 100,
    ) -> int:
        """Randomly generate a maze execution trace and return the number of
        generated traces.

        Args:
            is_test (bool): Insert trace into test set. If false insert into
                training set.
            width (int): Maze width.
            height (int): Maze height.
            deterministic (bool): Use deterministic A* if true, otherwise use
                nondeterministic A* search.
            num_attempts (int, optional): Maximum number of attempts of
                generating a unique maze planning task. Defaults to 100.

        Returns:
            int: Number of unique maze planning tasks that were generated.
        """
        indices = list(range(width * height))
        random.shuffle(indices)
        num_walls = random.randint(
            a=len(indices) // 10 * 3,
            b=len(indices) // 10 * 5,
        )
        indices_wall = indices[:num_walls]
        indices_wall.sort()
        indices_start_goal = indices[num_walls:]

        maze_hash: Optional[int] = None
        trace_dict: Dict[str, Dict[str, Any]] = {}
        for _ in range(num_attempts):
            random.shuffle(indices_start_goal)
            start, goal = indices_start_goal[:2]
            int_seq = [width, height, start, goal, *indices_wall]
            try:
                trace = MazeTrace.from_int_seq(int_seq, deterministic)
            except AStarCannotSolveTaskException:
                continue
            if not trace.plan_reaches_goal:
                continue
            if len(trace.plan) < max(width, height):
                continue
            assert not trace.plan_enters_wall
            assert trace.plan_start_correct
            assert trace.plan_steps_distances_correct

            trace_hash = trace.spec.id
            if maze_hash is None:
                maze_hash = trace.spec.maze_hash
            if trace_hash not in trace_dict.keys():
                trace_dict[trace_hash] = trace.to_dict()
        if len(trace_dict) == 0:
            return 0
        trace_hash_list = list(trace_dict.keys())

        try:
            self.trace_collection.insert_one(
                {"_id": maze_hash, "is_test": is_test, "batch": trace_dict}
            )
            self.index_collection.insert_one(
                {
                    "_id": maze_hash,
                    "is_test": is_test,
                    "trace_id_list": trace_hash_list,
                }
            )
            return len(trace_hash_list)
        except pymongo.errors.DuplicateKeyError:  # type: ignore
            return 0

    @property
    def index_list(self) -> List[int]:
        collection_it = self.index_collection.find({}, {"_id": 1})
        index_it = map(lambda d: d["_id"], collection_it)
        return list(index_it)

    def iterate_traces(
        self, rank: int = 0, world_size: int = 1, load_batch_size: int = 1000
    ) -> Iterator[Tuple[MazeTrace, bool]]:
        """Maze trace iterator.

        Only iterates over a portion of the data set that corresponds to the
        specified worker rank and world size.

        Args:
            rank (int, optional): Worker index ranging from [0, n-1]. Defaults
                to 0.
            world_size (int, optional): Number of workers. Defaults to 1.
            load_batch_size (int, optional): Batch size used for bulk loading
                of execution traces. Defaults to 1000.

        Yields:
            Iterator[Tuple[MazeTrace, bool]]: Iterator generating instances of
                MazeTrace class and booleans. If the boolean is true, then the
                trace belongs to the test set, otherwise it belongs to the
                training set.
        """
        index_list = self.index_list
        index_list.sort()
        logging.debug(f"Total number of mazes: {len(index_list)}")

        slice_size = math.ceil(len(index_list) / world_size)
        index_list = index_list[rank * slice_size : (rank + 1) * slice_size]
        logging.debug(f"Iterating over {len(index_list)} mazes.")

        collection = self.trace_collection
        for i in range(0, len(index_list), load_batch_size):
            index_batch = index_list[i : i + load_batch_size]
            for res in collection.find({"_id": {"$in": index_batch}}):
                is_test = bool(res["is_test"])
                for trace_dict in res["batch"].values():
                    trace = MazeTrace.from_dict(trace_dict)
                    yield trace, is_test

    def sample(self, n: int) -> Iterable[Tuple[MazeTrace, bool]]:
        res: List[Tuple[MazeTrace, bool]] = []
        for trace_dict in self.trace_collection.find():
            is_test = bool(trace_dict.get("is_test", False))
            for trace_dict in trace_dict.get("batch", []).values():
                trace = MazeTrace.from_dict(trace_dict)
                res.append((trace, is_test))

                if len(res) >= n:
                    return res
        return res


@click.group()
def main():
    pass


@main.command()
@click.option("--name", type=str, help="Maze trace dataset name.")
def drop_dataset(name: str):
    """Drop a maze trace dataset."""
    MazeTraceDataset(name).drop()


def fixed_size_maze_dataset_name(
    width: int,
    height: int,
    deterministic: bool,
) -> str:
    if deterministic:
        name = f"maze.{width}-by-{height}-deterministic"
    else:
        name = f"maze.{width}-by-{height}-nondeterministic"
    return name


@main.command()
@click.option("--width", type=int, default=10, help="Maze width.")
@click.option("--height", type=int, default=10, help="Maze height.")
@click.option("--deterministic", is_flag=True, help="Use deterministic A*.")
@click.option(
    "--num-samples", type=int, default=1000, help="Number of execution graces."
)
@click.option(
    "--num-attempts-per-maze",
    type=int,
    default=100,
    help="""Number of times the same maze (wall placement only) should be used 
    to create a unique navigation task. Mazes cannot be re-used between 
    training and test data.""",
)
def generate(
    width: int,
    height: int,
    deterministic: bool,
    num_samples: int,
    num_attempts_per_maze: int,
):
    """Generate a maze trace dataset by running A* search. If a dataset
    already exists, this keeps generating more traces.

    Note that the inserted maze planning tasks are always unique. Tasks
    cannot be duplicated.

    Args:
        width (int): Maze width.
        height (int): Maze height.
        deterministic (bool): Use deterministic A*.
        num_samples (int): Number of execution graces.
        num_attempts_per_maze (int): Number of times the same maze (wall
            placement only) should be used to create a unique navigation
            task. Mazes cannot be re-used between training and test data.
    """
    name = fixed_size_maze_dataset_name(width, height, deterministic)
    dataset = MazeTraceDataset(name)
    logging.info("Starting maze generation ...")
    samples_stored = 0
    while samples_stored < num_samples:
        if samples_stored < 9 * num_samples // 10:
            is_test = False
        else:
            is_test = True
        samples_stored += dataset.generate(
            is_test=is_test,
            width=width,
            height=height,
            deterministic=deterministic,
            num_attempts=num_attempts_per_maze,
        )
        logging.info(f"Generated {samples_stored} traces for {dataset.name}.")
    logging.info("Done.")


class SimpleMazeTokenizer(Tokenizer):
    """Tokenizer mapping a maze execution trace to a sequence of tokens."""

    def __init__(self, width: int, height: int):
        vocabulary: List[str] = []
        vocabulary += [str(i) for i in range(max(width, height))]
        vocabulary += [f"c{i}" for i in range(width * height)]
        vocabulary += ["start", "goal", "wall", "create", "close", "plan"]
        super().__init__(vocabulary)

    def tokenize(
        self,
        trace: MazeTrace,
        is_test: bool = False,
    ) -> TokenizedTrace:
        prompt = [
            "start",
            str(trace.spec.start.x),
            str(trace.spec.start.y),
            "goal",
            str(trace.spec.goal.x),
            str(trace.spec.goal.y),
        ]
        for wall in trace.spec.walls:
            prompt += ["wall", str(wall.x), str(wall.y)]

        reasoning: List[str] = []
        for step in trace.reasoning_trace:
            position = GridPos(**step.state)
            assert not math.isinf(step.cost_from_start)
            assert not math.isnan(step.cost_from_start)
            assert not math.isinf(step.heuristic)
            assert not math.isnan(step.heuristic)
            reasoning += [
                step.action,
                str(position.x),
                str(position.y),
                f"c{int(step.cost_from_start)}",
                f"c{int(step.heuristic)}",
            ]

        plan: List[str] = []
        for step in trace.plan:
            position = GridPos(**step.state)
            plan += [
                step.action,
                str(position.x),
                str(position.y),
            ]

        return TokenizedTrace(
            id=hash(trace),
            prompt=prompt,
            reasoning=reasoning,
            plan=plan,
        )


@main.command()
@click.option("--width", type=int, default=10, help="Maze width.")
@click.option("--height", type=int, default=10, help="Maze height.")
@click.option("--deterministic", is_flag=True, help="Use deterministic A*.")
@click.option(
    "--rank",
    type=int,
    default=0,
    help="Worker rank raning from [0, n-1].",
)
@click.option("--world-size", type=int, default=1, help="Number of workers.")
def tokenize(
    width: int,
    height: int,
    deterministic: bool,
    rank: int,
    world_size: int,
):
    """Tokenize a previously generate maze execution trace dataset to a token
    sequence dataset that can be used for training.
    """
    logging.info(f"rank={rank}, world_size={world_size}")
    name = fixed_size_maze_dataset_name(width, height, deterministic)
    maze_dataset = MazeTraceDataset(name)
    tokenizer = SimpleMazeTokenizer(width, height)
    tok_dataset = TokenizedDataset(f"{maze_dataset.name}.simple")
    tok_dataset.add_vocabulary(tokenizer.vocabulary)

    total_traces = 0
    for trace, is_test in maze_dataset.iterate_traces(rank, world_size):
        if tok_dataset.claim_id(hash(trace)):
            try:
                tok_trace = tokenizer(trace, is_test)
                if not is_test:
                    tok_dataset.add_train_trace(tok_trace)
                else:
                    tok_dataset.add_test_trace(tok_trace)
                tok_dataset.finish_id(hash(trace))
                total_traces += 1
            except CannotTokenizeException:
                pass
            if total_traces % 100 == 0:
                logging.info(f"Tokenized {total_traces} traces.")

    logging.info(
        f"Finished tokenizing {total_traces} "
        + f"traces for dataset {tok_dataset.name}"
    )


@dataclass
class TokGridPos:
    """Used for parsing plans from token sequences."""

    tok: str
    pos: GridPos

    @staticmethod
    def from_tok_sequence(tok_seq: List[str]) -> List["TokGridPos"]:
        if len(tok_seq) % 3 > 0:
            raise ValueError("Syntax of token sequence not correct.")
        seq: List["TokGridPos"] = []
        for i in range(0, len(tok_seq), 3):
            tok = tok_seq[i]
            x = int(tok_seq[i + 1])
            y = int(tok_seq[i + 2])
            seq.append(TokGridPos(tok=tok, pos=GridPos(x, y)))
        return seq


def evaluate_rollout_single(
    start: GridPos,
    goal: GridPos,
    walls: List[GridPos],
    rollout_tok: List[str],
) -> RolloutEvaluation:
    """Evaluate if a token sequence contains a correct or optimal plan.

    Args:
        start (GridPos): Start position.
        goal (GridPos): Goal position.
        walls (List[GridPos]): Number of wall positions.
        rollout_tok (List[str]): Token sequence that is evaluated.

    Returns:
        RolloutEvaluation: Evaluation result.
    """
    has_eos = rollout_tok[-1] == "eos"
    if has_eos:
        rollout_tok = rollout_tok[:-1]

    try:
        plan_start = rollout_tok.index("plan")
        trace_tokens = plan_start - 1
        plan_tok = rollout_tok[plan_start:]

        plan_tok_pos_list = TokGridPos.from_tok_sequence(plan_tok)
        plan_length = len(plan_tok_pos_list)
        syntax_correct = all(map(lambda t: t.tok == "plan", plan_tok_pos_list))
        plan_pos = [t.pos for t in plan_tok_pos_list]
        corr_start = start == plan_pos[0]
        corr_goal = goal == plan_pos[-1]
        corr_walls = not any([pos in walls for pos in plan_pos])
        p1_p2_it = zip(plan_pos[1:], plan_pos[:-1])
        corr_dist = all([p1.distance(p2) == 1 for p1, p2 in p1_p2_it])
        correct_plan = corr_dist and corr_walls and corr_start and corr_goal
    except ValueError:
        plan_length = 0
        trace_tokens = 0
        syntax_correct = False
        corr_start = False
        corr_goal = False
        correct_plan = False

    return RolloutEvaluation(
        has_eos=has_eos,
        plan_length=plan_length,
        plan_syntax_correct=syntax_correct,
        plan_correct_start=corr_start,
        plan_correct_goal=corr_goal,
        plan_correct=correct_plan,
        trace_tokens=trace_tokens,
    )


def evaluate_rollout(rollout: Rollout) -> RolloutEvaluationBatch:
    """Evaluate multiple rollouts.

    Args:
        rollout (Rollout): Rollout batch.

    Returns:
        RolloutEvaluationBatch: Batch evaluation result.
    """
    tok_seq = TokGridPos.from_tok_sequence(rollout.prompt)
    start = next(iter(filter(lambda t: t.tok == "start", tok_seq))).pos
    goal = next(iter(filter(lambda t: t.tok == "goal", tok_seq))).pos
    wall_tok_seq = filter(lambda t: t.tok == "wall", tok_seq)
    walls = list(map(lambda t: t.pos, wall_tok_seq))
    opt_plan_len = len(TokGridPos.from_tok_sequence(rollout.plan))

    eval_list: List[RolloutEvaluation] = []
    for rollout_tok in rollout.rollouts:
        eval_list.append(
            evaluate_rollout_single(
                start=start,
                goal=goal,
                walls=walls,
                rollout_tok=rollout_tok,
            )
        )
    return RolloutEvaluationBatch(
        id=rollout.id,
        optimal_plan_length=opt_plan_len,
        reasoning_length=len(rollout.reasoning),
        rollout=eval_list,
    )


@main.command()
@click.option("--rollout-id", type=str, help="ID of rollout dataset.")
@click.option(
    "--origin-dataset",
    type=str,
    help="Token sequence dataset used to generate rollout dataset.",
)
@click.option(
    "--rank",
    type=int,
    default=0,
    help="Worker rank raning form [0, n-1].",
)
@click.option("--world-size", type=int, default=1, help="Number of workers.")
def reduce_rollout_to_shortest_trace(
    rollout_id: str,
    origin_dataset: str,
    rank: int,
    world_size: int,
):
    """Map a rollout dataset to a token sequence dataset.

    This function is used for the Searchformer experiments. First all
    generated sequences are parsed and the shortest sequence containing an
    optimal plan is used to construct the new shorter sequence training
    dataset. If no optimal plan is found, the original training sequence
    is re-used.
    """
    rollout_dataset = RolloutDataStore().load_by_id(rollout_id)
    org_dataset = TokenizedDataset(origin_dataset)
    tok_dataset = TokenizedDataset(f"{rollout_id}.improved")
    tok_dataset.add_vocabulary(set(org_dataset.vocabulary))
    rollout_it = rollout_dataset.rollout_train_it(
        rank=rank,
        world_size=world_size,
    )
    for step, rollout in enumerate(rollout_it):
        eval_df = evaluate_rollout(rollout).to_dataframe()
        eval_df = eval_df[
            eval_df["plan_correct"]
            & (eval_df["plan_length"] == eval_df["optimal_plan_length"])
        ]
        if len(eval_df) == 0:
            tok_trace = TokenizedTrace(
                id=rollout.id,
                prompt=rollout.prompt,
                reasoning=rollout.reasoning,
                plan=rollout.plan,
            )
        else:
            eval_df = eval_df[["index", "trace_tokens"]].copy()
            eval_df.sort_values(
                by="trace_tokens", ascending=True, inplace=True
            )  # type: ignore
            best_rollout_tok = rollout.rollouts[eval_df["index"].values[0]]
            plan_start = best_rollout_tok.index("plan")
            tok_trace = TokenizedTrace(
                id=rollout.id,
                prompt=rollout.prompt,
                reasoning=best_rollout_tok[1:plan_start],
                plan=best_rollout_tok[plan_start:-1],
            )
        tok_dataset.add_train_trace(tok_trace)
        if (step + 1) % 10 == 0:
            logging.info(f"Transferred {step + 1} prompts.")
    logging.info(f"Created dataset with name {tok_dataset.name}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(levelname)s - %(asctime)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()
