# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import random
from abc import abstractmethod
from dataclasses import dataclass
from heapq import heapify, heappop, heappush
from typing import Any, Dict, List, Optional, Sequence


class AStarState:
    """This class implements a state or node in A* search. A specific
    implementation of A* search for a task should implement this class as a
    sub-class.
    """

    def __init__(
        self,
        parent: Optional["AStarState"] = None,
        deterministic: bool = True,
    ):
        """Instatiates A* state.

        Args:
            parent (Optional[&quot;AStarState&quot;], optional): Parent node.
                At start this is set to None. Defaults to None.
            deterministic (bool, optional): If false, the returned child
                nodes are shuffled. This flag is often also used in the
                __le__ and __lt__ methods to randomize A*'s search
                dynamics. Defaults to True.
        """
        self.parent = parent
        self.deterministic = deterministic

    @property
    @abstractmethod
    def state(self) -> Dict[str, Any]:
        raise NotImplementedError()

    @abstractmethod
    def _get_children(self) -> List["AStarState"]:
        raise NotImplementedError()

    @property
    def children(self) -> List["AStarState"]:
        child_node_list = self._get_children()
        if not self.deterministic:
            random.shuffle(child_node_list)
        return child_node_list

    @property
    @abstractmethod
    def heuristic(self) -> float:
        raise NotImplementedError()

    @property
    @abstractmethod
    def cost_from_start(self) -> float:
        raise NotImplementedError()

    @abstractmethod
    def __hash__(self) -> int:
        """The hash is used to establish identity between different states. This
        function is implemented specifically to control the search behaviour of A* and
        integrate an implicit ordering of child nodes.

        Raises:
            NotImplementedError: If not implemented.

        Returns:
            int: Hash value of state.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def is_goal(self) -> bool:
        raise NotImplementedError()

    @property
    def cost(self) -> float:
        return self.heuristic + self.cost_from_start

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AStarState):
            return False
        return hash(self) == hash(other)

    def __lt__(self, other: "AStarState") -> bool:
        if self.cost == other.cost and self.deterministic:
            return hash(self) < hash(other)
        elif self.cost == other.cost and not self.deterministic:
            return random.choice([False, True])
        else:
            return self.cost < other.cost

    def __le__(self, other: "AStarState") -> bool:
        if self.cost == other.cost and self.deterministic:
            return hash(self) < hash(other)
        elif self.cost == other.cost and not self.deterministic:
            return random.choice([False, True])
        else:
            return self.cost < other.cost


@dataclass
class TraceStep:
    """Data class to store an A* execution trace step."""

    action: str
    state: Dict[str, Any]
    cost_from_start: int
    heuristic: int

    def to_dict(self) -> Dict[str, Any]:
        return dict(
            action=self.action,
            state=self.state,
            cost_from_start=self.cost_from_start,
            heuristic=self.heuristic,
        )

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "TraceStep":
        return TraceStep(**d)


class CreateStep(TraceStep):
    def __init__(self, **kvargs):
        super().__init__(action="create", **kvargs)


class CloseStep(TraceStep):
    def __init__(self, **kvargs):
        super().__init__(action="close", **kvargs)


class PlanStep(TraceStep):
    def __init__(self, **kvargs):
        super().__init__(action="plan", **kvargs)


class AStarCannotSolveTaskException(Exception):
    def __init__(self, trace: Optional[List[TraceStep]] = None):
        super().__init__()
        self.trace = trace


def astar(start_state: AStarState) -> Sequence[TraceStep]:
    """A* implementation used for generating execution trace datasets.

    The start state (or node) is provided as input and expanded until an
    optimal plan is found.

    Args:
        start_state (AStarState): Start state.

    Raises:
        AStarCannotSolveTaskException: If no feasible plan is found
            and the goal is not reached.

    Returns:
        Sequence[TraceStep]: Sequence of execution trace steps.
    """
    open_heap: List[AStarState] = []
    open_dict: Dict[AStarState, AStarState] = {}
    closed_dict: Dict[AStarState, AStarState] = {}
    log: List[TraceStep] = []

    curr_node = start_state
    heappush(open_heap, curr_node)
    open_dict[curr_node] = curr_node
    log.append(
        CreateStep(
            state=curr_node.state,
            cost_from_start=curr_node.cost_from_start,
            heuristic=curr_node.heuristic,
        )
    )

    while len(open_heap) > 0:
        curr_node = heappop(open_heap)
        del open_dict[curr_node]
        closed_dict[curr_node] = curr_node
        log.append(
            CloseStep(
                state=curr_node.state,
                cost_from_start=curr_node.cost_from_start,
                heuristic=curr_node.heuristic,
            )
        )
        if curr_node.cost == float("inf"):
            raise AStarCannotSolveTaskException(log)
        if curr_node.is_goal:
            break

        for child_node in curr_node.children:
            if child_node in open_dict.keys():
                if open_dict[child_node].cost <= child_node.cost:
                    continue
                else:
                    # This deletion is necessary because the hash is a function
                    # of the state, not the cost of the node. If there a lower
                    # cost value is computed for the same state, the following
                    # will prevent adding multiple nodes with the same state
                    # but different costs to the heap.
                    open_heap.remove(child_node)
                    heapify(open_heap)
                    del open_dict[child_node]
            if child_node in closed_dict.keys():
                if closed_dict[child_node].cost <= child_node.cost:
                    continue

            heappush(open_heap, child_node)
            open_dict[child_node] = child_node
            log.append(
                CreateStep(
                    state=child_node.state,
                    cost_from_start=child_node.cost_from_start,
                    heuristic=child_node.heuristic,
                )
            )
    if not curr_node.is_goal:
        raise AStarCannotSolveTaskException(log)

    path: List[AStarState] = [curr_node]
    node = curr_node.parent
    while node is not None:
        path.insert(0, node)
        node = node.parent
    for node in path:
        log.append(
            PlanStep(
                state=node.state,
                cost_from_start=node.cost_from_start,
                heuristic=node.heuristic,
            )
        )

    return log
