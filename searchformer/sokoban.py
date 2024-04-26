# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import functools
import json
import logging
import math
import os
import random
import sys
from copy import deepcopy
from dataclasses import dataclass
from itertools import chain, product
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
import numpy as np
import pygame
import pymongo
from pymongo.collection import Collection

from .astar import AStarCannotSolveTaskException, AStarState, TraceStep, astar
from .maze import GridPos, TokGridPos
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


class CellState:
    """Sokoban cell state values."""

    floor = " "
    wall = "#"
    dock = "."
    worker_on_floor = "@"
    worker_on_dock = "+"
    box = "$"
    box_on_dock = "*"

    @staticmethod
    def is_valid(state: str) -> bool:
        if CellState.floor == state:
            return True
        elif CellState.wall == state:
            return True
        elif CellState.dock == state:
            return True
        elif CellState.worker_on_floor == state:
            return True
        elif CellState.worker_on_dock == state:
            return True
        elif CellState.box == state:
            return True
        elif CellState.box_on_dock == state:
            return True
        else:
            return False

    @staticmethod
    def can_enter_by_box(state: str) -> bool:
        return (
            (state == CellState.floor)
            or (state == CellState.dock)
            or (state == CellState.worker_on_dock)
            or (state == CellState.worker_on_floor)
            or (state == CellState.box)
            or (state == CellState.box_on_dock)
        )


class SokobanRenderer:
    def __init__(self, width: int, height: int, record_dir: Optional[str]):
        """Constructs the PyGame renderer of a Sokoban level.

        Args:
            width (int): Width of the level grid.
            height (int): Height of the level grid.
            record_dir (Optional[str]): Record directory saving each level
                image as a separate file.
        """
        self.screen = pygame.display.set_mode((width * 32, height * 32))
        self._record_dir = record_dir
        self._record_step = 0

    @functools.cached_property
    def floor_image(self) -> pygame.Surface:
        fn = "sokoban/images/floor.png"
        assert os.path.exists(
            fn
        ), f"File {fn} not found. Maybe submodules were not initialized."
        return pygame.image.load(fn)

    @functools.cached_property
    def wall_image(self) -> pygame.Surface:
        fn = "sokoban/images/wall.png"
        assert os.path.exists(
            fn
        ), f"File {fn} not found. Maybe submodules were not initialized."
        return pygame.image.load(fn)

    @functools.cached_property
    def dock_image(self) -> pygame.Surface:
        fn = "sokoban/images/dock.png"
        assert os.path.exists(
            fn
        ), f"File {fn} not found. Maybe submodules were not initialized."
        return pygame.image.load(fn)

    @functools.cached_property
    def box_image(self) -> pygame.Surface:
        fn = "sokoban/images/box.png"
        assert os.path.exists(
            fn
        ), f"File {fn} not found. Maybe submodules were not initialized."
        return pygame.image.load(fn)

    @functools.cached_property
    def box_on_dock_image(self) -> pygame.Surface:
        fn = "sokoban/images/box_docked.png"
        assert os.path.exists(
            fn
        ), f"File {fn} not found. Maybe submodules were not initialized."
        return pygame.image.load(fn)

    @functools.cached_property
    def worker_on_floor_image(self) -> pygame.Surface:
        fn = "sokoban/images/worker.png"
        assert os.path.exists(
            fn
        ), f"File {fn} not found. Maybe submodules were not initialized."
        return pygame.image.load(fn)

    @functools.cached_property
    def worker_on_dock_image(self) -> pygame.Surface:
        fn = "sokoban/images/worker_dock.png"
        assert os.path.exists(
            fn
        ), f"File {fn} not found. Maybe submodules were not initialized."
        return pygame.image.load(fn)

    def render(self, game_state: List[List[str]]):
        self.screen.fill((255, 226, 191))
        for y, row in enumerate(game_state):
            for x, cell in enumerate(row):
                x_pos = x * 32
                y_pos = y * 32
                if cell == CellState.floor:
                    img = self.floor_image
                elif cell == CellState.wall:
                    img = self.wall_image
                elif cell == CellState.dock:
                    img = self.dock_image
                elif cell == CellState.box:
                    img = self.box_image
                elif cell == CellState.box_on_dock:
                    img = self.box_on_dock_image
                elif cell == CellState.worker_on_dock:
                    img = self.worker_on_dock_image
                elif cell == CellState.worker_on_floor:
                    img = self.worker_on_floor_image
                self.screen.blit(img, (x_pos, y_pos))

    def img_to_file(self):
        assert (
            self._record_dir is not None
        ), "Cannot save to file because record_dir was not set."
        img_fn = os.path.join(
            self._record_dir,
            f"img-{self._record_step:04d}.png",
        )
        pygame.image.save(self.screen, img_fn)
        self._record_step += 1


def manhattan_distance(x: int, y: int, x_new: int, y_new: int) -> int:
    return abs(x_new - x) + abs(y_new - y)


def sokoban_state_to_string(state_list: List[List[str]]) -> str:
    rows = map(lambda r: "".join(r), state_list)
    return "-".join(rows)


def sokoban_state_to_pretty_string(state: List[List[str]]) -> str:
    return "\n".join(["".join(row) for row in state])


class Sokoban:
    """Sokoban task implementation."""

    action_up = "up"
    action_down = "down"
    action_left = "left"
    action_right = "right"

    def __init__(self, state: List[List[str]]):
        """Constructs a level of the Sokoban task using the provided 2D list.
        Each element of this list corresponds to a cell state character.

        Args:
            state (List[List[str]]): 2D list specifing level.
        """
        self._state = state
        self._worker_x, self._worker_y = self.find_worker()

    @staticmethod
    def from_positions(
        worker: GridPos,
        boxes: List[GridPos],
        docks: List[GridPos],
        walls: List[GridPos],
        width: int,
        height: int,
    ) -> "Sokoban":
        state = [[CellState.floor] * width] * height
        state[worker.y][worker.x] = CellState.worker_on_floor
        for p in boxes:
            state[p.y][p.x] = CellState.box
        for p in docks:
            state[p.y][p.x] = CellState.dock
        for p in walls:
            state[p.y][p.x] = CellState.wall
        return Sokoban(state)

    @staticmethod
    def from_xy_prompt(
        prompt_tok: List[str],
        width: int,
        height: int,
    ) -> "Sokoban":
        prompt_positions = TokGridPos.from_tok_sequence(prompt_tok)
        pos_to_cell = {}
        for tok_pos in prompt_positions:
            if tok_pos.tok == "worker":
                pos_to_cell[tok_pos.pos] = CellState.worker_on_floor
            elif tok_pos.tok == "box":
                pos_to_cell[tok_pos.pos] = CellState.box
            elif tok_pos.tok == "dock":
                pos_to_cell[tok_pos.pos] = CellState.dock
            elif tok_pos.tok == "wall":
                pos_to_cell[tok_pos.pos] = CellState.wall

        state = []
        for y in range(height):
            row = []
            for x in range(width):
                row.append(pos_to_cell.get(GridPos(x, y), CellState.floor))
            state.append(row)

        return Sokoban(state)

    def __hash__(self) -> int:
        rows = map(lambda r: "".join(r), self._state)
        return hash("-".join(rows))

    def __repr__(self) -> str:
        return f"Sokoban({sokoban_state_to_string(self._state)})"

    def find_walls(self) -> Optional[np.ndarray]:
        xy_list: List[Tuple[int, int]] = []
        for y, row in enumerate(self._state):
            for x, cell in enumerate(row):
                if cell in CellState.wall:
                    xy_list.append((x, y))
        if len(xy_list) == 0:
            return None
        return np.array(xy_list)  # type: ignore

    def find_worker(self) -> Tuple[int, int]:
        for y, row in enumerate(self._state):
            for x, cell in enumerate(row):
                if (
                    cell == CellState.worker_on_dock
                    or cell == CellState.worker_on_floor
                ):
                    return (x, y)
        raise ValueError("Cannot find worker.")

    def find_open_docks(self) -> Optional[np.ndarray]:
        dock_states = {CellState.dock, CellState.worker_on_dock}
        xy_list: List[Tuple[int, int]] = []
        for y, row in enumerate(self._state):
            for x, cell in enumerate(row):
                if cell in dock_states:
                    xy_list.append((x, y))
        if len(xy_list) == 0:
            return None
        return np.array(xy_list)  # type: ignore

    def find_to_move_boxes(self) -> Optional[np.ndarray]:
        xy_list: List[Tuple[int, int]] = []
        for y, row in enumerate(self._state):
            for x, cell in enumerate(row):
                if cell == CellState.box:
                    xy_list.append((x, y))
        if len(xy_list) == 0:
            return None
        return np.array(xy_list)  # type: ignore

    def is_box_movable(self, x: int, y: int) -> bool:
        if x > 0 and x < self.width - 1:
            left_free = CellState.can_enter_by_box(self.get_state(x - 1, y))
            right_free = CellState.can_enter_by_box(self.get_state(x + 1, y))
            horizontal_move = left_free and right_free
        else:
            horizontal_move = False
        if y > 0 and y < self.height - 1:
            top_free = CellState.can_enter_by_box(self.get_state(x, y - 1))
            bottom_free = CellState.can_enter_by_box(self.get_state(x, y + 1))
            vertical_move = top_free and bottom_free
        else:
            vertical_move = False
        return horizontal_move or vertical_move

    def heuristic(self) -> float:
        boxes_xy = self.find_to_move_boxes()
        docks_xy = self.find_open_docks()

        if boxes_xy is None or docks_xy is None:
            return 0
        for xy in boxes_xy:
            if not self.is_box_movable(xy[0], xy[1]):
                return float("inf")

        boxes_xy = boxes_xy.reshape(-1, 1, 2)
        docks_xy = docks_xy.reshape(1, -1, 2)

        pairwise_dist = np.abs(docks_xy - boxes_xy).sum(-1)
        closest_box = pairwise_dist.min(-1)  # type: ignore
        return float(closest_box.sum())

    def get_state(self, x: int, y: int) -> str:
        return self._state[y][x]

    def set_state(self, x: int, y: int, state: str):
        self._state[y][x] = state

    def _in_bounds(self, x: int, y: int) -> bool:
        return x >= 0 and x < self.width and y >= 0 and y < self.height

    def _try_move_worker(
        self, x: int, y: int, x_new: int, y_new: int
    ) -> Optional[Tuple[int, int]]:
        if not self._in_bounds(x, y):
            return None
        if not self._in_bounds(x_new, y_new):
            return None
        if manhattan_distance(x, y, x_new, y_new) != 1:
            return None
        curr_c = self.get_state(x, y)
        new_c = self.get_state(x_new, y_new)
        if new_c == CellState.floor and curr_c == CellState.worker_on_floor:
            # logging.debug("Worker floor -> floor")
            self.set_state(x_new, y_new, CellState.worker_on_floor)
            self.set_state(x, y, CellState.floor)
        elif new_c == CellState.dock and curr_c == CellState.worker_on_floor:
            # logging.debug("Worker floor -> dock")
            self.set_state(x_new, y_new, CellState.worker_on_dock)
            self.set_state(x, y, CellState.floor)
        elif new_c == CellState.floor and curr_c == CellState.worker_on_dock:
            # logging.debug("Worker dock -> floor")
            self.set_state(x_new, y_new, CellState.worker_on_floor)
            self.set_state(x, y, CellState.dock)
        elif new_c == CellState.dock and curr_c == CellState.worker_on_dock:
            # logging.debug("Worker dock -> dock")
            self.set_state(x_new, y_new, CellState.worker_on_dock)
            self.set_state(x, y, CellState.dock)
        else:
            return None
        return x_new, y_new

    def _try_move_box(
        self, x: int, y: int, x_new: int, y_new: int
    ) -> Optional[Tuple[int, int]]:
        if not self._in_bounds(x, y):
            return None
        if not self._in_bounds(x_new, y_new):
            return None
        if manhattan_distance(x, y, x_new, y_new) != 1:
            return None
        curr_c = self.get_state(x, y)
        new_c = self.get_state(x_new, y_new)
        if new_c == CellState.floor and curr_c == CellState.box:
            self.set_state(x_new, y_new, CellState.box)
            self.set_state(x, y, CellState.floor)
        elif new_c == CellState.dock and curr_c == CellState.box:
            self.set_state(x_new, y_new, CellState.box_on_dock)
            self.set_state(x, y, CellState.floor)
        elif new_c == CellState.floor and curr_c == CellState.box_on_dock:
            self.set_state(x_new, y_new, CellState.box)
            self.set_state(x, y, CellState.dock)
        elif new_c == CellState.dock and curr_c == CellState.box_on_dock:
            self.set_state(x_new, y_new, CellState.box_on_dock)
            self.set_state(x, y, CellState.dock)
        else:
            return None
        return x_new, y_new

    @property
    def state(self) -> List[List[str]]:
        return deepcopy(self._state)

    @functools.cached_property
    def height(self) -> int:
        return len(self._state)

    @functools.cached_property
    def width(self) -> int:
        return max([len(row) for row in self._state])

    @staticmethod
    def from_level_file(filename: str) -> "Sokoban":
        with open(filename, "r") as f:
            line_list = f.readlines()
        state = [list(line.replace("\n", "")) for line in line_list]
        return Sokoban(state)

    @property
    def is_complete(self) -> bool:
        return not any(
            map(lambda b: b == CellState.box, chain.from_iterable(self._state))
        )

    @staticmethod
    def action_from_position_change(p1: GridPos, p2: GridPos) -> str:
        if p1.x == p2.x and p1.y == p2.y - 1:
            return Sokoban.action_down
        elif p1.x == p2.x and p1.y == p2.y + 1:
            return Sokoban.action_up
        elif p1.y == p2.y and p1.x == p2.x - 1:
            return Sokoban.action_right
        elif p1.y == p2.y and p1.x == p2.x + 1:
            return Sokoban.action_left
        else:
            raise ValueError("Cannot determine action.")

    def move_up(self):
        self._try_move_box(
            x=self._worker_x,
            y=self._worker_y - 1,
            x_new=self._worker_x,
            y_new=self._worker_y - 2,
        )
        res_worker = self._try_move_worker(
            x=self._worker_x,
            y=self._worker_y,
            x_new=self._worker_x,
            y_new=self._worker_y - 1,
        )
        if res_worker is not None:
            self._worker_x, self._worker_y = res_worker

    def move_down(self):
        self._try_move_box(
            x=self._worker_x,
            y=self._worker_y + 1,
            x_new=self._worker_x,
            y_new=self._worker_y + 2,
        )
        res_worker = self._try_move_worker(
            x=self._worker_x,
            y=self._worker_y,
            x_new=self._worker_x,
            y_new=self._worker_y + 1,
        )
        if res_worker is not None:
            self._worker_x, self._worker_y = res_worker

    def move_left(self):
        self._try_move_box(
            x=self._worker_x - 1,
            y=self._worker_y,
            x_new=self._worker_x - 2,
            y_new=self._worker_y,
        )
        res_worker = self._try_move_worker(
            x=self._worker_x,
            y=self._worker_y,
            x_new=self._worker_x - 1,
            y_new=self._worker_y,
        )
        if res_worker is not None:
            self._worker_x, self._worker_y = res_worker

    def move_right(self):
        self._try_move_box(
            x=self._worker_x + 1,
            y=self._worker_y,
            x_new=self._worker_x + 2,
            y_new=self._worker_y,
        )
        res_worker = self._try_move_worker(
            x=self._worker_x,
            y=self._worker_y,
            x_new=self._worker_x + 1,
            y_new=self._worker_y,
        )
        if res_worker is not None:
            self._worker_x, self._worker_y = res_worker

    def move(self, action: str):
        if action == self.action_up:
            self.move_up()
        elif action == self.action_down:
            self.move_down()
        elif action == self.action_right:
            self.move_right()
        elif action == self.action_left:
            self.move_left()
        else:
            raise ValueError(f"Invalid action {action}.")


@click.group()
def main():
    pass


def _play(sokoban: Sokoban, record_dir: Optional[str] = None):
    """Play Sokoban game interactively.

    Args:
        sokoban (Sokoban): Sokoban task.
        record_dir (Optional[str], optional): Log directory for Sokoban game
            screen shots. Defaults to None.
    """
    pygame.init()
    renderer = SokobanRenderer(
        width=sokoban.width, height=sokoban.height, record_dir=record_dir
    )
    if record_dir is not None:
        os.makedirs(record_dir, exist_ok=True)
    screen_changed = True
    while 1:
        renderer.render(sokoban.state)
        pygame.display.update()
        if record_dir is not None and screen_changed:
            renderer.img_to_file()
            screen_changed = False
        if sokoban.is_complete:
            logging.info(f"Puzzle finished.")
            sys.exit(0)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit(0)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    logging.debug("Exiting.")
                    sys.exit(0)
                elif event.key == pygame.K_UP:
                    logging.debug("Move up.")
                    sokoban.move_up()
                    screen_changed = True
                elif event.key == pygame.K_DOWN:
                    logging.debug("Move down.")
                    sokoban.move_down()
                    screen_changed = True
                elif event.key == pygame.K_LEFT:
                    logging.debug("Move left.")
                    sokoban.move_left()
                    screen_changed = True
                elif event.key == pygame.K_RIGHT:
                    logging.debug("Move right.")
                    sokoban.move_right()
                    screen_changed = True
            # if record_dir is not None:
            #     renderer.img_to_file(record_step_count)
            #     record_step_count += 1


@main.command()
@click.option("--level", type=str, default="static/sokoban/example-7722.txt")
@click.option(
    "--record-dir",
    type=str,
    default=None,
    help="Level image record dir",
)
def play(level: str, record_dir: Optional[str] = None):
    """Play Sokoban game interactively."""
    sokoban = Sokoban.from_level_file(level)
    _play(sokoban, record_dir=record_dir)


class AStarSokobanState(AStarState):
    def __init__(
        self,
        sokoban: Sokoban,
        cost_from_start: float = 0.0,
        parent: Optional["AStarSokobanState"] = None,
        deterministic: bool = True,
    ):
        """Constructs a A* search state for the Sokoban task.

        Args:
            sokoban (Sokoban): Sokoban task instance.
            cost_from_start (float, optional): Cost from start.
                Defaults to 0.0.
            parent (Optional[quot;AStarSokobanState&quot;&], optional):
                Parent node. Defaults to None.
            deterministic (bool, optional): If True, use deterministic A*
                search, otherwise use nondeterministic A* search.
                Defaults to True.
        """
        super().__init__(parent, deterministic=deterministic)
        self.sokoban = sokoban
        self._cost_from_start = cost_from_start

    @property
    def state(self) -> Dict[str, Any]:
        return dict(state=self.sokoban.state)

    def _construct_child(self, direction: str) -> "AStarSokobanState":
        sokoban_child = deepcopy(self.sokoban)
        if direction == "up":
            sokoban_child.move_up()
        elif direction == "down":
            sokoban_child.move_down()
        elif direction == "left":
            sokoban_child.move_left()
        elif direction == "right":
            sokoban_child.move_right()
        return AStarSokobanState(
            sokoban=sokoban_child,
            cost_from_start=self.cost_from_start + 1,
            parent=self,
            deterministic=self.deterministic,
        )

    def _get_children(self) -> List["AStarState"]:
        directions = ["up", "down", "left", "right"]
        child_nodes: List["AStarState"] = []
        for direction in directions:
            node = self._construct_child(direction)
            if not math.isfinite(node.cost):
                continue
            child_nodes.append(node)
        return child_nodes

    @property
    def heuristic(self) -> float:
        return self.sokoban.heuristic()

    @property
    def cost_from_start(self) -> float:
        return self._cost_from_start

    def __hash__(self) -> int:
        return hash(self.sokoban)

    @property
    def is_goal(self) -> bool:
        return self.sokoban.is_complete


@main.command()
@click.option("--level", type=str)
@click.option("--trace", type=str)
def generate_trace_file(level: str, trace: str):
    """Generate single Sokoban A* execution trace.

    Args:
        level (str): Sokoban level file.
        trace (str): JSON trace file name.
    """
    sokoban = Sokoban.from_level_file(level)
    trace_seq = astar(AStarSokobanState(sokoban, deterministic=False))
    logging.info(f"Trace length: {len(trace_seq)}")
    with open(trace, "w") as f:
        json.dump([s.to_dict() for s in trace_seq], f)
    logging.info("Finished generation.")


def boundary_positions(width: int, height: int) -> Sequence[Tuple[int, int]]:
    boundary = [(x, 0) for x in range(width)]
    boundary += [(x, height - 1) for x in range(width)]
    boundary += [(0, y) for y in range(height)]
    boundary += [(width - 1, y) for y in range(height)]
    return boundary


def sample_box_positions(
    width: int,
    height: int,
    n: int,
    exclude: Optional[Iterable[Tuple[int, int]]] = None,
) -> Sequence[Tuple[int, int]]:
    """Sample box positions that are 2 steps away from boundary.

    Args:
        width (int): Level width.
        height (int): Level height.
        n (int): Number of boxes.
        exclude (Optional[Iterable[Tuple[int, int]]], optional): Positions
            that should be excluded from sampling. Defaults to None.

    Returns:
        Sequence[Tuple[int, int]]: List of (x,y) position pairs.
    """
    if exclude is None:
        exclude_set = set()
    else:
        exclude_set = set(exclude)

    xys = set(product(range(2, width - 2), range(2, height - 2)))
    xys = xys.difference(exclude_set)
    xys_list = list(xys)
    random.shuffle(xys_list)
    return xys_list[:n]


def sample_positions(
    width: int,
    height: int,
    n: int,
    exclude: Optional[Iterable[Tuple[int, int]]] = None,
) -> Sequence[Tuple[int, int]]:
    """Sample grid positions positions.

    Args:
        width (int): Level width.
        height (int): Level height.
        n (int): Number of boxes.
        exclude (Optional[Iterable[Tuple[int, int]]], optional): Positions
            that should be excluded from sampling. Defaults to None.

    Returns:
        Sequence[Tuple[int, int]]: List of (x,y) position pairs.
    """
    if exclude is None:
        exclude_set = set()
    else:
        exclude_set = set(exclude)

    xys = set(product(range(width), range(height)))
    xys = xys.difference(exclude_set)
    xys_list = list(xys)
    random.shuffle(xys_list)
    return xys_list[:n]


def generate_level(
    width: int, height: int, num_walls: int, num_boxes: int = 1
) -> Sokoban:
    """Randomly generate Sokoban level.

    Args:
        width (int): Grid width.
        height (int): Grid height.
        num_walls (int): Number of interior wall cells.
        num_boxes (int, optional): Number of boxes. Defaults to 1.

    Returns:
        Sokoban: Sokoban task state.
    """
    pos_to_obj = {}
    for xy in boundary_positions(width, height):
        pos_to_obj[xy] = CellState.wall
    for xy in sample_positions(width, height, num_walls, pos_to_obj.keys()):
        pos_to_obj[xy] = CellState.wall
    for xy in sample_box_positions(
        width,
        height,
        num_boxes,
        pos_to_obj.keys(),
    ):
        pos_to_obj[xy] = CellState.box
    for xy in sample_positions(width, height, 1, pos_to_obj.keys()):
        pos_to_obj[xy] = CellState.worker_on_floor
    for xy in sample_positions(width, height, num_boxes, pos_to_obj.keys()):
        pos_to_obj[xy] = CellState.dock

    level: List[List[str]] = []
    for i, (y, x) in enumerate(product(range(height), range(width))):
        if x == 0:
            level.append([])
        level[-1].append(pos_to_obj.get((x, y), CellState.floor))

    # print(sokoban_state_to_pretty_sting(level))
    return Sokoban(level)


@dataclass
class SokobanTrace:
    """Dataclass holding Sokoban execution trace."""

    sokoban_start: Sokoban
    trace: List[TraceStep]

    def __hash__(self) -> int:
        return hash(self.sokoban_start)

    def to_dict(self) -> Dict[str, Any]:
        return dict(
            sokoban_start=self.sokoban_start.state,
            trace=[step.to_dict() for step in self.trace],
        )

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "SokobanTrace":
        return SokobanTrace(
            sokoban_start=Sokoban(d["sokoban_start"]),
            trace=[TraceStep.from_dict(step) for step in d["trace"]],
        )

    @staticmethod
    def generate(
        width: int, height: int, num_walls: int, num_boxes: int
    ) -> "SokobanTrace":
        """Trows an AStarCannotSolveTaskException if generated level cannot
        be solved.

        Args:
            width (int): Width of level (including boundary)
            height (int): Height of level (including boundary)
            num_walls (int): Number of wall cells in level (excluding boundary)
            num_boxes (int): Number of boxes in level

        Returns:
            SokobanTrace: Trace object.
        """
        sokoban_start = generate_level(width, height, num_walls, num_boxes)
        trace = astar(AStarSokobanState(sokoban_start, deterministic=False))
        return SokobanTrace(sokoban_start, list(trace))

    @property
    def reasoning_trace(self) -> Sequence[TraceStep]:
        return [s for s in self.trace if s.action != "plan"]

    @property
    def plan(self) -> Sequence[TraceStep]:
        return [s for s in self.trace if s.action == "plan"]


SOKOBAN_DB_NAME = "sokobanDB"


class SokobanTraceDataset:
    """Class to write and read Sokoban execution traces from MongoDB."""

    def __init__(self, name: str):
        """Constructs dataset object from dataset name,

        Args:
            name (str): Trace dataset name.
        """
        self.name = name
        self.client = mongodb_client()
        self.db = self.client[SOKOBAN_DB_NAME]

    @functools.cached_property
    def trace_collection(self) -> Collection:
        return self.db[f"{self.name}.trace"]

    @functools.cached_property
    def index_collection(self) -> Collection:
        return self.db[f"{self.name}.index"]

    def drop(self):
        self.db.drop_collection(self.trace_collection)
        self.db.drop_collection(self.index_collection)

    def add(self, trace: SokobanTrace, is_test: bool):
        self.trace_collection.insert_one(
            {"_id": hash(trace), "is_test": is_test, "trace": trace.to_dict()},
        )
        self.index_collection.insert_one(
            {"_id": hash(trace), "is_test": is_test},
        )

    def generate(
        self,
        is_test: bool,
        width: int,
        height: int,
        num_walls: int,
        num_boxes: int,
    ) -> int:
        """Randomly generates Sokoban task and adds trace into dataset.

        Args:
            is_test (bool): If True, insert into test set. Otherwise insert
                into training set.
            width (int): Level width.
            height (int): Level height.
            num_walls (int): Number of wall cells.
            num_boxes (int): Number of boxes.

        Returns:
            int: Number of generated tasks. 1 if task was successfully added,
                0 if not task was added (either because the sampled task is
                not solvable or because it already exists in the dataset).
        """
        try:
            trace = SokobanTrace.generate(
                width=width,
                height=height,
                num_walls=num_walls,
                num_boxes=num_boxes,
            )
        except AStarCannotSolveTaskException:
            return 0

        try:
            self.add(trace, is_test=is_test)
            return 1
        except pymongo.errors.DuplicateKeyError:
            return 0
        except pymongo.errors.DocumentTooLarge:
            logging.warning(
                f"Could not store trace with {len(trace.trace)} steps.",
            )
            return 0

    @property
    def index_list(self) -> List[int]:
        collection_it = self.index_collection.find({}, {"_id": 1})
        index_it = map(lambda d: d["_id"], collection_it)
        return list(index_it)

    def iterate_traces(
        self, rank: int = 0, world_size: int = 1, load_batch_size: int = 1000
    ) -> Iterator[Tuple[SokobanTrace, bool]]:
        index_list = self.index_list
        index_list.sort()
        logging.info(f"Total number of Sokoban traces: {len(index_list)}")

        slice_size = math.ceil(len(index_list) / world_size)
        index_list = index_list[rank * slice_size : (rank + 1) * slice_size]
        logging.debug(f"Iterating over {len(index_list)} Sokoban sequences.")

        collection = self.trace_collection
        for i in range(0, len(index_list), load_batch_size):
            index_batch = index_list[i : i + load_batch_size]
            for res in collection.find({"_id": {"$in": index_batch}}):
                is_test = bool(res["is_test"])
                trace = SokobanTrace.from_dict(res["trace"])
                yield trace, is_test


@main.command()
@click.option("--name", type=str, help="Sokoban trace dataset name.")
def drop_dataset(name: str):
    logging.info(f"Dropping Sokoban dataset {name}")
    sokoban_dataset = SokobanTraceDataset(name)
    sokoban_dataset.drop()
    logging.info("Done.")


def generation_args_to_name(
    width: int,
    height: int,
    num_walls: int,
    num_boxes: int,
) -> str:
    return f"sokoban.{width}-by-{height}-walls-{num_walls}-boxes-{num_boxes}"


@main.command()
@click.option("--width", type=int, default=7, help="Level width")
@click.option("--height", type=int, default=7, help="Level height")
@click.option("--num-walls", type=int, default=3, help="Number of wall cells")
@click.option("--num-boxes", type=int, default=1, help="Number of box cells")
@click.option(
    "--num-samples",
    type=int,
    default=1000,
    help="Number of tasks to sample",
)
def generate(
    width: int,
    height: int,
    num_walls: int,
    num_boxes: int,
    num_samples: int,
):
    """Generate Sokoban tasks and insert A* execution traces into MongoDB."""
    name = generation_args_to_name(
        width=width,
        height=height,
        num_walls=num_walls,
        num_boxes=num_boxes,
    )
    dataset = SokobanTraceDataset(name)
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
            num_walls=num_walls,
            num_boxes=num_boxes,
        )
        logging.info(f"Generated {samples_stored} samples.")
    logging.info("Done.")


class SimpleSokobanTokenizer(Tokenizer):
    """Tokenizer base class used for generating Sokoban token datasets."""

    tok_worker = "worker"
    tok_box = "box"
    tok_dock = "dock"
    tok_wall = "wall"
    tok_create = "create"
    tok_close = "close"
    tok_plan = "plan"

    def __init__(self, width: int, height: int):
        vocabulary = [str(i) for i in range(max(width, height))]
        vocabulary += [f"c{i}" for i in range(40000)]
        vocabulary += [
            SimpleSokobanTokenizer.tok_box,
            SimpleSokobanTokenizer.tok_worker,
            SimpleSokobanTokenizer.tok_dock,
            SimpleSokobanTokenizer.tok_wall,
            SimpleSokobanTokenizer.tok_create,
            SimpleSokobanTokenizer.tok_close,
            SimpleSokobanTokenizer.tok_plan,
        ]
        super().__init__(vocabulary)

    def _tokens_walls(self, sokoban: Sokoban) -> List[str]:
        xys = sokoban.find_walls()
        if xys is None:
            return []
        xys_str = [f"{x}-{y}" for x, y in xys.tolist()]  # type: ignore
        xys_str.sort()
        tokens: List[str] = []
        for x_str, y_str in map(lambda s: s.split("-"), xys_str):
            tokens += [SimpleSokobanTokenizer.tok_wall, x_str, y_str]
        return tokens

    def _tokens_boxes(self, sokoban: Sokoban) -> List[str]:
        xys = sokoban.find_to_move_boxes()
        if xys is None:
            return []
        xys_str = [f"{x}-{y}" for x, y in xys.tolist()]  # type: ignore
        xys_str.sort()
        tokens: List[str] = []
        for x_str, y_str in map(lambda s: s.split("-"), xys_str):
            tokens += [SimpleSokobanTokenizer.tok_box, x_str, y_str]
        return tokens

    def _tokens_docks(self, sokoban: Sokoban) -> List[str]:
        xys = sokoban.find_open_docks()
        if xys is None:
            return []
        xys_str = [f"{x}-{y}" for x, y in xys.tolist()]  # type: ignore
        xys_str.sort()
        tokens: List[str] = []
        for x_str, y_str in map(lambda s: s.split("-"), xys_str):
            tokens += [SimpleSokobanTokenizer.tok_dock, x_str, y_str]
        return tokens

    def _tokens_worker(self, sokoban: Sokoban) -> List[str]:
        x, y = sokoban.find_worker()
        return [SimpleSokobanTokenizer.tok_worker, str(x), str(y)]

    def _tokenize_prompt(self, sokoban: Sokoban) -> List[str]:
        return [
            *self._tokens_worker(sokoban),
            *self._tokens_boxes(sokoban),
            *self._tokens_docks(sokoban),
            *self._tokens_walls(sokoban),
        ]

    def _tokenize_reasoning_step(self, step: TraceStep) -> List[str]:
        assert step.action in [
            SimpleSokobanTokenizer.tok_create,
            SimpleSokobanTokenizer.tok_close,
        ]
        if step.cost_from_start >= 40000:
            raise CannotTokenizeException("Cannot tokenize trace costs.")
        if step.heuristic >= 40000:
            raise CannotTokenizeException("Cannot tokenize trace heuristics.")
        sokoban = Sokoban(step.state["state"])
        assert not math.isinf(step.cost_from_start)
        assert not math.isnan(step.cost_from_start)
        assert not math.isinf(step.heuristic)
        assert not math.isnan(step.heuristic)
        x, y = sokoban.find_worker()
        return [
            step.action,
            str(x),
            str(y),
            f"c{int(step.cost_from_start)}",
            f"c{int(step.heuristic)}",
        ]

    def _tokenize_plan_step(self, step: TraceStep) -> List[str]:
        assert step.action == SimpleSokobanTokenizer.tok_plan
        sokoban = Sokoban(step.state["state"])
        assert not math.isinf(step.cost_from_start)
        assert not math.isnan(step.cost_from_start)
        assert not math.isinf(step.heuristic)
        assert not math.isnan(step.heuristic)
        x, y = sokoban.find_worker()
        return [
            step.action,
            str(x),
            str(y),
        ]

    def tokenize(
        self,
        trace: SokobanTrace,
        is_test: bool = False,
    ) -> TokenizedTrace:
        prompt = self._tokenize_prompt(trace.sokoban_start)
        reasoning = chain.from_iterable(
            map(self._tokenize_reasoning_step, trace.reasoning_trace)
        )
        plan = chain.from_iterable(map(self._tokenize_plan_step, trace.plan))
        return TokenizedTrace(
            id=hash(trace),
            prompt=prompt,
            reasoning=list(reasoning),
            plan=list(plan),
        )


class WithBoxSokobanTokenizer(SimpleSokobanTokenizer):
    """Tokenizer class used for generating Sokoban token datasets."""

    def _tokenize_reasoning_step(self, step: TraceStep) -> List[str]:
        assert step.action in [
            SimpleSokobanTokenizer.tok_create,
            SimpleSokobanTokenizer.tok_close,
        ]
        sokoban = Sokoban(step.state["state"])
        assert not math.isinf(step.cost_from_start)
        assert not math.isnan(step.cost_from_start)
        assert not math.isinf(step.heuristic)
        assert not math.isnan(step.heuristic)
        return [
            step.action,
            *self._tokens_worker(sokoban),
            *self._tokens_boxes(sokoban),
            f"c{int(step.cost_from_start)}",
            f"c{int(step.heuristic)}",
        ]


@main.command()
@click.option("--width", type=int, default=7, help="Level width.")
@click.option("--height", type=int, default=7, help="Level height.")
@click.option("--num-walls", type=int, default=3, help="Number of wall cells.")
@click.option("--num-boxes", type=int, default=1, help="Number of box cells.")
@click.option("--rank", type=int, default=0, help="Worker id.")
@click.option(
    "--world-size",
    type=int,
    default=1,
    help="Total number of workers.",
)
def tokenize(
    width: int,
    height: int,
    num_walls: int,
    num_boxes: int,
    rank: int,
    world_size: int,
):
    """Tokenize trace dataset.

    Width, height, number of wall cells, and number of box cells are used to
    indentify the specific training dataset.
    """
    name = generation_args_to_name(
        width=width, height=height, num_walls=num_walls, num_boxes=num_boxes
    )
    sokoban_dataset = SokobanTraceDataset(name)
    tokenizer = WithBoxSokobanTokenizer(width, height)
    tok_dataset = TokenizedDataset(f"{sokoban_dataset.name}.with-box-40k")
    tok_dataset.add_vocabulary(tokenizer.vocabulary)

    total_traces = 0
    for trace, is_test in sokoban_dataset.iterate_traces(rank, world_size):
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


def evaluate_rollout_single(
    prompt_tok: List[str], rollout_tok: List[str]
) -> RolloutEvaluation:
    sokoban = Sokoban.from_xy_prompt(prompt_tok, 7, 7)
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
        p1_p2_it = zip(plan_pos[1:], plan_pos[:-1])
        correct_dist = all([p1.distance(p2) == 1 for p1, p2 in p1_p2_it])
        if correct_dist:
            for p1, p2 in zip(plan_pos[:-1], plan_pos[1:]):
                action = Sokoban.action_from_position_change(p1, p2)
                sokoban.move(action)
            correct_plan = sokoban.is_complete
        else:
            correct_plan = False
    except ValueError:
        plan_length = 0
        trace_tokens = 0
        syntax_correct = False
        correct_plan = False

    return RolloutEvaluation(
        has_eos=has_eos,
        plan_length=plan_length,
        plan_syntax_correct=syntax_correct,
        plan_correct_start=correct_plan,
        plan_correct_goal=correct_plan,
        plan_correct=correct_plan,
        trace_tokens=trace_tokens,
    )


def evaluate_rollout(rollout: Rollout) -> RolloutEvaluationBatch:
    opt_plan_len = len(rollout.plan) // 3
    eval_list: List[RolloutEvaluation] = []

    for rollout_tok in rollout.rollouts:
        eval_list.append(evaluate_rollout_single(rollout.prompt, rollout_tok))
    return RolloutEvaluationBatch(
        id=rollout.id,
        optimal_plan_length=opt_plan_len,
        reasoning_length=len(rollout.reasoning),
        rollout=eval_list,
    )


@main.command()
@click.option("--rollout-id", type=str)
@click.option("--origin-dataset", type=str)
@click.option("--rank", type=int, default=0)
@click.option("--world-size", type=int, default=1)
def reduce_rollout_to_shortest_trace(
    rollout_id: str,
    origin_dataset: str,
    rank: int,
    world_size: int,
):
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


@main.command()
@click.option(
    "--trace-data",
    type=str,
    help="Token dataset name whose test prompts are used.",
)
@click.option(
    "--sequence-id",
    type=int,
    help="Test prompt id for which dataset is generated.",
)
@click.option(
    "--num-samples",
    type=int,
    help="Number of execution traces that are sampled from test task.",
)
def generate_tokenized_astar_samples(
    trace_data: str,
    sequence_id: int,
    num_samples: int,
):
    """Generates A* reference dataset to evaluate Searchformer results."""
    logging.info(f"Generating for sequence_id={sequence_id}")
    trace_dataset = SokobanTraceDataset(trace_data)

    trace_doc = trace_dataset.trace_collection.find_one(
        {"_id": {"$eq": sequence_id}},
    )
    assert trace_doc is not None, f"Cannot find trace with id {sequence_id}."
    trace_sok = SokobanTrace.from_dict(trace_doc["trace"])
    tokenizer = WithBoxSokobanTokenizer(7, 7)

    client = mongodb_client()
    db = client["sokobanAStarRefDataDB"]

    for i in range(num_samples):
        logging.info(f"Generating sample {i + 1}/{num_samples}")

        sokoban_start = deepcopy(trace_sok.sokoban_start)
        astar_state = AStarSokobanState(sokoban_start, deterministic=False)
        astar_trace = astar(astar_state)
        sokoban_trace = SokobanTrace(sokoban_start, list(astar_trace))
        tok_trace = tokenizer(sokoban_trace)
        tok_trace_doc = tok_trace.to_dict()
        tok_trace_doc["sequence_id"] = tok_trace_doc["_id"]
        del tok_trace_doc["_id"]

        db.trace_variance.insert_one(tok_trace_doc)

    logging.info("Finished generating samples.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(levelname)s - %(asctime)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()
