# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from typing import Any, Dict, List

import pytest


def test_astar_walls():
    import json

    import pandas as pd

    from searchformer.astar import astar
    from searchformer.maze import AStarMazeState, GridPos, MazeSpec

    spec = MazeSpec(
        width=3,
        height=3,
        start=GridPos(0, 0),
        goal=GridPos(0, 2),
        walls=[GridPos(0, 1), GridPos(1, 1)],
    )
    trace = astar(AStarMazeState(spec))
    trace_dict = [s.to_dict() for s in trace]
    trace_df = pd.json_normalize(trace_dict)
    with open("test/trace_maze_walls.json", "r") as f:
        trace_corr = pd.json_normalize(json.load(f))

    columns = set(trace_df.columns.to_list() + trace_corr.columns.to_list())
    for column in columns:
        assert (trace_df[column] == trace_corr[column]).all()


def test_astar_rail():
    from searchformer.astar import AStarCannotSolveTaskException, astar
    from searchformer.maze import AStarMazeState, GridPos, MazeSpec

    spec = MazeSpec(
        width=3,
        height=3,
        start=GridPos(0, 0),
        goal=GridPos(0, 2),
        walls=[GridPos(0, 1), GridPos(1, 1), GridPos(2, 1)],
    )
    try:
        astar(AStarMazeState(spec))
        pytest.fail()
    except AStarCannotSolveTaskException:
        pass


def test_astar_walls_short():
    import json

    import pandas as pd

    from searchformer.astar import astar
    from searchformer.maze import AStarMazeState, GridPos, MazeSpec

    spec = MazeSpec(
        width=3,
        height=3,
        start=GridPos(0, 0),
        goal=GridPos(0, 2),
        walls=[GridPos(0, 1)],
    )
    trace = astar(AStarMazeState(spec))
    trace_dict = [s.to_dict() for s in trace]
    trace_df = pd.json_normalize(trace_dict)
    with open("test/trace_maze_short.json", "r") as f:
        trace_corr = pd.json_normalize(json.load(f))

    columns = set(trace_df.columns.to_list() + trace_corr.columns.to_list())
    for column in columns:
        assert (trace_df[column] == trace_corr[column]).all()


def test_astar():
    global CORR_TRACE_JSON
    import json

    import pandas as pd

    from searchformer.astar import astar
    from searchformer.maze import AStarMazeState, GridPos, MazeSpec

    spec = MazeSpec(
        width=3,
        height=3,
        start=GridPos(0, 0),
        goal=GridPos(2, 2),
        walls=[],
    )
    trace = astar(AStarMazeState(spec))
    trace_dict = [s.to_dict() for s in trace]
    trace_df = pd.json_normalize(trace_dict)
    with open("test/trace_maze.json", "r") as f:
        trace_corr = pd.json_normalize(json.load(f))

    columns = set(trace_df.columns.to_list() + trace_corr.columns.to_list())
    for column in columns:
        assert (trace_df[column] == trace_corr[column]).all()


def test_trace_to_json():
    import json

    from searchformer.astar import TraceStep, astar
    from searchformer.maze import AStarMazeState, GridPos, MazeSpec

    spec = MazeSpec(
        width=3,
        height=3,
        start=GridPos(0, 0),
        goal=GridPos(0, 2),
        walls=[GridPos(0, 1), GridPos(1, 1)],
    )
    trace = astar(AStarMazeState(spec))
    trace_json = json.dumps([s.to_dict() for s in trace])
    trace_dict = json.loads(trace_json)
    trace_recon = [TraceStep.from_dict(d) for d in trace_dict]

    for t1, t2 in zip(trace, trace_recon):
        assert t1.action == t2.action
        assert t1.state["x"] == t2.state["x"]
        assert t1.state["y"] == t2.state["y"]
        assert t1.cost_from_start == t2.cost_from_start
        assert t1.heuristic == t2.heuristic


def test_grid_pos():
    from itertools import product

    from searchformer.maze import GridPos

    for x, y in product(range(4), range(5)):
        pos = GridPos(x, y)
        idx = pos.to_idx(4)
        pos_recon = GridPos.from_idx(idx, 4)
        assert pos == pos_recon


def test_grid_spec_descr():
    from searchformer.maze import GridPos, MazeSpec

    spec = MazeSpec(
        width=3,
        height=3,
        start=GridPos(0, 0),
        goal=GridPos(0, 2),
        walls=[GridPos(0, 1), GridPos(1, 1)],
    )
    assert spec.id == "3-3-0-6-3-4"


def test_grid_spec_int_seq_constructor():
    from searchformer.maze import MazeSpec

    int_seq = [3, 3, 0, 6, 3, 4]
    spec = MazeSpec.from_int_seq(int_seq)
    assert spec.id == "-".join(map(str, int_seq))


def test_astar_sokoban():
    import json

    import pandas as pd

    from searchformer.astar import astar
    from searchformer.sokoban import (
        AStarSokobanState,
        Sokoban,
        sokoban_state_to_string,
    )

    sokoban = Sokoban(
        [
            ["#", "#", "#", "#", "#"],
            ["#", "@", "$", ".", "#"],
            ["#", "#", "#", "#", "#"],
        ]
    )
    trace = astar(AStarSokobanState(sokoban))
    trace_dict_list: List[Dict[str, Any]] = []
    for step in trace:
        step_dict = step.to_dict()
        state_list = step_dict["state"]["state"]
        step_dict["state"] = sokoban_state_to_string(state_list)
        trace_dict_list.append(step_dict)

    trace_df = pd.json_normalize(trace_dict_list)
    with open("test/trace_sokoban.json", "r") as f:
        trace_corr = pd.json_normalize(json.load(f))
    columns = set(trace_df.columns.to_list() + trace_corr.columns.to_list())
    for column in columns:
        assert (trace_df[column] == trace_corr[column]).all()
