# Training Transformers

Training of each Transformer model is distributed across multiple GPUs with [DDP](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html).
Each training process loads data from MongoDB, writes logs into MongoDB (loss values and other stats), and checkpoints the model after a set step interval.
Furthermore, each training process is identified with a ID that is stored together with the runs checkpoints, parametrization, and logs.
If the a process is launched again with an already existing ID, then the training run will first reconstruct from a checkpoint if one is found.

There are two methods for starting a training run: as a single run or as a hyper-parameter sweep.

## Starting a single run

A single (2 GPU) run can be started with

```
torchrun \
    --rdzv-backend=c10d \
    --rdzv-endpoint=localhost:0 \
    --nnodes=1 \
    --nproc-per-node=2 \
    --module searchformer.train single --run-id=$ID ...
```

All hyper-parameters can be viewed with `python -m searchformer.train single --help`.

## Starting a run as part of a sweep

Because setting hyper-parameters for each run individually is cumbersome for larger sweeps, the json files in `config/sweep` contain a list of all hyper-parameters that are to be tested in a single sweep.
A run from this sweep can then launched by running the command

```
torchrun \
    --rdzv-backend=c10d \
    --rdzv-endpoint=localhost:0 \
    --nnodes=1 \
    --nproc-per-node=8 \
    --module searchformer.train \
    sweep \
    --run-id=$ID \
    --index=$RUN_ID \
    --sweep=$CONFIG_JSON
```

where `$CONFIG_JSON` is the path to the config file, `$RUN_ID` is the index of the run (the index of the hyper-parameter combination contained in the json file), and `$ID` assigns a unique id to each training run.
Note that experiments can be repeated simply by re-using the same `$CONFIG_JSON` and `$RUN_ID` but assigning a different `$ID` for each run.
The following sweep configs are included in this repository.

| Experiment                               | Config file                                                                                                             |
| ---                                      | ---                                                                                                                     |
| Maze, deterministic A*, small models     | [`config/sweep/maze_sweep_rep_det_small.json`             ](../config/sweep/maze_sweep_rep_det_small.json             ) |
| Maze, deterministic A*, medium models    | [`config/sweep/maze_sweep_rep_det_medium.json`            ](../config/sweep/maze_sweep_rep_det_medium.json            ) |
| Maze, deterministic A*, large models     | [`config/sweep/maze_sweep_rep_det_large.json`             ](../config/sweep/maze_sweep_rep_det_large.json             ) |
| Maze, nondeterministic A*, small models  | [`config/sweep/maze_sweep_rep_nondet_small.json`          ](../config/sweep/maze_sweep_rep_nondet_small.json          ) |
| Maze, nondeterministic A*, medium models | [`config/sweep/maze_sweep_rep_nondet_medium.json`         ](../config/sweep/maze_sweep_rep_nondet_medium.json         ) |
| Maze, nondeterministic A*, large models  | [`config/sweep/maze_sweep_rep_nondet_large.json`          ](../config/sweep/maze_sweep_rep_nondet_large.json          ) |
| Sokoban, extra-large solution-only model | [`config/sweep/sokoban-7722-xl-plan-only-100k.json`       ](../config/sweep/sokoban-7722-xl-plan-only-100k.json       ) |
| Sokoban, large solution-only model       | [`config/sweep/sokoban-7722-l-plan-only-100k.json`        ](../config/sweep/sokoban-7722-l-plan-only-100k.json        ) |
| Sokoban, medium solution-only model      | [`config/sweep/sokoban-7722-m-plan-only-100k.json`        ](../config/sweep/sokoban-7722-m-plan-only-100k.json        ) |
| Sokoban, large search-augmented model    | [`config/sweep/sokoban-7722-l-trace-plan-100k.json`       ](../config/sweep/sokoban-7722-l-trace-plan-100k.json       ) |
| Sokoban, medium search-augmented model   | [`config/sweep/sokoban-7722-m-trace-plan-100k.json`       ](../config/sweep/sokoban-7722-m-trace-plan-100k.json       ) |
| Searchformer model, step 1               | [`config/sweep/sokoban-7722-m-trace-plan-100k-step-1.json`](../config/sweep/sokoban-7722-m-trace-plan-100k-step-1.json) |
| Searchformer model, step 2               | [`config/sweep/sokoban-7722-m-trace-plan-100k-step-2.json`](../config/sweep/sokoban-7722-m-trace-plan-100k-step-2.json) |
| Searchformer model, step 3               | [`config/sweep/sokoban-7722-m-trace-plan-100k-step-3.json`](../config/sweep/sokoban-7722-m-trace-plan-100k-step-3.json) |


## Listing and deleting runs

All training runs can be listed with

```
python -m searchformer.train list-all
```

An individual run with `$ID` can be deleted with 

```
python -m searchformer.train drop-run --run-id=$ID
```

## Importing checkpoints

Checkpoints stored in MongoDB can be listed by running `python -m searchformer.train list-all-checkpoints`.
A checkpoint can be imported with `python -m searchformer.train import-checkpoint` and exported with `python -m searchformer.train export-checkpoint`.

## Related commands

The following commands can be used with the training module.
The `--help` flag will print the corresponding help text.

```
python -m searchformer.train bulk-drop-run --help
python -m searchformer.train bulk-export-checkpoint --help
python -m searchformer.train drop-run --help
python -m searchformer.train export-checkpoint --help
python -m searchformer.train import-checkpoint --help
python -m searchformer.train list-all-checkpoints --help
python -m searchformer.train list-all --help
python -m searchformer.train single --help
python -m searchformer.train sweep --help
```