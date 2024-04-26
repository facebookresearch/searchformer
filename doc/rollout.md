# Generating Response Sequences

Models trained on nondeterministic A* data or the Sokoban tasks are evaluated by first generating a response token sequence for each test prompt.
These response token sequences are stored in MongoDB as a dataset.
Subsequently, each model is evaluated by parsing and analyzing these generated sequences.
This evaluation is implemented in the Jupyter notebooks found in the folder `notebook`.

Response sequences are generated and stored in MongoDB using the Rollout workflow.
Here, a checkpoint is loaded by a worker and this worker iterates over a slice of the test token dataset.
For each test prompt multiple response sequences are generated and stored in MongoDB.

For example, to launch a rollout worker that evaluated the checkpoint from the training run with ID `maze-sweep-rep-nondet-large-70` on the test prompt of the token dataset `maze.20-by-20-nondeterministic.simple` run

```
python -m searchformer.rollout probability \
    --rank=0 --world-size=20 \
    --test-sequences=100 \
    --batch-size=16 --rollout-repeats=64 --rollout-len=10000 \
    --dataset-name=maze.20-by-20-nondeterministic.simple \
    --checkpoint-id=maze-sweep-rep-nondet-large-70 \
```

Here, the job is split across 20 workers that evaluate the model in parallel and the command above launches the first worker with rank index 0.
The rank and world size specify which test set slice the worker iterates over.

These jobs use a single GPU (DDP is not used here).
The `--rollout-repeats` parameter sets how many sequences are generated while the `--batch-size` parameter sets how many sequences are generated in parallel on the GPU.
This parameter is set according to available GPU memory.

We include the generated response datasets for each of the experiments presented in the main paper.

## Indexing of rollout datasets

Rollout datasets are uniquely indexed by the parameters provided in the `python -m searchformer.rollout probability`.
For each parameter combination a separate rollout dataset is generated with a unique id.
If the same command is run multiple times, the worker will attempt to create more response sequences.
If responses are exist in MongoDB for a particular test prompt, then the rollout worker will skip this test prompt.

The rollout datasets stored in MongoDB can be viewed by running:

```
from searchformer.rollout import RolloutDataStore


datastore = RolloutDataStore()
rollout_data = datastore.list_all()   # returns a Pandas dataframe with all dataset ids and hyper-parameters
print(rollout_data)
```

## Searchformer experiments

The Searchformer experiments use this Rollout workflow as well to generate the shorter sequence training datasets.
Here, the same command is used but the `--include-train` flag and the `--train-sequences` option is set to generate responses for the training prompt of the specified training dataset.

Once the rollout dataset is generated, a separate workflow is started map the generated sequence dataset to a token dataset that can be used for training.
This reduce worker can be launched with

```
python -m searchformer.sokoban reduce-rollout-to-shortest-trace \
    --rollout-id=$ROLLOUT_ID \
    --origin-dataset=$TOKEN_DATASET \
    --rank=0 \
    --world-size=100
```

where `$ROLLOUT_ID` is the id of the rollout dataset, `$TOKEN_DATASET` is the token dataset that was used to generate the rollout dataset.
As above, rank and world size specify the worker index and the total number of workers that run concurrently to map the dataset.

This operation will then generate a token dataset with the name `$ROLLOUT_ID/improved` that can be used to train or finetune a Transformer model as described [here](train.md).

We also include the generated response dataset for each of the Searchformer models and improvement steps.
