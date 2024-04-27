# Searchformer

Official code base for the paper titled [_Beyond A*: Better Planning with Transformers via Search Dynamics Bootstrapping_](https://arxiv.org/abs/2402.14083).

This repository contains code for accessing the generated datasets and trained models, re-producing the figures of the main paper, and the code used for running the presented experiments.

## Overview

All code is designed around storing and transforming datasets stored in a MongoDB instance.
The [`notebook`](./notebook) folder contains Jupyer notebooks with examples demonstrating how to access token dataset and prompting a trained Searchformer model.
This folder also contains notebooks that read data from MongoDB to generate all figures included in the main paper.

The `searchformer` module contains all code used in the presented experiments.
Please refer to the documentation in the folder [`doc`](./doc) for using this module to train models, evaluate them, and generate training datasets.

## Setup and installation

This code base uses `python=3.10`.
To run python code and the included Jupyer notebooks, a virtual environment can be created using the included `requirements.txt` file.
For example, this virtual environment can be created with

```
$ python3.10 -m venv venv
$ source venv/bin/activate
(venv) $ pip install -r requirements.txt 
```

For the code to run correctly, a MongoDB instance needs to be setup.
This code base is designed to work with [MongoDB Community Edition](https://www.mongodb.com/try/download/community) and connects by default directly to `mongodb://localhost:27017/mongo` without any user authentication setup.
For example, to explore the included token datasets and checkpoints, a MonogDB instance could be installed locally in a laptop and this code base can be used to access a MongoDB instance running on `localhost`.
To direct the `searchformer` module to connect to a different MongoDB instance, the default MongoDB URI can be overwritten by setting the environment variable

```
export MONGODB_URI=mongodb://localhost:27017/mongo
```

before running any python code.
The code segment used for connecting to MongoDB can be found in `searchformer.utils` in function `mongodb_client`.

To import the released datasets the command line tool [`mongorestore`](https://www.mongodb.com/docs/database-tools/mongorestore/) is used.
Instructions for downloading and installing MongoDB Database Tools can be found [here](https://www.mongodb.com/docs/database-tools/).

**As a convention, all commands are run from this repository's root directory and not from any sub-directory. The root directory is the directory that contains this README file.**

For example, a correct setup of the python environment can be tested by running the following from the repositories root directory: 

```
$ python -m searchformer.train --help
Usage: python -m searchformer.train [OPTIONS] COMMAND [ARGS]...

Options:
  --help  Show this message and exit.

Commands:
  bulk-drop-run           Drop multiple training runs.
  bulk-export-checkpoint  Bulk export of all checkpoints into current...
  bulk-import-checkpoint  Bulk import checkpoints from directory.
  drop-run                Drop individual training run.
  export-checkpoint       Export a checkpoint stored in MongoDB to a file.
  import-checkpoint       Import a checkpoint into MongoDB from file.
  list-all                List all training runs.
  list-all-checkpoints    List all stored checkpoints.
  single                  Start single DDP training run.
  sweep                   Start single DDP training run with provided...
```

## Getting started

We have included Jupyter notebooks in the [`notebook`](./notebook) folder.
These notebooks contain instructions how to download and populate the MongoDB instance with our datasets and how to access the data.
The following notebooks are included:

* `notebook/ExampleLoadCheckpoint.ipynb`: Loading and prompting a search-augmented model with a 10x10 maze navigation task.
* `notebook/ExampleRolloutDatasets.ipynb`: Loading rollout datasets with the generated response sequences.
* `notebook/ExampleTokenDatasets.ipynb`: Loading token datasets used for training.
* `notebook/Maze.ipynb`: Generating all figures contained in the paper regarding any of the maze experiments.
* `notebook/PerfTable.ipynb`: Generating the table with the presented performance numbers.
* `notebook/Searchformer.ipynb`: Generating figures related to the Searchformer experiments.
* `notebook/SearchformerScatter.ipynb`: Generating scatter plot figure related to the Searchformer experiments.
* `notebook/TraceComparison.ipynb`: Generating box plots showing token sequence lengths for different datasets.

The [`doc`](./doc) folder contains documentation about the rest of the code base:

* `doc/train.md`: Outlines how to run the training loop for each model.
* `doc/rollout.md`: Outlines how to generate response sequence datasets using the trained checkpoints.
* `doc/trace_generation.md`: Outlines how to generate the used training data.
* `doc/sokoban.md`: We include a small example to play a Sokoban level from the training data interactively.
* `doc/mongodb.md`: Provides an overview of the included datasets with download links.
* `doc/checkpoint_index.csv`: Lists all included checkpoints with download links.

## License
See the [LICENSE](./LICENSE) file for details about the license under which this code is made available.

## Citation
If you find this repository useful in your research, please consider giving a star :star: and a citation
```
@misc{lehnert2024beyondastar,
      title={Beyond A*: Better Planning with Transformers via Search Dynamics Bootstrapping}, 
      author={Lucas Lehnert and Sainbayar Sukhbaatar and DiJia Su and Qinqing Zheng and Paul Mcvay and Michael Rabbat and Yuandong Tian},
      year={2024},
      eprint={2402.14083},
      archivePrefix={arXiv},
      primaryClass={cs.AI}
}
```
