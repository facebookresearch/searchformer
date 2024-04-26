# MongoDB Bulk Imports

With the exception of individual checkpoint files listed in [`checkpoint_index.csv`](checkpoint_index.csv), all files can be imported with `mongorestore` into a MongoDB instance.
The following table provides an overview of all datasets that can be imported with `mongorestore`:

| File                                                                                                            | Size  | Description                                            |
| --                                                                                                              | ---   | ---                                                    |
| [maze.gz                   ](https://dl.fbaipublicfiles.com/searchformer/tokenSeqDB/maze.gz                   ) | 25GB  | Maze token training data                               |
| [maze.vocabulary.gz        ](https://dl.fbaipublicfiles.com/searchformer/tokenSeqDB/maze.vocabulary.gz        ) | 7.2kB | Maze token meta data                                   |
| [sokoban.gz                ](https://dl.fbaipublicfiles.com/searchformer/tokenSeqDB/sokoban.gz                ) | 1.4GB | Sokoban token training data                            |
| [sokoban.vocabulary.gz     ](https://dl.fbaipublicfiles.com/searchformer/tokenSeqDB/sokoban.vocabulary.gz     ) | 312B  | Sokoban token meta data                                |
| [searchformer.gz           ](https://dl.fbaipublicfiles.com/searchformer/tokenSeqDB/searchformer.gz           ) | 8.6GB | Searchformer token training data                       |
| [searchformer.vocabulary.gz](https://dl.fbaipublicfiles.com/searchformer/tokenSeqDB/searchformer.vocabulary.gz) | 1.9MB | Searchformer token meta data                           |
| [trainDB.gz                ](https://dl.fbaipublicfiles.com/searchformer/trainDB.gz                           ) | 66MB  | Training logs                                          |
| [rolloutDB.gz              ](https://dl.fbaipublicfiles.com/searchformer/rolloutDB.gz                         ) | 66MB  | Response sequences on test tasks                       |
| [sokobanAStarRefDataDB.gz  ](https://dl.fbaipublicfiles.com/searchformer/sokobanAStarRefDataDB.gz             ) | 166MB | A* reference dataset used for Searchformer comparisons |

The large files linked above can be downloaded with `curl` over a lossy or unstable connection. 
For example, the 25GB large maze token dataset can be downloaded with

```
curl --continue-at - https://dl.fbaipublicfiles.com/searchformer/tokenSeqDB/maze.gz -O
```

The `--continue-at -` option instructs `curl` to resume the download at a particular byte position.
This byte position is determined based on an already existing file.
For more information, please refer to [this documentation](https://everything.curl.dev/usingcurl/downloads/resume.html).

## Searchformer response sequence datasets

The response datasets generated for each Searchformer improvement step can be downloaded from the following links.
This data is stored as a rollout dataset and is used to generate the short token sequence datasets linked above.
These short token sequence datasets are then used for fine-tuning the model.

| Checkpoint ID                           | Rollout Dataset ID       | Dataset File                                                                                      | Meta-data File                                                                                              |
| ---                                     | ---                      | ---                                                                                               | ---                                                                                                         |
| sokoban-7722-m-trace-plan-100k-0        | 65b8382b9ee4fbaa76e005b7 | [data.gz](https://dl.fbaipublicfiles.com/searchformer/rolloutDB/train/65b8382b9ee4fbaa76e005b7/data.gz) | [meta_data.gz](https://dl.fbaipublicfiles.com/searchformer/rolloutDB/train/65b8382b9ee4fbaa76e005b7/meta_data.gz) |
| sokoban-7722-m-trace-plan-100k-1        | 65b8398ff7574141c3ba77ae | [data.gz](https://dl.fbaipublicfiles.com/searchformer/rolloutDB/train/65b8398ff7574141c3ba77ae/data.gz) | [meta_data.gz](https://dl.fbaipublicfiles.com/searchformer/rolloutDB/train/65b8398ff7574141c3ba77ae/meta_data.gz) |
| sokoban-7722-m-trace-plan-100k-2        | 65b8495e2382373d6a21ca99 | [data.gz](https://dl.fbaipublicfiles.com/searchformer/rolloutDB/train/65b8495e2382373d6a21ca99/data.gz) | [meta_data.gz](https://dl.fbaipublicfiles.com/searchformer/rolloutDB/train/65b8495e2382373d6a21ca99/meta_data.gz) |
| sokoban-7722-m-trace-plan-100k-0-step-1 | 65ba856d986d307d60c563ca | [data.gz](https://dl.fbaipublicfiles.com/searchformer/rolloutDB/train/65ba856d986d307d60c563ca/data.gz) | [meta_data.gz](https://dl.fbaipublicfiles.com/searchformer/rolloutDB/train/65ba856d986d307d60c563ca/meta_data.gz) |
| sokoban-7722-m-trace-plan-100k-1-step-1 | 65ba8e2678ad82e62025d0c6 | [data.gz](https://dl.fbaipublicfiles.com/searchformer/rolloutDB/train/65ba8e2678ad82e62025d0c6/data.gz) | [meta_data.gz](https://dl.fbaipublicfiles.com/searchformer/rolloutDB/train/65ba8e2678ad82e62025d0c6/meta_data.gz) |
| sokoban-7722-m-trace-plan-100k-2-step-1 | 65ba8ee773586ec314e0da96 | [data.gz](https://dl.fbaipublicfiles.com/searchformer/rolloutDB/train/65ba8ee773586ec314e0da96/data.gz) | [meta_data.gz](https://dl.fbaipublicfiles.com/searchformer/rolloutDB/train/65ba8ee773586ec314e0da96/meta_data.gz) |
| sokoban-7722-m-trace-plan-100k-0-step-2 | 65c8b912c3dd164d1eb691a4 | [data.gz](https://dl.fbaipublicfiles.com/searchformer/rolloutDB/train/65c8b912c3dd164d1eb691a4/data.gz) | [meta_data.gz](https://dl.fbaipublicfiles.com/searchformer/rolloutDB/train/65c8b912c3dd164d1eb691a4/meta_data.gz) |
| sokoban-7722-m-trace-plan-100k-1-step-2 | 65ca34d2e25a422484c5d3da | [data.gz](https://dl.fbaipublicfiles.com/searchformer/rolloutDB/train/65ca34d2e25a422484c5d3da/data.gz) | [meta_data.gz](https://dl.fbaipublicfiles.com/searchformer/rolloutDB/train/65ca34d2e25a422484c5d3da/meta_data.gz) |
| sokoban-7722-m-trace-plan-100k-2-step-2 | 65ca57d67f455f390d05bf33 | [data.gz](https://dl.fbaipublicfiles.com/searchformer/rolloutDB/train/65ca57d67f455f390d05bf33/data.gz) | [meta_data.gz](https://dl.fbaipublicfiles.com/searchformer/rolloutDB/train/65ca57d67f455f390d05bf33/meta_data.gz) |

Each `data.gz` file is 45GB in size.
The `meta_data.gz` files are less than 1kB in size.