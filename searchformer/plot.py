# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import functools

import pandas as pd

from .train import TrainRunData
from .utils import split_df_columns_to_value


class TrainLogs:
    """This class is used for extracting training logs and converting them
    to Pandas data frames.
    """

    def __init__(self, id_regex: str):
        """Instatiates a log set for all training runs whose ID matches with
        the provided regex.

        Args:
            id_regex (str): Regular expression used for matching.
        """
        self.id_regex = id_regex

    @property
    def logs(self) -> pd.DataFrame:
        """Training logs.

        Returns:
            pd.DataFrame: Training logs.
        """
        df = pd.concat((self.train_logs, self.test_logs))
        df = pd.merge(df, self.configs, on="_id")
        df.sort_values(by=["step"], inplace=True)
        return df.copy()

    @functools.cached_property
    def run_data(self) -> TrainRunData:
        return TrainRunData()

    @property
    def configs(self) -> pd.DataFrame:
        """Returns data frame with hyper-parameters with all training runs.

        Returns:
            pd.DataFrame: Data frame with all training run hyper-parameter.
        """
        df = self.run_data.load_sweep_config(
            regex=self.id_regex,
            remove_common_configs=False,
        )

        if "data.plan_only" in df.columns:
            method_map = {
                True: "plan-only",
                False: "trace-plan",
            }
            df["Method"] = [method_map[b] for b in df["data.plan_only"]]
            df.drop(columns=["data.plan_only"], inplace=True)  # type: ignore
        if "encoder" in df.columns and "decoder" in df.columns:
            enc_dec_it = df[["encoder", "decoder"]].values
            df["Model"] = [f"{e} {d}" for e, d in enc_dec_it]
        return df.copy()

    @functools.cached_property
    def train_logs(self) -> pd.DataFrame:
        id_list = self.configs["_id"].values.tolist()
        assert type(id_list) is list
        logs = self.run_data.bulk_load_train_logs(id_list)
        logs = split_df_columns_to_value(
            logs,
            index=["_id", "step"],
            split_columns=("type", "portion"),
        )
        logs["is_test"] = False
        return logs

    @functools.cached_property
    def test_logs(self) -> pd.DataFrame:
        id_list = self.configs["_id"].values.tolist()
        assert type(id_list) is list
        logs = self.run_data.bulk_load_test_logs(id_list)
        logs = split_df_columns_to_value(
            logs,
            index=["_id", "step"],
            split_columns=("type", "portion"),
        )
        logs["is_test"] = True
        return logs
