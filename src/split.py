"""M3: time-based train/validation split. Contract: docs/contracts.md#m3."""
from datetime import timedelta

import pandas as pd

from src import config


def time_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split on `created_at_ist` at the config dates, keeping row order and index. Every row must fall in a window."""
    created = df["created_at_ist"]

    def within(first, last):
        # Half-open on the day after `last`, so times late on the last day are kept.
        return (created >= pd.Timestamp(first)) & (created < pd.Timestamp(last + timedelta(days=1)))

    in_train = within(config.TRAIN_START, config.TRAIN_END)
    in_valid = within(config.VALID_START, config.VALID_END)
    outside = int((~(in_train | in_valid)).sum())
    if outside:
        raise ValueError(f"{outside} rows are outside the train and validation windows or have no date")
    return df[in_train].copy(), df[in_valid].copy()
