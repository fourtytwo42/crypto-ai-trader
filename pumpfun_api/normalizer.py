from __future__ import annotations

import numpy as np
import pandas as pd


class RollingNormalizer:
    def __init__(self, window: int = 60, min_periods: int | None = None) -> None:
        self.window = window
        self.min_periods = min_periods or window

    def fit_transform(self, df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        result = df.copy()
        for col in columns:
            if col not in result.columns:
                continue
            rolling_mean = result[col].rolling(window=self.window, min_periods=self.min_periods).mean()
            rolling_std = result[col].rolling(window=self.window, min_periods=self.min_periods).std()
            rolling_std = rolling_std.replace(0, 1.0)
            result[col] = (result[col] - rolling_mean) / rolling_std
        return result
