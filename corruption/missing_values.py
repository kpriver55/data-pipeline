"""
Missing values corruption strategy.

Introduces NULL/NaN values, empty strings, and sentinel values.
"""

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from corruption.base import CorruptionStrategy


class MissingValuesCorruption(CorruptionStrategy):
    """Introduce missing values in various forms."""

    def __init__(self, seed=None):
        super().__init__(seed)
        self.rng = np.random.default_rng(seed)

    def corrupt(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """
        Introduce missing values into the dataframe.

        Strategy:
        - Numeric columns: Replace with NaN or sentinel values (-999, -1, 0)
        - Text columns: Replace with NaN, empty string, or None

        Args:
            df: Clean dataframe
            config: Must contain 'missing_rate' (fraction of cells to corrupt)

        Returns:
            Corrupted dataframe with missing values
        """
        df_corrupt = df.copy()
        missing_rate = config.get("missing_rate", 0.15)

        numeric_cols = df_corrupt.select_dtypes(include=[np.number]).columns.tolist()
        text_cols = df_corrupt.select_dtypes(include=["object"]).columns.tolist()

        total_corrupted = 0

        # Corrupt numeric columns
        for col in numeric_cols:
            num_to_corrupt = int(len(df_corrupt) * missing_rate)
            if num_to_corrupt == 0:
                continue

            # Randomly select rows to corrupt
            corrupt_indices = self.rng.choice(
                len(df_corrupt), size=num_to_corrupt, replace=False
            )

            # Randomly choose corruption type for each row
            for idx in corrupt_indices:
                corruption_type = self.rng.choice(
                    ["nan", "sentinel_negative", "sentinel_zero", "sentinel_large"]
                )

                if corruption_type == "nan":
                    df_corrupt.loc[idx, col] = np.nan
                elif corruption_type == "sentinel_negative":
                    df_corrupt.loc[idx, col] = -999
                elif corruption_type == "sentinel_zero":
                    df_corrupt.loc[idx, col] = 0
                elif corruption_type == "sentinel_large":
                    # Use a large sentinel value
                    df_corrupt.loc[idx, col] = 999999

            total_corrupted += num_to_corrupt

        # Corrupt text columns
        for col in text_cols:
            num_to_corrupt = int(len(df_corrupt) * missing_rate)
            if num_to_corrupt == 0:
                continue

            corrupt_indices = self.rng.choice(
                len(df_corrupt), size=num_to_corrupt, replace=False
            )

            for idx in corrupt_indices:
                corruption_type = self.rng.choice(
                    ["nan", "empty_string", "none_string"]
                )

                if corruption_type == "nan":
                    df_corrupt.loc[idx, col] = np.nan
                elif corruption_type == "empty_string":
                    df_corrupt.loc[idx, col] = ""
                elif corruption_type == "none_string":
                    df_corrupt.loc[idx, col] = None

            total_corrupted += num_to_corrupt

        self.log_corruption(
            "missing_values",
            {
                "total_cells_corrupted": total_corrupted,
                "missing_rate": missing_rate,
                "numeric_columns": numeric_cols,
                "text_columns": text_cols,
            },
        )

        return df_corrupt

    def get_required_operations(self) -> List[str]:
        """Return agent operations needed to fix this corruption."""
        return ["handle_missing_values"]
