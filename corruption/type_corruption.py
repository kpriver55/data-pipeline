"""
Type corruption strategy.

Converts columns to wrong but parseable types.
"""

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from corruption.base import CorruptionStrategy


class TypeCorruption(CorruptionStrategy):
    """Convert columns to incorrect but parseable data types."""

    def __init__(self, seed=None):
        super().__init__(seed)
        self.rng = np.random.default_rng(seed)

    def corrupt(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """
        Convert columns to wrong data types.

        Strategy:
        - Numeric columns → string (e.g., 123 → "123")
        - Datetime columns → string in pandas-parseable format
        - Keep conversions reversible with coercion

        Args:
            df: Clean dataframe
            config: Must contain 'type_corruption_rate' (fraction of columns)

        Returns:
            Corrupted dataframe with type mismatches
        """
        df_corrupt = df.copy()
        type_corruption_rate = config.get("type_corruption_rate", 0.30)

        corrupted_columns = []

        # Get candidate columns (numeric and datetime)
        numeric_cols = df_corrupt.select_dtypes(include=[np.number]).columns.tolist()
        datetime_cols = df_corrupt.select_dtypes(include=["datetime64"]).columns.tolist()

        candidate_cols = numeric_cols + datetime_cols

        if len(candidate_cols) == 0:
            return df_corrupt

        # Determine how many columns to corrupt
        num_to_corrupt = max(1, int(len(candidate_cols) * type_corruption_rate))
        num_to_corrupt = min(num_to_corrupt, len(candidate_cols))

        # Randomly select columns to corrupt
        cols_to_corrupt = self.rng.choice(
            candidate_cols, size=num_to_corrupt, replace=False
        ).tolist()

        for col in cols_to_corrupt:
            original_type = df_corrupt[col].dtype

            if col in numeric_cols:
                # Convert numeric to string
                df_corrupt[col] = df_corrupt[col].astype(str)
                corrupted_columns.append(
                    {"column": col, "from": str(original_type), "to": "string (from numeric)"}
                )

            elif col in datetime_cols:
                # Convert datetime to string in parseable format
                # Use ISO format which pandas can parse
                df_corrupt[col] = df_corrupt[col].dt.strftime("%Y-%m-%d %H:%M:%S")
                corrupted_columns.append(
                    {"column": col, "from": str(original_type), "to": "string (from datetime)"}
                )

        self.log_corruption(
            "type_corruption",
            {
                "num_columns_corrupted": len(corrupted_columns),
                "type_corruption_rate": type_corruption_rate,
                "corrupted_columns": corrupted_columns,
            },
        )

        return df_corrupt

    def get_required_operations(self) -> List[str]:
        """Return agent operations needed to fix this corruption."""
        return ["convert_data_types"]
