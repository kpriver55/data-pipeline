"""
Duplicates corruption strategy.

Introduces exact duplicate rows.
"""

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from corruption.base import CorruptionStrategy


class DuplicatesCorruption(CorruptionStrategy):
    """Introduce exact duplicate rows."""

    def __init__(self, seed=None):
        super().__init__(seed)
        self.rng = np.random.default_rng(seed)

    def corrupt(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """
        Introduce exact duplicate rows.

        Strategy:
        - Randomly select rows to duplicate
        - Insert duplicates at random positions (not adjacent to original)
        - Create exact copies (no variations)

        Args:
            df: Clean dataframe
            config: Must contain 'duplicate_rate' (fraction of rows to duplicate)

        Returns:
            Corrupted dataframe with duplicate rows
        """
        df_corrupt = df.copy()
        duplicate_rate = config.get("duplicate_rate", 0.05)

        num_to_duplicate = int(len(df_corrupt) * duplicate_rate)

        if num_to_duplicate == 0:
            return df_corrupt

        # Randomly select rows to duplicate
        rows_to_duplicate = self.rng.choice(
            len(df_corrupt), size=num_to_duplicate, replace=True
        )

        # Get the duplicate rows
        duplicate_rows = df_corrupt.iloc[rows_to_duplicate].copy()

        # Append duplicates to the dataframe
        df_corrupt = pd.concat([df_corrupt, duplicate_rows], ignore_index=True)

        # Shuffle to randomize duplicate positions
        # (so they're not all at the end)
        df_corrupt = df_corrupt.sample(frac=1, random_state=self.seed).reset_index(
            drop=True
        )

        self.log_corruption(
            "duplicates",
            {
                "num_duplicates_added": num_to_duplicate,
                "duplicate_rate": duplicate_rate,
                "original_rows": len(df),
                "final_rows": len(df_corrupt),
            },
        )

        return df_corrupt

    def get_required_operations(self) -> List[str]:
        """Return agent operations needed to fix this corruption."""
        return ["remove_duplicates"]
