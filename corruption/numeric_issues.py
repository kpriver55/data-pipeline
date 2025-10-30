"""
Numeric corruption strategies.

Introduces outliers detectable by IQR or z-score methods.
"""

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from corruption.base import CorruptionStrategy


class NumericOutliersCorruption(CorruptionStrategy):
    """Introduce outliers into numeric columns."""

    # Cap outlier rate to prevent distribution shift
    MAX_OUTLIER_RATE = 0.05  # 5% maximum

    def __init__(self, seed=None):
        super().__init__(seed)
        self.rng = np.random.default_rng(seed)

    def corrupt(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """
        Introduce outliers into numeric columns.

        Strategy:
        - Make values extreme enough to be caught by IQR or z-score methods
        - Use standard IQR threshold of 1.5, but create outliers at 2.5x severity
        - Add safety margin (1.2x multiplier) to account for distribution shift
        - Cap outlier rate at 5% to prevent significant distribution changes

        Args:
            df: Clean dataframe
            config: Must contain 'outlier_rate' and 'outlier_severity'

        Returns:
            Corrupted dataframe with outliers
        """
        df_corrupt = df.copy()
        outlier_rate = min(
            config.get("outlier_rate", 0.03), self.MAX_OUTLIER_RATE
        )
        severity = config.get("outlier_severity", 2.5)

        # Safety multiplier to ensure outliers remain detectable after distribution shift
        # This is modest (1.2x) since we're already using severity=2.5 (vs standard 1.5)
        safety_margin = 1.2

        numeric_cols = df_corrupt.select_dtypes(include=[np.number]).columns.tolist()
        total_corrupted = 0

        for col in numeric_cols:
            # Skip if all NaN
            if df_corrupt[col].isna().all():
                continue

            num_to_corrupt = int(len(df_corrupt) * outlier_rate)
            if num_to_corrupt == 0:
                continue

            # Get valid (non-NaN) indices
            valid_mask = df_corrupt[col].notna()
            valid_indices = df_corrupt[valid_mask].index.tolist()

            if len(valid_indices) == 0:
                continue

            # Limit corruptions to available valid indices
            num_to_corrupt = min(num_to_corrupt, len(valid_indices))
            corrupt_indices = self.rng.choice(
                valid_indices, size=num_to_corrupt, replace=False
            )

            # Calculate IQR bounds on CLEAN distribution
            q25 = df_corrupt[col].quantile(0.25)
            q75 = df_corrupt[col].quantile(0.75)
            iqr = q75 - q25

            # Create bounds using severity threshold
            lower_bound = q25 - severity * iqr
            upper_bound = q75 + severity * iqr

            for idx in corrupt_indices:
                # Randomly choose to create high or low outlier
                if self.rng.random() > 0.5:
                    # High outlier - go beyond upper bound with safety margin
                    # Use safety_margin to ensure still detectable after distribution shift
                    margin = iqr * safety_margin * self.rng.uniform(0.5, 1.5)
                    outlier_value = upper_bound + margin
                else:
                    # Low outlier - go below lower bound with safety margin
                    margin = iqr * safety_margin * self.rng.uniform(0.5, 1.5)
                    outlier_value = lower_bound - margin

                # Ensure we don't create negative values for columns that shouldn't have them
                # (Heuristic: if current min is >= 0, assume non-negative column)
                if df_corrupt[col].min() >= 0 and outlier_value < 0:
                    # Create high outlier instead
                    margin = iqr * safety_margin * self.rng.uniform(0.5, 1.5)
                    outlier_value = upper_bound + margin

                # For integer columns, round the outlier value to avoid dtype warning
                if pd.api.types.is_integer_dtype(df_corrupt[col]):
                    outlier_value = int(np.round(outlier_value))

                df_corrupt.loc[idx, col] = outlier_value

            total_corrupted += num_to_corrupt

        self.log_corruption(
            "numeric_outliers",
            {
                "total_cells_corrupted": total_corrupted,
                "outlier_rate": outlier_rate,
                "severity": severity,
                "safety_margin": safety_margin,
                "numeric_columns": numeric_cols,
            },
        )

        return df_corrupt

    def get_required_operations(self) -> List[str]:
        """Return agent operations needed to fix this corruption."""
        return ["remove_outliers"]
