"""
Text corruption strategies.

Introduces whitespace, case variations, and special characters.
"""

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from corruption.base import CorruptionStrategy


class TextWhitespaceCorruption(CorruptionStrategy):
    """Add whitespace issues to text columns."""

    def __init__(self, seed=None):
        super().__init__(seed)
        self.rng = np.random.default_rng(seed)

    def corrupt(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """
        Add whitespace to text columns.

        Patterns:
        - Leading spaces
        - Trailing spaces
        - Multiple internal spaces

        Args:
            df: Clean dataframe
            config: Must contain 'whitespace_rate'

        Returns:
            Corrupted dataframe
        """
        df_corrupt = df.copy()
        whitespace_rate = config.get("whitespace_rate", 0.20)

        text_cols = df_corrupt.select_dtypes(include=["object"]).columns.tolist()
        total_corrupted = 0

        for col in text_cols:
            # Skip if column has NaN values
            valid_mask = df_corrupt[col].notna()
            valid_indices = df_corrupt[valid_mask].index.tolist()

            if len(valid_indices) == 0:
                continue

            num_to_corrupt = int(len(valid_indices) * whitespace_rate)
            if num_to_corrupt == 0:
                continue

            corrupt_indices = self.rng.choice(
                valid_indices, size=num_to_corrupt, replace=False
            )

            for idx in corrupt_indices:
                value = str(df_corrupt.loc[idx, col])
                corruption_type = self.rng.choice(
                    ["leading", "trailing", "both", "internal"]
                )

                if corruption_type == "leading":
                    value = "  " + value
                elif corruption_type == "trailing":
                    value = value + "  "
                elif corruption_type == "both":
                    value = "  " + value + "  "
                elif corruption_type == "internal":
                    # Add extra spaces between words
                    value = value.replace(" ", "  ")

                df_corrupt.loc[idx, col] = value

            total_corrupted += num_to_corrupt

        self.log_corruption(
            "text_whitespace",
            {
                "total_cells_corrupted": total_corrupted,
                "whitespace_rate": whitespace_rate,
                "text_columns": text_cols,
            },
        )

        return df_corrupt

    def get_required_operations(self) -> List[str]:
        """Return agent operations needed to fix this corruption."""
        return ["clean_text_columns"]


class TextCaseCorruption(CorruptionStrategy):
    """Add case variations to text columns."""

    def __init__(self, seed=None):
        super().__init__(seed)
        self.rng = np.random.default_rng(seed)

    def corrupt(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """
        Vary case in text columns.

        Patterns:
        - UPPERCASE
        - lowercase
        - Title Case
        - rAnDoM CaSe

        Args:
            df: Clean dataframe
            config: Must contain 'case_rate'

        Returns:
            Corrupted dataframe
        """
        df_corrupt = df.copy()
        case_rate = config.get("case_rate", 0.25)

        text_cols = df_corrupt.select_dtypes(include=["object"]).columns.tolist()
        total_corrupted = 0

        for col in text_cols:
            valid_mask = df_corrupt[col].notna()
            valid_indices = df_corrupt[valid_mask].index.tolist()

            if len(valid_indices) == 0:
                continue

            num_to_corrupt = int(len(valid_indices) * case_rate)
            if num_to_corrupt == 0:
                continue

            corrupt_indices = self.rng.choice(
                valid_indices, size=num_to_corrupt, replace=False
            )

            for idx in corrupt_indices:
                value = str(df_corrupt.loc[idx, col])
                corruption_type = self.rng.choice(
                    ["upper", "lower", "title", "random"]
                )

                if corruption_type == "upper":
                    value = value.upper()
                elif corruption_type == "lower":
                    value = value.lower()
                elif corruption_type == "title":
                    value = value.title()
                elif corruption_type == "random":
                    # Randomly capitalize each character
                    value = "".join(
                        c.upper() if self.rng.random() > 0.5 else c.lower()
                        for c in value
                    )

                df_corrupt.loc[idx, col] = value

            total_corrupted += num_to_corrupt

        self.log_corruption(
            "text_case",
            {
                "total_cells_corrupted": total_corrupted,
                "case_rate": case_rate,
                "text_columns": text_cols,
            },
        )

        return df_corrupt

    def get_required_operations(self) -> List[str]:
        """Return agent operations needed to fix this corruption."""
        return ["clean_text_columns"]


class TextSpecialCharsCorruption(CorruptionStrategy):
    """Add special characters to text columns."""

    def __init__(self, seed=None):
        super().__init__(seed)
        self.rng = np.random.default_rng(seed)

    def corrupt(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """
        Add special characters to text columns.

        Patterns:
        - Punctuation: !!!, ???, ...
        - Symbols: @@@, ###, $$$
        - Mixed: value@@@, ###value

        Args:
            df: Clean dataframe
            config: Must contain 'special_char_rate'

        Returns:
            Corrupted dataframe
        """
        df_corrupt = df.copy()
        special_char_rate = config.get("special_char_rate", 0.10)

        text_cols = df_corrupt.select_dtypes(include=["object"]).columns.tolist()
        total_corrupted = 0

        special_chars = ["!", "?", ".", "@", "#", "$", "*", "&", "%"]

        for col in text_cols:
            valid_mask = df_corrupt[col].notna()
            valid_indices = df_corrupt[valid_mask].index.tolist()

            if len(valid_indices) == 0:
                continue

            num_to_corrupt = int(len(valid_indices) * special_char_rate)
            if num_to_corrupt == 0:
                continue

            corrupt_indices = self.rng.choice(
                valid_indices, size=num_to_corrupt, replace=False
            )

            for idx in corrupt_indices:
                value = str(df_corrupt.loc[idx, col])
                char = self.rng.choice(special_chars)
                corruption_type = self.rng.choice(["prefix", "suffix", "both"])

                if corruption_type == "prefix":
                    value = char * 3 + value
                elif corruption_type == "suffix":
                    value = value + char * 3
                elif corruption_type == "both":
                    value = char * 2 + value + char * 2

                df_corrupt.loc[idx, col] = value

            total_corrupted += num_to_corrupt

        self.log_corruption(
            "text_special_chars",
            {
                "total_cells_corrupted": total_corrupted,
                "special_char_rate": special_char_rate,
                "text_columns": text_cols,
            },
        )

        return df_corrupt

    def get_required_operations(self) -> List[str]:
        """Return agent operations needed to fix this corruption."""
        return ["clean_text_columns"]
