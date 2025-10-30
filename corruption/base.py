"""
Base corruption strategy interface.

All corruption strategies inherit from this base class.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


class CorruptionStrategy(ABC):
    """Abstract base class for corruption strategies."""

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize corruption strategy.

        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        self.corruption_log: List[Dict[str, Any]] = []

    @abstractmethod
    def corrupt(
        self, df: pd.DataFrame, config: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Apply corruption to a dataframe.

        Args:
            df: Clean dataframe to corrupt
            config: Configuration parameters for this corruption strategy

        Returns:
            Corrupted dataframe
        """
        pass

    @abstractmethod
    def get_required_operations(self) -> List[str]:
        """
        Return list of agent operations needed to fix this corruption.

        Returns:
            List of operation names (e.g., ["handle_missing_values"])
        """
        pass

    def log_corruption(self, operation: str, details: Dict[str, Any]):
        """
        Log a corruption operation for tracking.

        Args:
            operation: Name of the corruption operation
            details: Details about what was corrupted
        """
        self.corruption_log.append({"operation": operation, "details": details})

    def get_corruption_log(self) -> List[Dict[str, Any]]:
        """
        Get the log of all corruptions applied.

        Returns:
            List of corruption operation logs
        """
        return self.corruption_log

    def clear_log(self):
        """Clear the corruption log."""
        self.corruption_log = []


class CorruptionConfig:
    """Configuration for corruption engine."""

    # Fixed rates for each corruption type (these don't vary by difficulty)
    DEFAULT_RATES = {
        "missing_rate": 0.15,  # 15% of cells
        "whitespace_rate": 0.20,  # 20% of text cells
        "case_rate": 0.25,  # 25% of text cells
        "special_char_rate": 0.10,  # 10% of text cells
        "duplicate_rate": 0.05,  # 5% of rows
        "outlier_rate": 0.03,  # 3% of numeric rows
        "outlier_severity": 2.5,  # IQR multiplier / z-score threshold
    }

    def __init__(
        self,
        strategies: List[str],
        seed: Optional[int] = None,
        rates: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize corruption configuration.

        Args:
            strategies: List of corruption strategy names to apply
            seed: Random seed for reproducibility
            rates: Optional custom rates (defaults to DEFAULT_RATES)
        """
        self.strategies = strategies
        self.seed = seed
        self.rates = {**self.DEFAULT_RATES, **(rates or {})}

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "strategies": self.strategies,
            "seed": self.seed,
            "rates": self.rates,
        }

    @classmethod
    def from_preset(
        cls, level: str, seed: Optional[int] = None
    ) -> "CorruptionConfig":
        """
        Create config from preset difficulty level with RANDOM strategy selection.

        Difficulty is determined by number of corruption types:
        - Easy: 1-2 random strategies
        - Medium: 3-4 random strategies
        - Hard: 5-7 random strategies

        This randomization prevents models from learning fixed corruption patterns.

        Args:
            level: "easy", "medium", or "hard"
            seed: Random seed for reproducibility

        Returns:
            CorruptionConfig with randomly selected strategies
        """
        rng = np.random.default_rng(seed)

        available = [
            "missing_values",
            "text_whitespace",
            "text_case",
            "text_special_chars",
            "type_corruption",
            "outliers",
            "duplicates",
        ]

        if level == "easy":
            # Easy: 1-2 operations randomly selected
            num_strategies = rng.integers(1, 3)  # 1 or 2

        elif level == "medium":
            # Medium: 3-4 operations randomly selected
            num_strategies = rng.integers(3, 5)  # 3 or 4

        elif level == "hard":
            # Hard: 5-7 operations randomly selected
            num_strategies = rng.integers(5, 8)  # 5, 6, or 7

        else:
            raise ValueError(
                f"Unknown preset level: {level}. Use 'easy', 'medium', or 'hard'"
            )

        # Randomly select strategies (without replacement)
        selected_strategies = rng.choice(
            available, size=num_strategies, replace=False
        ).tolist()

        return cls(strategies=selected_strategies, seed=seed)

    @classmethod
    def custom(
        cls,
        strategies: List[str],
        seed: Optional[int] = None,
        rates: Optional[Dict[str, float]] = None,
    ) -> "CorruptionConfig":
        """
        Create a custom configuration with specific strategies.

        Args:
            strategies: List of corruption strategy names
            seed: Random seed
            rates: Optional custom rates

        Returns:
            CorruptionConfig with custom settings
        """
        return cls(strategies=strategies, seed=seed, rates=rates)


# Available corruption strategies
AVAILABLE_STRATEGIES = [
    "missing_values",
    "text_whitespace",
    "text_case",
    "text_special_chars",
    "type_corruption",
    "outliers",
    "duplicates",
]
