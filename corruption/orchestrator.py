"""
Corruption orchestrator.

Coordinates multiple corruption strategies to create dirty datasets.
"""

from typing import Any, Dict, List, Optional

import pandas as pd

from corruption.base import CorruptionConfig, CorruptionStrategy
from corruption.duplicates import DuplicatesCorruption
from corruption.missing_values import MissingValuesCorruption
from corruption.numeric_issues import NumericOutliersCorruption
from corruption.text_issues import (
    TextCaseCorruption,
    TextSpecialCharsCorruption,
    TextWhitespaceCorruption,
)
from corruption.type_corruption import TypeCorruption


class CorruptionOrchestrator:
    """Orchestrate multiple corruption strategies."""

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize corruption orchestrator.

        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        self.strategies: Dict[str, CorruptionStrategy] = {
            "missing_values": MissingValuesCorruption(seed),
            "text_whitespace": TextWhitespaceCorruption(seed),
            "text_case": TextCaseCorruption(seed),
            "text_special_chars": TextSpecialCharsCorruption(seed),
            "type_corruption": TypeCorruption(seed),
            "outliers": NumericOutliersCorruption(seed),
            "duplicates": DuplicatesCorruption(seed),
        }

    def corrupt(
        self, df_clean: pd.DataFrame, config: CorruptionConfig
    ) -> tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Apply corruption strategies to create a dirty dataset.

        Args:
            df_clean: Clean dataframe
            config: Corruption configuration

        Returns:
            Tuple of (corrupted dataframe, metadata dict)
        """
        df_corrupt = df_clean.copy()
        applied_strategies = []
        all_operations = []

        print(f"\nApplying {len(config.strategies)} corruption strategies...")

        # Apply each selected strategy
        for strategy_name in config.strategies:
            if strategy_name not in self.strategies:
                print(f"Warning: Unknown strategy '{strategy_name}', skipping")
                continue

            strategy = self.strategies[strategy_name]
            print(f"  - Applying {strategy_name}...")

            # Apply corruption
            df_corrupt = strategy.corrupt(df_corrupt, config.rates)

            # Track metadata
            applied_strategies.append(strategy_name)
            all_operations.extend(strategy.get_required_operations())

        # Generate metadata
        metadata = {
            "corruption_config": config.to_dict(),
            "applied_strategies": applied_strategies,
            "required_operations": list(set(all_operations)),  # Unique operations
            "clean_shape": df_clean.shape,
            "corrupt_shape": df_corrupt.shape,
            "corruption_summary": self._generate_summary(df_clean, df_corrupt),
        }

        return df_corrupt, metadata

    def _generate_summary(
        self, df_clean: pd.DataFrame, df_corrupt: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Generate a summary of corruptions applied.

        Args:
            df_clean: Original clean dataframe
            df_corrupt: Corrupted dataframe

        Returns:
            Dictionary with corruption statistics
        """
        summary = {}

        # Missing values
        clean_missing = df_clean.isnull().sum().sum()
        corrupt_missing = df_corrupt.isnull().sum().sum()
        summary["missing_values_added"] = int(corrupt_missing - clean_missing)
        summary["missing_percentage"] = round(
            (corrupt_missing / df_corrupt.size) * 100, 2
        )

        # Duplicates
        clean_duplicates = df_clean.duplicated().sum()
        corrupt_duplicates = df_corrupt.duplicated().sum()
        summary["duplicates_added"] = int(corrupt_duplicates - clean_duplicates)
        summary["duplicate_percentage"] = round(
            (corrupt_duplicates / len(df_corrupt)) * 100, 2
        )

        # Row count change
        summary["rows_added"] = len(df_corrupt) - len(df_clean)

        # Type changes
        clean_types = df_clean.dtypes.value_counts().to_dict()
        corrupt_types = df_corrupt.dtypes.value_counts().to_dict()
        summary["type_changes"] = {
            "clean_types": {str(k): v for k, v in clean_types.items()},
            "corrupt_types": {str(k): v for k, v in corrupt_types.items()},
        }

        return summary

    def get_strategy_logs(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get corruption logs from all strategies.

        Returns:
            Dictionary mapping strategy name to its log
        """
        logs = {}
        for name, strategy in self.strategies.items():
            log = strategy.get_corruption_log()
            if log:
                logs[name] = log
        return logs

    def clear_all_logs(self):
        """Clear logs from all strategies."""
        for strategy in self.strategies.values():
            strategy.clear_log()


def corrupt_dataset(
    df_clean: pd.DataFrame,
    config: Optional[CorruptionConfig] = None,
    seed: Optional[int] = None,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Convenience function to corrupt a clean dataset.

    Args:
        df_clean: Clean dataframe to corrupt
        config: Corruption configuration (defaults to medium preset)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (corrupted dataframe, metadata dict)
    """
    if config is None:
        config = CorruptionConfig.from_preset("medium", seed=seed)

    orchestrator = CorruptionOrchestrator(seed=seed)
    return orchestrator.corrupt(df_clean, config)
