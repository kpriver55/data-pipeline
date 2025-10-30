"""
Categorical data generator for clean datasets.

Supports frequency distributions and LLM-assisted value generation.
"""

from typing import Optional

import numpy as np
import pandas as pd

from schema.schema import CategoricalConfig


class CategoricalGenerator:
    """Generate categorical data based on schema configuration."""

    def __init__(self, seed: Optional[int] = None, llm_client=None):
        """
        Initialize categorical generator.

        Args:
            seed: Random seed for reproducibility
            llm_client: Optional LLM client for generating values
        """
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.llm_client = llm_client

    def generate(self, config: CategoricalConfig, num_rows: int) -> pd.Series:
        """
        Generate categorical data according to configuration.

        Args:
            config: CategoricalConfig with values and frequencies
            num_rows: Number of values to generate

        Returns:
            pandas Series of categorical values
        """
        values = config.values
        frequencies = config.frequencies

        # If no frequencies specified, use uniform distribution
        if frequencies is None:
            frequencies = [1.0 / len(values)] * len(values)

        # Generate data according to frequency distribution
        data = self.rng.choice(values, size=num_rows, p=frequencies)

        return pd.Series(data, dtype="object")


def generate_categorical_column(
    config: CategoricalConfig, num_rows: int, seed: Optional[int] = None
) -> pd.Series:
    """
    Convenience function to generate a categorical column.

    Args:
        config: CategoricalConfig with values and frequencies
        num_rows: Number of values to generate
        seed: Random seed for reproducibility

    Returns:
        pandas Series of categorical values
    """
    generator = CategoricalGenerator(seed=seed)
    return generator.generate(config, num_rows)
