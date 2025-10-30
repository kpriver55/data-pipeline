"""
Numeric data generator for clean datasets.

Supports various distributions and constraints.
"""

import numpy as np
from typing import Optional

from schema.schema import NumericConfig, NumericDistribution


class NumericGenerator:
    """Generate numeric data based on schema configuration."""

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize numeric generator.

        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def generate(
        self, config: NumericConfig, num_rows: int, is_float: bool = True
    ) -> np.ndarray:
        """
        Generate numeric data according to configuration.

        Args:
            config: NumericConfig with distribution and constraints
            num_rows: Number of values to generate
            is_float: Whether to generate float or integer values

        Returns:
            numpy array of generated values
        """
        # Generate data based on distribution
        if config.distribution == NumericDistribution.NORMAL:
            data = self._generate_normal(config, num_rows)
        elif config.distribution == NumericDistribution.UNIFORM:
            data = self._generate_uniform(config, num_rows)
        elif config.distribution == NumericDistribution.EXPONENTIAL:
            data = self._generate_exponential(config, num_rows)
        elif config.distribution == NumericDistribution.POISSON:
            data = self._generate_poisson(config, num_rows)
        elif config.distribution == NumericDistribution.BINOMIAL:
            data = self._generate_binomial(config, num_rows)
        else:
            raise ValueError(f"Unknown distribution: {config.distribution}")

        # Apply constraints
        data = self._apply_constraints(data, config)

        # Convert to appropriate type
        if is_float:
            data = np.round(data, config.decimals)
        else:
            data = np.round(data).astype(np.int64)

        return data

    def _generate_normal(self, config: NumericConfig, num_rows: int) -> np.ndarray:
        """Generate data from normal distribution."""
        mean = config.mean if config.mean is not None else 0.0
        std = config.std if config.std is not None else 1.0

        data = self.rng.normal(loc=mean, scale=std, size=num_rows)
        return data

    def _generate_uniform(self, config: NumericConfig, num_rows: int) -> np.ndarray:
        """Generate data from uniform distribution."""
        min_val = config.min_value if config.min_value is not None else 0.0
        max_val = config.max_value if config.max_value is not None else 1.0

        data = self.rng.uniform(low=min_val, high=max_val, size=num_rows)
        return data

    def _generate_exponential(self, config: NumericConfig, num_rows: int) -> np.ndarray:
        """Generate data from exponential distribution."""
        # Use mean as scale parameter (mean = 1/lambda)
        scale = config.mean if config.mean is not None else 1.0

        data = self.rng.exponential(scale=scale, size=num_rows)

        # Shift if min_value is specified
        if config.min_value is not None:
            data = data + config.min_value

        return data

    def _generate_poisson(self, config: NumericConfig, num_rows: int) -> np.ndarray:
        """Generate data from Poisson distribution."""
        # Use mean as lambda parameter
        lam = config.mean if config.mean is not None else 5.0

        data = self.rng.poisson(lam=lam, size=num_rows).astype(float)
        return data

    def _generate_binomial(self, config: NumericConfig, num_rows: int) -> np.ndarray:
        """Generate data from binomial distribution."""
        # Use max_value as n (number of trials), mean to derive p
        n = int(config.max_value) if config.max_value is not None else 10

        # If mean is provided, derive p from it: mean = n*p => p = mean/n
        if config.mean is not None:
            p = config.mean / n
            p = np.clip(p, 0.0, 1.0)  # Ensure valid probability
        else:
            p = 0.5  # Default probability

        data = self.rng.binomial(n=n, p=p, size=num_rows).astype(float)
        return data

    def _apply_constraints(
        self, data: np.ndarray, config: NumericConfig
    ) -> np.ndarray:
        """Apply min/max constraints and handle negative values."""
        # Clip to min/max if specified
        if config.min_value is not None:
            data = np.maximum(data, config.min_value)

        if config.max_value is not None:
            data = np.minimum(data, config.max_value)

        # Handle negative values constraint
        if not config.allow_negative:
            data = np.abs(data)
            # Re-apply min constraint after taking absolute value
            if config.min_value is not None:
                data = np.maximum(data, config.min_value)

        return data


def generate_numeric_column(
    config: NumericConfig,
    num_rows: int,
    is_float: bool = True,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Convenience function to generate a numeric column.

    Args:
        config: NumericConfig with distribution and constraints
        num_rows: Number of values to generate
        is_float: Whether to generate float or integer values
        seed: Random seed for reproducibility

    Returns:
        numpy array of generated values
    """
    generator = NumericGenerator(seed=seed)
    return generator.generate(config, num_rows, is_float)
