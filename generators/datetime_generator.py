"""
Datetime data generator for clean datasets.

Supports various datetime patterns and constraints.
"""

from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from schema.schema import DatetimeConfig


class DatetimeGenerator:
    """Generate datetime data based on schema configuration."""

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize datetime generator.

        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def generate(self, config: DatetimeConfig, num_rows: int) -> pd.Series:
        """
        Generate datetime data according to configuration.

        Args:
            config: DatetimeConfig with date range and constraints
            num_rows: Number of values to generate

        Returns:
            pandas Series of datetime values
        """
        # Parse date range
        start_dt = pd.to_datetime(config.start_date)
        end_dt = pd.to_datetime(config.end_date)

        # Generate random timestamps as Series
        timestamps = self._generate_random_timestamps(start_dt, end_dt, num_rows)

        # Apply time constraints
        if config.include_time:
            if config.business_hours_only:
                timestamps = self._apply_business_hours(timestamps)
        else:
            # Strip time component if not needed
            timestamps = pd.Series(pd.to_datetime(timestamps.dt.date))

        # Apply weekday constraint
        if config.weekdays_only:
            timestamps = self._apply_weekdays_only(timestamps, start_dt, end_dt)

        return timestamps

    def _generate_random_timestamps(
        self, start_dt: pd.Timestamp, end_dt: pd.Timestamp, num_rows: int
    ) -> pd.Series:
        """Generate random timestamps between start and end dates."""
        # Convert to Unix timestamps (seconds since epoch)
        start_ts = start_dt.value / 10**9
        end_ts = end_dt.value / 10**9

        # Generate random timestamps
        random_ts = self.rng.uniform(start_ts, end_ts, size=num_rows)

        # Convert back to datetime as Series
        timestamps = pd.Series(pd.to_datetime(random_ts, unit="s"))

        return timestamps

    def _apply_business_hours(self, timestamps: pd.Series) -> pd.Series:
        """Constrain times to business hours (9 AM - 5 PM)."""
        # Extract time components
        hours = timestamps.dt.hour
        minutes = timestamps.dt.minute
        seconds = timestamps.dt.second

        # Set hours to be between 9 and 17 (5 PM)
        business_hours = self.rng.integers(9, 17, size=len(timestamps))

        # Create new timestamps with business hours
        new_timestamps = pd.to_datetime(
            {
                "year": timestamps.dt.year,
                "month": timestamps.dt.month,
                "day": timestamps.dt.day,
                "hour": business_hours,
                "minute": minutes,
                "second": seconds,
            }
        )

        return pd.Series(new_timestamps)

    def _apply_weekdays_only(
        self, timestamps: pd.Series, start_dt: pd.Timestamp, end_dt: pd.Timestamp
    ) -> pd.Series:
        """Filter out weekend dates, regenerate for weekdays only."""
        # Identify weekends (Saturday=5, Sunday=6)
        is_weekend = timestamps.dt.dayofweek.isin([5, 6])

        # Regenerate weekend dates as weekdays
        while is_weekend.any():
            num_weekends = is_weekend.sum()

            # Generate new timestamps for weekends
            new_timestamps = self._generate_random_timestamps(
                start_dt, end_dt, num_weekends
            )

            # Replace weekend timestamps
            timestamps.loc[is_weekend] = new_timestamps.values

            # Recheck for weekends
            is_weekend = timestamps.dt.dayofweek.isin([5, 6])

        return timestamps


def generate_datetime_column(
    config: DatetimeConfig, num_rows: int, seed: Optional[int] = None
) -> pd.Series:
    """
    Convenience function to generate a datetime column.

    Args:
        config: DatetimeConfig with date range and constraints
        num_rows: Number of values to generate
        seed: Random seed for reproducibility

    Returns:
        pandas Series of datetime values
    """
    generator = DatetimeGenerator(seed=seed)
    return generator.generate(config, num_rows)
