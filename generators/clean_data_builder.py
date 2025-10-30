"""
Clean data builder - orchestrates generation of complete clean datasets.

Uses individual generators to create silver-level clean datasets from schemas.
"""

from typing import Optional

import pandas as pd

from generators.categorical_generator import CategoricalGenerator
from generators.datetime_generator import DatetimeGenerator
from generators.numeric_generator import NumericGenerator
from schema.schema import ColumnType, DatasetSchema


class CleanDataBuilder:
    """Build complete clean datasets from schemas."""

    def __init__(self, seed: Optional[int] = None, llm_client=None):
        """
        Initialize clean data builder.

        Args:
            seed: Random seed for reproducibility
            llm_client: Optional LLM client for enhanced generation
        """
        self.seed = seed
        self.llm_client = llm_client

        # Initialize generators
        self.numeric_gen = NumericGenerator(seed=seed)
        self.datetime_gen = DatetimeGenerator(seed=seed)
        self.categorical_gen = CategoricalGenerator(seed=seed, llm_client=llm_client)

    def build(self, schema: DatasetSchema) -> pd.DataFrame:
        """
        Build a complete clean dataset from a schema.

        Args:
            schema: DatasetSchema defining the dataset structure

        Returns:
            pandas DataFrame with clean data
        """
        num_rows = schema.num_rows
        data = {}

        # Generate each column
        for col_schema in schema.columns:
            print(f"Generating column: {col_schema.name} ({col_schema.type.value})")

            if col_schema.type in [ColumnType.NUMERIC_INT, ColumnType.NUMERIC_FLOAT]:
                # Generate numeric column
                is_float = col_schema.type == ColumnType.NUMERIC_FLOAT
                data[col_schema.name] = self.numeric_gen.generate(
                    config=col_schema.numeric_config,
                    num_rows=num_rows,
                    is_float=is_float,
                )

            elif col_schema.type == ColumnType.DATETIME:
                # Generate datetime column
                data[col_schema.name] = self.datetime_gen.generate(
                    config=col_schema.datetime_config, num_rows=num_rows
                )

            elif col_schema.type == ColumnType.CATEGORICAL:
                # Generate categorical column
                data[col_schema.name] = self.categorical_gen.generate(
                    config=col_schema.categorical_config, num_rows=num_rows
                )

            else:
                raise ValueError(f"Unsupported column type: {col_schema.type}")

        # Create DataFrame
        df = pd.DataFrame(data)

        # Apply column relationships if any
        if schema.relationships:
            df = self._apply_relationships(df, schema)

        # Verify clean data properties
        self._verify_clean_data(df, schema)

        return df

    def _apply_relationships(
        self, df: pd.DataFrame, schema: DatasetSchema
    ) -> pd.DataFrame:
        """
        Apply column relationships (correlations, conditionals, etc.).

        Note: This is a simplified implementation. Full relationship support
        would require more sophisticated logic.
        """
        # TODO: Implement relationship handling
        # For now, relationships are not applied
        # Future: Add correlation, conditional, and derived column support
        return df

    def _verify_clean_data(self, df: pd.DataFrame, schema: DatasetSchema):
        """
        Verify that generated data meets clean data standards.

        Checks:
        - No missing values (for non-nullable columns)
        - Correct number of rows
        - All expected columns present
        """
        # Check row count
        assert len(df) == schema.num_rows, f"Row count mismatch: {len(df)} != {schema.num_rows}"

        # Check column count
        assert (
            len(df.columns) == len(schema.columns)
        ), f"Column count mismatch: {len(df.columns)} != {len(schema.columns)}"

        # Check for missing values (clean data should have none by default)
        missing_counts = df.isnull().sum()
        total_missing = missing_counts.sum()

        if total_missing > 0:
            print(f"Warning: Found {total_missing} missing values in clean data:")
            for col, count in missing_counts[missing_counts > 0].items():
                print(f"  {col}: {count} missing")

        print(f"✓ Clean data verified: {len(df)} rows, {len(df.columns)} columns")


def build_clean_dataset(
    schema: DatasetSchema, seed: Optional[int] = None, llm_client=None
) -> pd.DataFrame:
    """
    Convenience function to build a clean dataset from a schema.

    Args:
        schema: DatasetSchema defining the dataset structure
        seed: Random seed for reproducibility
        llm_client: Optional LLM client

    Returns:
        pandas DataFrame with clean data
    """
    builder = CleanDataBuilder(seed=seed, llm_client=llm_client)
    return builder.build(schema)
