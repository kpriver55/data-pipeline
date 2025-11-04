"""
Consistency checker for raw→recovered clean dataset pairs.

Verifies that the recovered clean dataset is logically derivable from
the raw dirty dataset, ensuring data integrity is maintained during cleaning.
"""

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


class ConsistencyChecker:
    """Validates consistency between dirty and recovered clean datasets."""

    def __init__(self):
        """Initialize consistency checker."""
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def validate_pair(
        self,
        df_dirty: pd.DataFrame,
        df_recovered_clean: pd.DataFrame,
        metadata: Dict[str, Any],
    ) -> Tuple[bool, List[str], List[str]]:
        """
        Validate consistency between dirty and recovered clean datasets.

        Args:
            df_dirty: Dirty dataset
            df_recovered_clean: Recovered clean dataset
            metadata: Corruption/recovery metadata

        Returns:
            Tuple of (is_valid, errors, warnings)
        """
        self.errors = []
        self.warnings = []

        # Run all validation checks
        self._check_column_consistency(df_dirty, df_recovered_clean)
        self._check_row_count_consistency(
            df_dirty, df_recovered_clean, metadata
        )
        self._check_data_derivability(df_dirty, df_recovered_clean, metadata)
        self._check_no_information_added(df_dirty, df_recovered_clean)

        is_valid = len(self.errors) == 0

        return is_valid, self.errors, self.warnings

    def _check_column_consistency(
        self, df_dirty: pd.DataFrame, df_recovered: pd.DataFrame
    ):
        """Verify that columns match between dirty and recovered datasets."""
        dirty_cols = set(df_dirty.columns)
        recovered_cols = set(df_recovered.columns)

        if dirty_cols != recovered_cols:
            missing_in_recovered = dirty_cols - recovered_cols
            extra_in_recovered = recovered_cols - dirty_cols

            if missing_in_recovered:
                self.errors.append(
                    f"Columns missing in recovered clean: {missing_in_recovered}"
                )
            if extra_in_recovered:
                self.errors.append(
                    f"Extra columns in recovered clean: {extra_in_recovered}"
                )

    def _check_row_count_consistency(
        self,
        df_dirty: pd.DataFrame,
        df_recovered: pd.DataFrame,
        metadata: Dict[str, Any],
    ):
        """
        Verify that row count changes are expected based on operations.

        Row count can decrease due to:
        - Duplicate removal
        - Outlier removal

        Row count should NOT increase (no data generation allowed).
        """
        dirty_rows = len(df_dirty)
        recovered_rows = len(df_recovered)

        if recovered_rows > dirty_rows:
            self.errors.append(
                f"Recovered dataset has MORE rows than dirty ({recovered_rows} > {dirty_rows}). "
                "Cleaning should not generate new rows."
            )
            return

        if recovered_rows < dirty_rows:
            # Check if this is expected
            applied_ops = [
                op["operation"] for op in metadata.get("cleaning_operations", [])
            ]

            row_reducing_ops = {"remove_duplicates", "remove_outliers"}
            has_row_reducing_ops = any(
                op in row_reducing_ops for op in applied_ops
            )

            if not has_row_reducing_ops:
                self.warnings.append(
                    f"Recovered dataset has fewer rows ({recovered_rows} vs {dirty_rows}) "
                    f"but no row-reducing operations were applied. Applied ops: {applied_ops}"
                )
            else:
                # This is expected - just a note
                rows_removed = dirty_rows - recovered_rows
                self.warnings.append(
                    f"Note: {rows_removed} rows removed during cleaning (expected due to {row_reducing_ops & set(applied_ops)})"
                )

    def _check_data_derivability(
        self,
        df_dirty: pd.DataFrame,
        df_recovered: pd.DataFrame,
        metadata: Dict[str, Any],
    ):
        """
        Verify that recovered data is derivable from dirty data.

        This is a heuristic check since perfect derivability is complex.
        We check:
        - Type conversions are valid
        - Value ranges are preserved
        - Imputed values are reasonable
        """
        # For each column, check if the transformation is reasonable
        for col in df_recovered.columns:
            if col not in df_dirty.columns:
                continue  # Already caught by column consistency check

            dirty_col = df_dirty[col]
            recovered_col = df_recovered[col]

            # Check type compatibility
            self._check_type_derivability(col, dirty_col, recovered_col, metadata)

            # Check value derivability (are values in reasonable range?)
            self._check_value_derivability(col, dirty_col, recovered_col, metadata)

    def _check_type_derivability(
        self,
        col: str,
        dirty_col: pd.Series,
        recovered_col: pd.Series,
        metadata: Dict[str, Any],
    ):
        """Check if type conversion is valid."""
        # If types differ, ensure there was a convert_data_types operation
        if dirty_col.dtype != recovered_col.dtype:
            cleaning_ops = [
                op["operation"] for op in metadata.get("cleaning_operations", [])
            ]

            if "convert_data_types" not in cleaning_ops:
                self.warnings.append(
                    f"Column '{col}' changed type ({dirty_col.dtype} → {recovered_col.dtype}) "
                    "without convert_data_types operation"
                )

    def _check_value_derivability(
        self,
        col: str,
        dirty_col: pd.Series,
        recovered_col: pd.Series,
        metadata: Dict[str, Any],
    ):
        """
        Check if recovered values could have been derived from dirty values.

        For numeric columns: check if imputed values are within reasonable range
        For categorical: check if imputed values existed in dirty data
        """
        import numpy as np

        # Skip if all values are missing
        if recovered_col.isna().all():
            return

        # For numeric columns (check RECOVERED column type, not dirty)
        if pd.api.types.is_numeric_dtype(recovered_col):
            # Get non-missing values from both
            dirty_values = dirty_col.dropna()
            recovered_values = recovered_col.dropna()

            if len(dirty_values) == 0:
                return  # Can't validate if dirty had no non-missing values

            # If dirty column is also numeric, check range
            if pd.api.types.is_numeric_dtype(dirty_col):
                # Check if recovered values are within reasonable range of dirty values
                dirty_min = dirty_values.min()
                dirty_max = dirty_values.max()

                # Allow some tolerance for imputed values
                tolerance_factor = 1.5  # Imputed values should be within 1.5x the original range
                expanded_min = dirty_min * tolerance_factor if dirty_min < 0 else dirty_min / tolerance_factor
                expanded_max = dirty_max * tolerance_factor
            else:
                # Dirty column is not numeric (was converted from string)
                # Try to convert dirty values to numeric to check range
                try:
                    dirty_numeric = pd.to_numeric(dirty_values, errors="coerce")
                    dirty_numeric = dirty_numeric.dropna()

                    if len(dirty_numeric) == 0:
                        return  # No valid numeric values in dirty column

                    dirty_min = dirty_numeric.min()
                    dirty_max = dirty_numeric.max()

                    tolerance_factor = 1.5
                    expanded_min = dirty_min * tolerance_factor if dirty_min < 0 else dirty_min / tolerance_factor
                    expanded_max = dirty_max * tolerance_factor
                except:
                    return  # Can't validate, skip

            out_of_range = (recovered_values < expanded_min) | (recovered_values > expanded_max)

            if out_of_range.any():
                self.warnings.append(
                    f"Column '{col}': {out_of_range.sum()} values in recovered dataset "
                    f"are outside reasonable range [{expanded_min:.2f}, {expanded_max:.2f}]"
                )

        # For categorical/text columns
        elif pd.api.types.is_object_dtype(recovered_col) or pd.api.types.is_categorical_dtype(recovered_col):
            dirty_unique = set(dirty_col.dropna().unique())
            recovered_unique = set(recovered_col.dropna().unique())

            # Values in recovered should mostly be from dirty (except for imputed values or text cleaning)
            new_values = recovered_unique - dirty_unique

            if len(new_values) > 0:
                # Check if there was missing value imputation or text cleaning
                cleaning_ops = [
                    op["operation"] for op in metadata.get("cleaning_operations", [])
                ]

                if "handle_missing_values" in cleaning_ops:
                    # New values are expected (imputed)
                    self.warnings.append(
                        f"Column '{col}': {len(new_values)} new values appeared "
                        f"(likely imputed): {new_values if len(new_values) <= 5 else list(new_values)[:5]}"
                    )
                elif "clean_text_columns" in cleaning_ops:
                    # New values are expected from text cleaning (whitespace removal, case normalization, etc.)
                    self.warnings.append(
                        f"Column '{col}': {len(new_values)} values changed due to text cleaning "
                        f"(expected): {new_values if len(new_values) <= 5 else list(new_values)[:5]}"
                    )
                else:
                    # New values without expected operations - suspicious
                    self.errors.append(
                        f"Column '{col}': New values appeared without handle_missing_values or clean_text_columns: {new_values}"
                    )

    def _check_no_information_added(
        self, df_dirty: pd.DataFrame, df_recovered: pd.DataFrame
    ):
        """
        Verify that no external information was added during cleaning.

        This is a soft check - we verify that:
        - No new columns were created
        - No completely new data appeared (beyond imputation)
        """
        # Column check already done in _check_column_consistency

        # Check for suspicious patterns (this is heuristic)
        for col in df_recovered.columns:
            if col not in df_dirty.columns:
                continue

            dirty_missing = df_dirty[col].isna().sum()
            recovered_missing = df_recovered[col].isna().sum()

            # If ALL missing values in dirty became non-missing in recovered,
            # verify there was imputation
            if dirty_missing > 0 and recovered_missing == 0:
                self.warnings.append(
                    f"Column '{col}': All {dirty_missing} missing values were imputed (expected if handle_missing_values was used)"
                )


def validate_dataset_pair(
    df_dirty: pd.DataFrame,
    df_recovered_clean: pd.DataFrame,
    metadata: Dict[str, Any],
    verbose: bool = True,
) -> bool:
    """
    Convenience function to validate a dataset pair.

    Args:
        df_dirty: Dirty dataset
        df_recovered_clean: Recovered clean dataset
        metadata: Corruption/recovery metadata
        verbose: Print validation results

    Returns:
        True if validation passes, False otherwise
    """
    checker = ConsistencyChecker()
    is_valid, errors, warnings = checker.validate_pair(
        df_dirty, df_recovered_clean, metadata
    )

    if verbose:
        print("\n" + "=" * 60)
        print("CONSISTENCY VALIDATION RESULTS")
        print("=" * 60)

        if errors:
            print(f"\n✗ VALIDATION FAILED ({len(errors)} errors)")
            for i, error in enumerate(errors, 1):
                print(f"  {i}. ERROR: {error}")
        else:
            print("\n✓ VALIDATION PASSED (no errors)")

        if warnings:
            print(f"\n⚠ {len(warnings)} warnings:")
            for i, warning in enumerate(warnings, 1):
                print(f"  {i}. {warning}")

        print("=" * 60)

    return is_valid
