"""
Solvability checker for data cleaning tasks.

Verifies that all corruptions in the dirty dataset can be fixed using the
cleaning agent's available tools, ensuring the dataset is actually solvable.
"""

from typing import Any, Dict, List, Tuple

import pandas as pd
import numpy as np


class SolvabilityChecker:
    """Validates that corruptions are solvable by the cleaning agent."""

    # Agent's available operations (from data-cleaner)
    AVAILABLE_OPERATIONS = {
        "handle_missing_values",
        "remove_duplicates",
        "remove_outliers",
        "clean_text_columns",
        "convert_data_types",
        "inspect_data",
    }

    def __init__(self):
        """Initialize solvability checker."""
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def validate_solvability(
        self,
        df_dirty: pd.DataFrame,
        metadata: Dict[str, Any],
    ) -> Tuple[bool, List[str], List[str]]:
        """
        Validate that all corruptions are solvable by the agent.

        Args:
            df_dirty: Dirty dataset
            metadata: Corruption metadata

        Returns:
            Tuple of (is_solvable, errors, warnings)
        """
        self.errors = []
        self.warnings = []

        # Get cleaning operations from metadata
        cleaning_ops = metadata.get("cleaning_operations", [])

        # Run solvability checks
        self._check_operations_available(cleaning_ops)
        self._check_missing_values_solvable(df_dirty, cleaning_ops)
        self._check_text_issues_solvable(df_dirty, cleaning_ops)
        self._check_type_conversions_solvable(df_dirty, cleaning_ops)
        self._check_outliers_detectable(df_dirty, cleaning_ops)
        self._check_duplicates_exact(df_dirty, cleaning_ops)

        is_solvable = len(self.errors) == 0

        return is_solvable, self.errors, self.warnings

    def _check_operations_available(self, cleaning_ops: List[Dict[str, Any]]):
        """Verify that all required operations are available to the agent."""
        for op in cleaning_ops:
            operation_name = op.get("operation", "")

            if operation_name not in self.AVAILABLE_OPERATIONS:
                self.errors.append(
                    f"Operation '{operation_name}' is not available to the cleaning agent. "
                    f"Available operations: {self.AVAILABLE_OPERATIONS}"
                )

    def _check_missing_values_solvable(
        self, df_dirty: pd.DataFrame, cleaning_ops: List[Dict[str, Any]]
    ):
        """
        Verify that missing values can be imputed with available strategies.

        The agent can handle:
        - Numeric: mean, median, mode, interpolate, zero, knn
        - Categorical: most_frequent, unknown, empty_string, missing, forward_fill, back_fill, constant
        """
        # Check if there are missing values
        missing_count = df_dirty.isna().sum().sum()

        if missing_count == 0:
            return  # No missing values, nothing to check

        # Verify handle_missing_values operation is present
        has_missing_handler = any(
            op["operation"] == "handle_missing_values" for op in cleaning_ops
        )

        if not has_missing_handler:
            self.errors.append(
                f"Dataset has {missing_count} missing values but no handle_missing_values operation planned"
            )
            return

        # All standard imputation strategies are solvable by the agent
        # This is a sanity check - all missing value corruptions should be solvable
        self.warnings.append(
            f"Note: {missing_count} missing values will be imputed using handle_missing_values"
        )

    def _check_text_issues_solvable(
        self, df_dirty: pd.DataFrame, cleaning_ops: List[Dict[str, Any]]
    ):
        """
        Verify that text issues can be fixed with clean_text_columns.

        The agent can handle:
        - Whitespace: strip
        - Case: lower, upper
        - Special characters: remove_special
        - Numbers in text: remove_numbers

        The agent CANNOT handle:
        - Typos/misspellings (no fuzzy matching)
        - Complex text transformations
        """
        text_cols = df_dirty.select_dtypes(include=["object"]).columns

        if len(text_cols) == 0:
            return  # No text columns

        # Get clean_text_columns operation if present
        text_cleaning_ops = [
            op for op in cleaning_ops if op["operation"] == "clean_text_columns"
        ]

        if not text_cleaning_ops:
            # No text cleaning planned - check if text columns look dirty
            for col in text_cols:
                sample_values = df_dirty[col].dropna().head(20)

                # Check for obvious whitespace issues
                has_whitespace = sample_values.str.strip() != sample_values
                if has_whitespace.any():
                    self.warnings.append(
                        f"Column '{col}' appears to have whitespace issues but no clean_text_columns planned"
                    )

    def _check_type_conversions_solvable(
        self, df_dirty: pd.DataFrame, cleaning_ops: List[Dict[str, Any]]
    ):
        """
        Verify that type conversions can be done with pd.to_numeric/to_datetime coercion.

        The agent can handle:
        - Converting strings to numeric (with errors='coerce')
        - Converting strings to datetime (with errors='coerce')
        - Simple type conversions

        The agent CANNOT handle:
        - Complex date format parsing beyond pandas defaults
        - Custom parsing logic
        """
        # Get convert_data_types operation if present
        type_conversion_ops = [
            op for op in cleaning_ops if op["operation"] == "convert_data_types"
        ]

        if not type_conversion_ops:
            return  # No type conversion planned

        # Check for columns that look like they should be numeric/datetime but are strings
        for col in df_dirty.columns:
            if df_dirty[col].dtype == "object":
                sample_values = df_dirty[col].dropna().head(50)

                if len(sample_values) == 0:
                    continue

                # Try to convert to numeric
                try:
                    numeric_result = pd.to_numeric(sample_values, errors="coerce")
                    success_rate = numeric_result.notna().sum() / len(sample_values)

                    if success_rate < 0.5:
                        # Less than 50% converted successfully
                        self.warnings.append(
                            f"Column '{col}': Only {success_rate*100:.1f}% of values can be converted to numeric. "
                            "Some values may become NaN."
                        )
                except:
                    pass

                # Try to convert to datetime
                try:
                    datetime_result = pd.to_datetime(sample_values, errors="coerce")
                    success_rate = datetime_result.notna().sum() / len(sample_values)

                    if 0.5 < success_rate < 1.0:
                        self.warnings.append(
                            f"Column '{col}': {success_rate*100:.1f}% of values can be converted to datetime. "
                            "Some values may become NaN."
                        )
                except:
                    pass

    def _check_outliers_detectable(
        self, df_dirty: pd.DataFrame, cleaning_ops: List[Dict[str, Any]]
    ):
        """
        Verify that outliers are detectable with IQR or z-score methods.

        The agent can handle:
        - IQR method with configurable threshold
        - Z-score method with configurable threshold

        Outliers must be extreme enough to be detected with reasonable thresholds
        (typically IQR threshold ≤ 3.0 or z-score ≤ 3.0)
        """
        # Get remove_outliers operation if present
        outlier_ops = [
            op for op in cleaning_ops if op["operation"] == "remove_outliers"
        ]

        if not outlier_ops:
            return  # No outlier removal planned

        # Get the outlier removal parameters
        outlier_op = outlier_ops[0]
        method = outlier_op.get("method", "iqr")
        threshold = outlier_op.get("threshold", 1.5)

        # Check if threshold is reasonable
        if method == "iqr" and threshold > 3.0:
            self.warnings.append(
                f"IQR threshold ({threshold}) is very high. "
                "Standard thresholds are 1.5 (aggressive) to 3.0 (conservative)."
            )
        elif method == "zscore" and threshold > 3.0:
            self.warnings.append(
                f"Z-score threshold ({threshold}) is very high. "
                "Standard thresholds are 2.0 (aggressive) to 3.0 (conservative)."
            )

        # For each numeric column, verify outliers are detectable
        numeric_cols = df_dirty.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            col_data = df_dirty[col].dropna()

            if len(col_data) < 4:
                continue  # Not enough data

            if method == "iqr":
                q25 = col_data.quantile(0.25)
                q75 = col_data.quantile(0.75)
                iqr = q75 - q25

                if iqr == 0:
                    self.warnings.append(
                        f"Column '{col}': IQR is 0 (no variance), outliers cannot be detected"
                    )

    def _check_duplicates_exact(
        self, df_dirty: pd.DataFrame, cleaning_ops: List[Dict[str, Any]]
    ):
        """
        Verify that duplicates are exact (not near-duplicates).

        The agent can handle:
        - Exact duplicate rows
        - Duplicates on subset of columns

        The agent CANNOT handle:
        - Near-duplicates (fuzzy matching)
        - Similarity-based deduplication
        """
        # Get remove_duplicates operation if present
        duplicate_ops = [
            op for op in cleaning_ops if op["operation"] == "remove_duplicates"
        ]

        if not duplicate_ops:
            # Check if there are duplicates
            dup_count = df_dirty.duplicated().sum()
            if dup_count > 0:
                self.warnings.append(
                    f"Dataset has {dup_count} duplicate rows but no remove_duplicates operation planned"
                )
            return

        # The agent uses pandas drop_duplicates, which only handles exact duplicates
        # This is always solvable, just note it
        dup_count = df_dirty.duplicated().sum()
        if dup_count > 0:
            self.warnings.append(
                f"Note: {dup_count} exact duplicate rows will be removed"
            )


def validate_solvability(
    df_dirty: pd.DataFrame,
    metadata: Dict[str, Any],
    verbose: bool = True,
) -> bool:
    """
    Convenience function to validate solvability.

    Args:
        df_dirty: Dirty dataset
        metadata: Corruption metadata
        verbose: Print validation results

    Returns:
        True if all corruptions are solvable, False otherwise
    """
    checker = SolvabilityChecker()
    is_solvable, errors, warnings = checker.validate_solvability(df_dirty, metadata)

    if verbose:
        print("\n" + "=" * 60)
        print("SOLVABILITY VALIDATION RESULTS")
        print("=" * 60)

        if errors:
            print(f"\n✗ VALIDATION FAILED ({len(errors)} errors)")
            for i, error in enumerate(errors, 1):
                print(f"  {i}. ERROR: {error}")
        else:
            print("\n✓ VALIDATION PASSED (all corruptions are solvable)")

        if warnings:
            print(f"\n⚠ {len(warnings)} warnings:")
            for i, warning in enumerate(warnings, 1):
                print(f"  {i}. {warning}")

        print("=" * 60)

    return is_solvable
