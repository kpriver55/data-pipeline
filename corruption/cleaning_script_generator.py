"""
Cleaning script generator.

Generates cleaning operations from corruption metadata to produce
the "recovered clean" dataset that the agent should actually achieve.
"""

from typing import Any, Dict, List, Tuple

import pandas as pd


class CleaningScriptGenerator:
    """Generate cleaning scripts from corruption metadata."""

    # Operation ordering - some must happen before others
    OPERATION_ORDER = {
        "clean_text_columns": 1,  # First: clean text (so type conversion can work)
        "convert_data_types": 2,  # Second: fix types (after text is clean)
        "handle_missing_values": 3,  # Third: handle missing data (after types are correct)
        "remove_outliers": 4,  # Fourth: remove outliers (needs correct types)
        "remove_duplicates": 5,  # Last: remove duplicates
    }

    def generate_cleaning_plan(
        self, metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Generate an ordered cleaning plan from corruption metadata.

        Args:
            metadata: Corruption metadata from orchestrator

        Returns:
            List of cleaning operations in execution order
        """
        operations = []

        applied_strategies = metadata.get("applied_strategies", [])

        # Map strategies to operations
        for strategy in applied_strategies:
            if strategy == "type_corruption":
                operations.append(self._plan_type_conversion(metadata))

            elif strategy == "missing_values":
                operations.append(self._plan_missing_value_handling(metadata))

            elif strategy in [
                "text_whitespace",
                "text_case",
                "text_special_chars",
            ]:
                # All text operations can be handled together
                if not any(op["operation"] == "clean_text_columns" for op in operations):
                    operations.append(self._plan_text_cleaning(metadata))

            elif strategy == "outliers":
                operations.append(self._plan_outlier_removal(metadata))

            elif strategy == "duplicates":
                operations.append(self._plan_duplicate_removal(metadata))

        # Sort by operation order
        operations.sort(
            key=lambda op: self.OPERATION_ORDER.get(op["operation"], 999)
        )

        return operations

    def _plan_type_conversion(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Plan type conversion operations in agent-compatible format."""
        # Format compatible with data-cleaner agent expectations
        # Columns will be auto-detected during execution
        return {
            "operation": "convert_data_types",
            "rationale": "Convert string columns back to appropriate numeric/datetime types using coercion",
            # Note: columns and type_conversions will be determined at runtime
            # The agent should infer appropriate types from the data
        }

    def _plan_missing_value_handling(
        self, metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Plan missing value handling in agent-compatible format."""
        return {
            "operation": "handle_missing_values",
            "numeric_strategy": "median",  # Standard approach for numeric columns
            "categorical_strategy": "most_frequent",  # Standard for categorical columns
            "rationale": "Impute missing values using median for numeric columns and most frequent value for categorical columns",
        }

    def _plan_text_cleaning(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Plan text cleaning operations in agent-compatible format."""
        # Determine which operations are needed based on applied strategies
        applied = metadata.get("applied_strategies", [])
        operations = []

        if "text_whitespace" in applied:
            operations.append("strip")

        if "text_case" in applied:
            operations.append("lower")  # Normalize to lowercase

        if "text_special_chars" in applied:
            operations.append("remove_special")

        if not operations:
            operations = ["strip", "lower"]

        # Build rationale based on operations
        op_descriptions = {
            "strip": "remove whitespace",
            "lower": "normalize case",
            "remove_special": "remove special characters"
        }
        rationale_parts = [op_descriptions.get(op, op) for op in operations]
        rationale = f"Clean text columns: {', '.join(rationale_parts)}"

        return {
            "operation": "clean_text_columns",
            "operations": operations,  # Flattened from parameters
            "rationale": rationale,
        }

    def _plan_outlier_removal(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Plan outlier removal in agent-compatible format."""
        outlier_severity = metadata.get("corruption_config", {}).get("rates", {}).get("outlier_severity", 2.5)

        return {
            "operation": "remove_outliers",
            "method": "iqr",  # Flattened from parameters
            "threshold": 1.5,  # Standard IQR threshold for detection
            "rationale": f"Remove outliers using IQR method with threshold 1.5 (outliers were created with severity {outlier_severity})",
        }

    def _plan_duplicate_removal(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Plan duplicate removal in agent-compatible format."""
        return {
            "operation": "remove_duplicates",
            "keep": "first",  # Flattened from parameters
            "rationale": "Remove exact duplicate rows, keeping first occurrence",
        }


def auto_clean_dataset(
    df_dirty: pd.DataFrame, metadata: Dict[str, Any]
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Automatically clean a dirty dataset using the corruption metadata.

    This produces the "recovered clean" dataset that represents what
    the agent should actually achieve (not the perfect synthetic data).

    Args:
        df_dirty: Dirty dataset to clean
        metadata: Corruption metadata from orchestrator

    Returns:
        Tuple of (recovered clean dataframe, cleaning operations applied)
    """
    generator = CleaningScriptGenerator()
    cleaning_plan = generator.generate_cleaning_plan(metadata)

    df_clean = df_dirty.copy()
    applied_operations = []

    print(f"\nAuto-cleaning dataset using {len(cleaning_plan)} operations...")

    for i, step in enumerate(cleaning_plan, 1):
        operation = step["operation"]
        rationale = step.get("rationale", "")

        print(f"  {i}. {operation}: {rationale}")

        try:
            # Pass the entire step dict to helper functions for access to all parameters
            if operation == "convert_data_types":
                df_clean = _apply_type_conversion(df_clean, step)

            elif operation == "handle_missing_values":
                df_clean = _apply_missing_value_handling(df_clean, step)

            elif operation == "clean_text_columns":
                df_clean = _apply_text_cleaning(df_clean, step)

            elif operation == "remove_outliers":
                df_clean = _apply_outlier_removal(df_clean, step)

            elif operation == "remove_duplicates":
                df_clean = _apply_duplicate_removal(df_clean, step)

            applied_operations.append(step)

        except Exception as e:
            print(f"    Warning: Failed to apply {operation}: {e}")
            step["error"] = str(e)
            applied_operations.append(step)

    print(f"✓ Auto-cleaning complete: {len(df_dirty)} → {len(df_clean)} rows")

    return df_clean, applied_operations


# Helper functions to apply each operation


def _apply_type_conversion(
    df: pd.DataFrame, operation: Dict[str, Any]
) -> pd.DataFrame:
    """Apply type conversion to columns that were corrupted."""
    df_clean = df.copy()

    # Auto-detect columns that should be numeric or datetime
    # Note: In the future, could use operation.get("type_conversions", {})
    # for explicit column->type mappings
    for col in df_clean.columns:
        if df_clean[col].dtype == "object":
            # Try to convert to numeric
            try:
                df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")
            except:
                pass

            # Try to convert to datetime
            if df_clean[col].dtype == "object":
                try:
                    df_clean[col] = pd.to_datetime(df_clean[col], errors="coerce")
                except:
                    pass

    return df_clean


def _apply_missing_value_handling(
    df: pd.DataFrame, operation: Dict[str, Any]
) -> pd.DataFrame:
    """Apply missing value imputation."""
    import numpy as np

    df_clean = df.copy()

    numeric_strategy = operation.get("numeric_strategy", "median")
    categorical_strategy = operation.get("categorical_strategy", "most_frequent")

    # Handle numeric columns
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df_clean[col].isnull().any():
            if numeric_strategy == "median":
                fill_value = df_clean[col].median()
            elif numeric_strategy == "mean":
                fill_value = df_clean[col].mean()
            else:
                fill_value = df_clean[col].median()

            df_clean[col] = df_clean[col].fillna(fill_value)

    # Handle categorical columns
    categorical_cols = df_clean.select_dtypes(include=["object"]).columns
    for col in categorical_cols:
        if df_clean[col].isnull().any():
            if categorical_strategy == "most_frequent":
                fill_value = df_clean[col].mode().iloc[0] if len(df_clean[col].mode()) > 0 else ""
            else:
                fill_value = "Unknown"

            df_clean[col] = df_clean[col].fillna(fill_value)

    return df_clean


def _apply_text_cleaning(df: pd.DataFrame, operation: Dict[str, Any]) -> pd.DataFrame:
    """Apply text cleaning operations."""
    df_clean = df.copy()
    operations = operation.get("operations", ["strip", "lower"])

    text_cols = df_clean.select_dtypes(include=["object"]).columns

    for col in text_cols:
        if "strip" in operations:
            df_clean[col] = df_clean[col].astype(str).str.strip()

        if "lower" in operations:
            df_clean[col] = df_clean[col].astype(str).str.lower()

        if "upper" in operations:
            df_clean[col] = df_clean[col].astype(str).str.upper()

        if "remove_special" in operations:
            df_clean[col] = (
                df_clean[col].astype(str).str.replace(r"[^a-zA-Z0-9\s]", "", regex=True)
            )

        if "remove_numbers" in operations:
            df_clean[col] = df_clean[col].astype(str).str.replace(r"\d+", "", regex=True)

    return df_clean


def _apply_outlier_removal(df: pd.DataFrame, operation: Dict[str, Any]) -> pd.DataFrame:
    """Apply outlier removal."""
    import numpy as np

    df_clean = df.copy()
    method = operation.get("method", "iqr")
    threshold = operation.get("threshold", 1.5)

    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        if method == "iqr":
            q25 = df_clean[col].quantile(0.25)
            q75 = df_clean[col].quantile(0.75)
            iqr = q75 - q25
            lower_bound = q25 - threshold * iqr
            upper_bound = q75 + threshold * iqr

            outlier_mask = (df_clean[col] < lower_bound) | (
                df_clean[col] > upper_bound
            )
            df_clean = df_clean[~outlier_mask]

    return df_clean


def _apply_duplicate_removal(
    df: pd.DataFrame, operation: Dict[str, Any]
) -> pd.DataFrame:
    """Apply duplicate removal."""
    keep = operation.get("keep", "first")
    subset = operation.get("subset", None)  # Optional: specific columns to check
    return df.drop_duplicates(subset=subset, keep=keep)
