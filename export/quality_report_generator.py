"""
Data quality report generator for datasets.

Generates detailed quality reports including per-column metrics,
corruption analysis, cleaning operation details, and recommendations.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime

import pandas as pd
import numpy as np


class QualityReportGenerator:
    """Generate comprehensive data quality reports."""

    @staticmethod
    def generate_quality_report(
        df_dirty: pd.DataFrame,
        df_recovered_clean: pd.DataFrame,
        metadata: Dict[str, Any],
        df_perfect: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        Generate comprehensive quality report.

        Args:
            df_dirty: Dirty dataset
            df_recovered_clean: Recovered clean dataset
            metadata: Pipeline metadata
            df_perfect: Optional perfect synthetic dataset

        Returns:
            Quality report dictionary

        Example:
            >>> report = QualityReportGenerator.generate_quality_report(
            ...     df_dirty, df_clean, metadata
            ... )
            >>> print(report["summary"]["overall_quality_score"])
        """
        report = {
            "generated_at": datetime.now().isoformat(),
            "report_version": "1.0.0",
        }

        # Overall summary
        report["summary"] = QualityReportGenerator._generate_summary(
            df_dirty, df_recovered_clean, metadata
        )

        # Per-column quality metrics
        report["column_quality"] = QualityReportGenerator._generate_column_quality(
            df_dirty, df_recovered_clean
        )

        # Corruption analysis
        report["corruption_analysis"] = (
            QualityReportGenerator._generate_corruption_analysis(
                df_dirty, df_recovered_clean, metadata
            )
        )

        # Cleaning operation details
        report["cleaning_analysis"] = (
            QualityReportGenerator._generate_cleaning_analysis(metadata)
        )

        # Data loss analysis
        report["data_loss"] = QualityReportGenerator._generate_data_loss_analysis(
            df_dirty, df_recovered_clean, metadata
        )

        # Recommendations
        report["recommendations"] = QualityReportGenerator._generate_recommendations(
            df_dirty, df_recovered_clean, metadata
        )

        # Perfect vs recovered comparison (if available)
        if df_perfect is not None:
            report["perfect_comparison"] = (
                QualityReportGenerator._compare_to_perfect(
                    df_perfect, df_recovered_clean
                )
            )

        return report

    @staticmethod
    def _generate_summary(
        df_dirty: pd.DataFrame,
        df_recovered_clean: pd.DataFrame,
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate overall summary metrics."""
        # Calculate quality score (0-100)
        missing_improvement = (
            df_dirty.isna().sum().sum() - df_recovered_clean.isna().sum().sum()
        ) / max(df_dirty.isna().sum().sum(), 1)

        duplicate_improvement = (
            df_dirty.duplicated().sum() - df_recovered_clean.duplicated().sum()
        ) / max(df_dirty.duplicated().sum(), 1)

        # Row retention rate
        row_retention = len(df_recovered_clean) / len(df_dirty)

        # Overall quality score (weighted average)
        quality_score = (
            missing_improvement * 40  # 40% weight on missing value handling
            + duplicate_improvement * 30  # 30% weight on duplicate removal
            + row_retention * 30  # 30% weight on data retention
        ) * 100

        recovery_summary = metadata.get("recovery_summary", {})

        return {
            "overall_quality_score": round(quality_score, 2),
            "total_rows_dirty": len(df_dirty),
            "total_rows_recovered": len(df_recovered_clean),
            "row_retention_rate": round(row_retention * 100, 2),
            "rows_lost": len(df_dirty) - len(df_recovered_clean),
            "missing_values_dirty": int(df_dirty.isna().sum().sum()),
            "missing_values_recovered": int(df_recovered_clean.isna().sum().sum()),
            "missing_reduction_rate": round(missing_improvement * 100, 2),
            "duplicates_dirty": int(df_dirty.duplicated().sum()),
            "duplicates_recovered": int(df_recovered_clean.duplicated().sum()),
            "duplicate_reduction_rate": round(duplicate_improvement * 100, 2),
            "strategies_applied": len(metadata.get("applied_strategies", [])),
            "cleaning_operations": len(metadata.get("cleaning_operations", [])),
            "difficulty": metadata.get("corruption_config", {}).get(
                "difficulty", "unknown"
            ),
        }

    @staticmethod
    def _generate_column_quality(
        df_dirty: pd.DataFrame, df_recovered_clean: pd.DataFrame
    ) -> Dict[str, Dict[str, Any]]:
        """Generate per-column quality metrics."""
        column_quality = {}

        for col in df_dirty.columns:
            if col not in df_recovered_clean.columns:
                continue

            dirty_col = df_dirty[col]
            clean_col = df_recovered_clean[col]

            # Basic metrics
            metrics = {
                "type": str(dirty_col.dtype),
                "dirty": {
                    "missing": int(dirty_col.isna().sum()),
                    "missing_pct": round((dirty_col.isna().sum() / len(dirty_col)) * 100, 2),
                    "unique": int(dirty_col.nunique()),
                    "unique_pct": round((dirty_col.nunique() / len(dirty_col)) * 100, 2),
                },
                "recovered": {
                    "missing": int(clean_col.isna().sum()),
                    "missing_pct": round((clean_col.isna().sum() / len(clean_col)) * 100, 2),
                    "unique": int(clean_col.nunique()),
                    "unique_pct": round((clean_col.nunique() / len(clean_col)) * 100, 2),
                },
            }

            # Type-specific metrics
            if pd.api.types.is_numeric_dtype(clean_col):
                metrics["numeric_stats"] = QualityReportGenerator._get_numeric_stats(
                    dirty_col, clean_col
                )
            elif pd.api.types.is_string_dtype(dirty_col) or dirty_col.dtype == "object":
                metrics["text_stats"] = QualityReportGenerator._get_text_stats(
                    dirty_col, clean_col
                )

            # Quality improvement score for this column
            missing_improvement = (
                dirty_col.isna().sum() - clean_col.isna().sum()
            ) / max(dirty_col.isna().sum(), 1)
            metrics["improvement_score"] = round(missing_improvement * 100, 2)

            column_quality[col] = metrics

        return column_quality

    @staticmethod
    def _get_numeric_stats(
        dirty_col: pd.Series, clean_col: pd.Series
    ) -> Dict[str, Any]:
        """Get numeric column statistics."""
        # Convert to numeric, coercing errors to NaN
        dirty_numeric = pd.to_numeric(dirty_col, errors='coerce')
        clean_numeric = pd.to_numeric(clean_col, errors='coerce')

        dirty_valid = dirty_numeric.dropna()
        clean_valid = clean_numeric.dropna()

        return {
            "dirty": {
                "mean": round(float(dirty_valid.mean()), 2) if len(dirty_valid) > 0 else None,
                "median": round(float(dirty_valid.median()), 2) if len(dirty_valid) > 0 else None,
                "std": round(float(dirty_valid.std()), 2) if len(dirty_valid) > 0 else None,
                "min": round(float(dirty_valid.min()), 2) if len(dirty_valid) > 0 else None,
                "max": round(float(dirty_valid.max()), 2) if len(dirty_valid) > 0 else None,
            },
            "recovered": {
                "mean": round(float(clean_valid.mean()), 2) if len(clean_valid) > 0 else None,
                "median": round(float(clean_valid.median()), 2) if len(clean_valid) > 0 else None,
                "std": round(float(clean_valid.std()), 2) if len(clean_valid) > 0 else None,
                "min": round(float(clean_valid.min()), 2) if len(clean_valid) > 0 else None,
                "max": round(float(clean_valid.max()), 2) if len(clean_valid) > 0 else None,
            },
        }

    @staticmethod
    def _get_text_stats(dirty_col: pd.Series, clean_col: pd.Series) -> Dict[str, Any]:
        """Get text column statistics."""
        dirty_valid = dirty_col.dropna().astype(str)
        clean_valid = clean_col.dropna().astype(str)

        return {
            "dirty": {
                "avg_length": round(dirty_valid.str.len().mean(), 2) if len(dirty_valid) > 0 else None,
                "min_length": int(dirty_valid.str.len().min()) if len(dirty_valid) > 0 else None,
                "max_length": int(dirty_valid.str.len().max()) if len(dirty_valid) > 0 else None,
                "has_whitespace_issues": int((dirty_valid != dirty_valid.str.strip()).sum()),
                "has_case_variations": int(
                    (dirty_valid != dirty_valid.str.lower()).sum()
                ),
            },
            "recovered": {
                "avg_length": round(clean_valid.str.len().mean(), 2) if len(clean_valid) > 0 else None,
                "min_length": int(clean_valid.str.len().min()) if len(clean_valid) > 0 else None,
                "max_length": int(clean_valid.str.len().max()) if len(clean_valid) > 0 else None,
                "has_whitespace_issues": int((clean_valid != clean_valid.str.strip()).sum()),
                "has_case_variations": int(
                    (clean_valid != clean_valid.str.lower()).sum()
                ),
            },
        }

    @staticmethod
    def _generate_corruption_analysis(
        df_dirty: pd.DataFrame,
        df_recovered_clean: pd.DataFrame,
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Analyze corruption details."""
        strategies = metadata.get("applied_strategies", [])
        corruption_config = metadata.get("corruption_config", {})
        rates = corruption_config.get("rates", {})

        # Count actual corruptions by type
        corruption_counts = {}
        for strategy in strategies:
            # Estimate based on strategy and rates
            if "missing" in strategy.lower():
                corruption_counts[strategy] = int(df_dirty.isna().sum().sum())
            elif "duplicate" in strategy.lower():
                corruption_counts[strategy] = int(df_dirty.duplicated().sum())
            elif "whitespace" in strategy.lower() or "case" in strategy.lower():
                # Estimate text corruptions
                text_cols = df_dirty.select_dtypes(include=["object"]).columns
                corruption_counts[strategy] = len(text_cols) * int(
                    len(df_dirty) * rates.get("whitespace_rate", 0.1)
                )
            elif "outlier" in strategy.lower():
                # Estimate outliers in numeric columns
                numeric_cols = df_dirty.select_dtypes(include=[np.number]).columns
                corruption_counts[strategy] = len(numeric_cols) * int(
                    len(df_dirty) * rates.get("outlier_rate", 0.05)
                )

        return {
            "strategies": strategies,
            "num_strategies": len(strategies),
            "corruption_rates": rates,
            "estimated_corruptions_by_strategy": corruption_counts,
            "total_estimated_corruptions": sum(corruption_counts.values()),
            "difficulty_level": corruption_config.get("difficulty", "unknown"),
        }

    @staticmethod
    def _generate_cleaning_analysis(metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze cleaning operations."""
        cleaning_ops = metadata.get("cleaning_operations", [])

        # Group by operation type
        ops_by_type = {}
        for op in cleaning_ops:
            op_type = op["operation"]
            if op_type not in ops_by_type:
                ops_by_type[op_type] = []
            ops_by_type[op_type].append(op)

        # Calculate stats per operation type
        operation_stats = {}
        for op_type, ops in ops_by_type.items():
            total_changes = sum(
                op.get("changes_made", 0) + op.get("rows_affected", 0) for op in ops
            )
            operation_stats[op_type] = {
                "count": len(ops),
                "total_changes": total_changes,
                "columns_affected": [op.get("column", "N/A") for op in ops],
            }

        return {
            "total_operations": len(cleaning_ops),
            "operation_sequence": [op["operation"] for op in cleaning_ops],
            "operations_by_type": operation_stats,
            "recovery_summary": metadata.get("recovery_summary", {}),
        }

    @staticmethod
    def _generate_data_loss_analysis(
        df_dirty: pd.DataFrame,
        df_recovered_clean: pd.DataFrame,
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Analyze data loss during recovery."""
        recovery_summary = metadata.get("recovery_summary", {})

        rows_lost = len(df_dirty) - len(df_recovered_clean)
        row_loss_pct = (rows_lost / len(df_dirty)) * 100 if len(df_dirty) > 0 else 0

        # Estimate reasons for data loss
        loss_reasons = []
        if df_dirty.duplicated().sum() > 0:
            loss_reasons.append({
                "reason": "Duplicate removal",
                "estimated_rows": int(df_dirty.duplicated().sum()),
            })

        # Check for outlier removal in cleaning ops
        cleaning_ops = metadata.get("cleaning_operations", [])
        outlier_ops = [op for op in cleaning_ops if "outlier" in op.get("operation", "").lower()]
        if outlier_ops:
            outlier_rows = sum(op.get("rows_affected", 0) for op in outlier_ops)
            loss_reasons.append({
                "reason": "Outlier removal",
                "estimated_rows": outlier_rows,
            })

        return {
            "rows_lost": rows_lost,
            "row_loss_percentage": round(row_loss_pct, 2),
            "rows_retained": len(df_recovered_clean),
            "retention_rate": round((len(df_recovered_clean) / len(df_dirty)) * 100, 2),
            "loss_breakdown": loss_reasons,
            "is_acceptable": row_loss_pct < 20,  # Flag if >20% data loss
        }

    @staticmethod
    def _generate_recommendations(
        df_dirty: pd.DataFrame,
        df_recovered_clean: pd.DataFrame,
        metadata: Dict[str, Any],
    ) -> List[Dict[str, str]]:
        """Generate recommendations based on quality analysis."""
        recommendations = []

        # Check data loss
        rows_lost = len(df_dirty) - len(df_recovered_clean)
        loss_pct = (rows_lost / len(df_dirty)) * 100 if len(df_dirty) > 0 else 0

        if loss_pct > 20:
            recommendations.append({
                "severity": "warning",
                "category": "data_loss",
                "message": f"High data loss ({loss_pct:.1f}%). Consider reviewing duplicate and outlier removal strategies.",
            })
        elif loss_pct > 10:
            recommendations.append({
                "severity": "info",
                "category": "data_loss",
                "message": f"Moderate data loss ({loss_pct:.1f}%). This may be acceptable depending on use case.",
            })

        # Check missing values
        missing_pct = (df_recovered_clean.isna().sum().sum() / df_recovered_clean.size) * 100
        if missing_pct > 10:
            recommendations.append({
                "severity": "warning",
                "category": "missing_values",
                "message": f"Recovered dataset still has {missing_pct:.1f}% missing values. Consider additional imputation strategies.",
            })

        # Check duplicates
        duplicates = df_recovered_clean.duplicated().sum()
        if duplicates > 0:
            recommendations.append({
                "severity": "info",
                "category": "duplicates",
                "message": f"{duplicates} duplicate rows remain. Verify if these are intentional.",
            })

        # Check column types
        for col in df_recovered_clean.columns:
            if df_recovered_clean[col].dtype == "object":
                recommendations.append({
                    "severity": "info",
                    "category": "data_types",
                    "message": f"Column '{col}' is object type. Verify if type conversion is needed.",
                })

        # If no issues found
        if not recommendations:
            recommendations.append({
                "severity": "success",
                "category": "overall",
                "message": "Data quality looks good! No major issues detected.",
            })

        return recommendations

    @staticmethod
    def _compare_to_perfect(
        df_perfect: pd.DataFrame, df_recovered_clean: pd.DataFrame
    ) -> Dict[str, Any]:
        """Compare recovered clean to perfect synthetic data."""
        # Shape comparison
        shape_match = df_perfect.shape == df_recovered_clean.shape

        # Column comparison
        columns_match = set(df_perfect.columns) == set(df_recovered_clean.columns)

        # Data similarity (if shapes match)
        similarity = None
        if shape_match and columns_match:
            # Calculate percentage of matching values
            matching_cells = 0
            total_cells = df_perfect.size

            for col in df_perfect.columns:
                matching_cells += (df_perfect[col] == df_recovered_clean[col]).sum()

            similarity = (matching_cells / total_cells) * 100

        return {
            "shape_match": shape_match,
            "perfect_shape": df_perfect.shape,
            "recovered_shape": df_recovered_clean.shape,
            "columns_match": columns_match,
            "missing_columns": list(set(df_perfect.columns) - set(df_recovered_clean.columns)),
            "extra_columns": list(set(df_recovered_clean.columns) - set(df_perfect.columns)),
            "data_similarity_pct": round(similarity, 2) if similarity is not None else None,
            "note": "Perfect synthetic data is unrealistically clean - recovered clean is the actual achievable target",
        }


# =============================================================================
# Convenience function
# =============================================================================


def generate_quality_report(
    df_dirty: pd.DataFrame,
    df_recovered_clean: pd.DataFrame,
    metadata: Dict[str, Any],
    df_perfect: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Convenience function to generate quality report.

    Args:
        df_dirty: Dirty dataset
        df_recovered_clean: Recovered clean dataset
        metadata: Pipeline metadata
        df_perfect: Optional perfect synthetic dataset

    Returns:
        Quality report dictionary

    Example:
        >>> report = generate_quality_report(df_dirty, df_clean, metadata)
        >>> print(f"Quality score: {report['summary']['overall_quality_score']}")
    """
    return QualityReportGenerator.generate_quality_report(
        df_dirty, df_recovered_clean, metadata, df_perfect
    )
