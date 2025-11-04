"""
Comprehensive metadata generator for datasets.

Generates detailed metadata including schema info, corruption details,
cleaning operations, statistics, and usage instructions.
"""

from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd


class MetadataGenerator:
    """Generate comprehensive metadata for datasets."""

    @staticmethod
    def generate_comprehensive_metadata(
        df_dirty: pd.DataFrame,
        df_recovered_clean: pd.DataFrame,
        base_metadata: Dict[str, Any],
        df_perfect: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        Generate comprehensive metadata document.

        Args:
            df_dirty: Dirty dataset
            df_recovered_clean: Recovered clean dataset
            base_metadata: Base metadata from corruption/orchestrator
            df_perfect: Optional perfect synthetic dataset

        Returns:
            Comprehensive metadata dictionary
        """
        metadata = {
            "generated_at": datetime.now().isoformat(),
            "pipeline_version": "1.0.0",
            "description": "Synthetic dataset pair for data cleaning agent training",
        }

        # Add base metadata
        metadata.update(base_metadata)

        # Add dataset information
        metadata["datasets"] = MetadataGenerator._generate_dataset_info(
            df_dirty, df_recovered_clean, df_perfect
        )

        # Add schema information
        if "dataset_config" in base_metadata:
            metadata["schema_info"] = MetadataGenerator._generate_schema_info(
                base_metadata["dataset_config"]
            )

        # Add corruption details
        metadata["corruption_details"] = MetadataGenerator._generate_corruption_details(
            base_metadata
        )

        # Add cleaning information
        metadata["cleaning_info"] = MetadataGenerator._generate_cleaning_info(
            base_metadata
        )

        # Add statistics
        metadata["statistics"] = MetadataGenerator._generate_statistics(
            df_dirty, df_recovered_clean, base_metadata
        )

        # Add usage instructions
        metadata["usage"] = MetadataGenerator._generate_usage_instructions(
            base_metadata
        )

        return metadata

    @staticmethod
    def _generate_dataset_info(
        df_dirty: pd.DataFrame,
        df_recovered_clean: pd.DataFrame,
        df_perfect: Optional[pd.DataFrame],
    ) -> Dict[str, Any]:
        """Generate dataset information."""
        info = {
            "dirty": {
                "filename": "dirty.csv",
                "shape": {"rows": len(df_dirty), "columns": len(df_dirty.columns)},
                "columns": list(df_dirty.columns),
                "dtypes": {col: str(dtype) for col, dtype in df_dirty.dtypes.items()},
                "description": "Raw dataset with data quality issues",
            },
            "clean": {
                "filename": "clean.csv",
                "shape": {
                    "rows": len(df_recovered_clean),
                    "columns": len(df_recovered_clean.columns),
                },
                "columns": list(df_recovered_clean.columns),
                "dtypes": {
                    col: str(dtype) for col, dtype in df_recovered_clean.dtypes.items()
                },
                "description": "Recovered clean dataset - the actual achievable target for the cleaning agent",
            },
        }

        if df_perfect is not None:
            info["perfect_synthetic"] = {
                "filename": "perfect_synthetic.csv",
                "shape": {"rows": len(df_perfect), "columns": len(df_perfect.columns)},
                "columns": list(df_perfect.columns),
                "dtypes": {
                    col: str(dtype) for col, dtype in df_perfect.dtypes.items()
                },
                "description": "Perfect synthetic dataset (unrealistically clean, for reference only)",
            }

        return info

    @staticmethod
    def _generate_schema_info(config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate schema information."""
        return {
            "name": config.get("name", "unknown"),
            "domain": config.get("domain", "unknown"),
            "num_rows": config.get("num_rows", 0),
            "difficulty": config.get("difficulty", "unknown"),
            "description": config.get("description", ""),
            "seed": config.get("seed"),
        }

    @staticmethod
    def _generate_corruption_details(metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Generate detailed corruption information."""
        corruption_config = metadata.get("corruption_config", {})
        rates = corruption_config.get("rates", {})

        return {
            "strategies_applied": metadata.get("applied_strategies", []),
            "num_strategies": len(metadata.get("applied_strategies", [])),
            "difficulty_level": corruption_config.get("difficulty", "unknown"),
            "corruption_rates": {
                "missing_rate": rates.get("missing_rate"),
                "whitespace_rate": rates.get("whitespace_rate"),
                "case_rate": rates.get("case_rate"),
                "special_char_rate": rates.get("special_char_rate"),
                "duplicate_rate": rates.get("duplicate_rate"),
                "outlier_rate": rates.get("outlier_rate"),
                "outlier_severity": rates.get("outlier_severity"),
            },
            "seed": corruption_config.get("seed"),
        }

    @staticmethod
    def _generate_cleaning_info(metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Generate cleaning operation information."""
        cleaning_ops = metadata.get("cleaning_operations", [])

        # Extract operation sequence
        operation_sequence = [op["operation"] for op in cleaning_ops]

        # Group operations by type
        operations_by_type = {}
        for op in cleaning_ops:
            op_type = op["operation"]
            if op_type not in operations_by_type:
                operations_by_type[op_type] = []
            operations_by_type[op_type].append(op)

        return {
            "total_operations": len(cleaning_ops),
            "operation_sequence": operation_sequence,
            "operations": cleaning_ops,
            "operations_by_type": operations_by_type,
            "recovery_summary": metadata.get("recovery_summary", {}),
        }

    @staticmethod
    def _generate_statistics(
        df_dirty: pd.DataFrame,
        df_recovered_clean: pd.DataFrame,
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate comprehensive statistics."""
        import numpy as np

        recovery_summary = metadata.get("recovery_summary", {})

        stats = {
            "row_counts": {
                "perfect": recovery_summary.get("perfect_rows", "N/A"),
                "dirty": len(df_dirty),
                "recovered": len(df_recovered_clean),
                "rows_lost": recovery_summary.get("rows_lost_in_recovery", 0),
            },
            "missing_values": {
                "dirty": {
                    "total": int(df_dirty.isna().sum().sum()),
                    "percentage": round(
                        (df_dirty.isna().sum().sum() / df_dirty.size) * 100, 2
                    ),
                    "by_column": {
                        col: int(df_dirty[col].isna().sum())
                        for col in df_dirty.columns
                        if df_dirty[col].isna().sum() > 0
                    },
                },
                "recovered": {
                    "total": int(df_recovered_clean.isna().sum().sum()),
                    "percentage": round(
                        (df_recovered_clean.isna().sum().sum() / df_recovered_clean.size)
                        * 100,
                        2,
                    ),
                    "by_column": {
                        col: int(df_recovered_clean[col].isna().sum())
                        for col in df_recovered_clean.columns
                        if df_recovered_clean[col].isna().sum() > 0
                    },
                },
            },
            "duplicates": {
                "dirty": int(df_dirty.duplicated().sum()),
                "recovered": int(df_recovered_clean.duplicated().sum()),
            },
            "memory_usage": {
                "dirty_mb": round(df_dirty.memory_usage(deep=True).sum() / (1024 * 1024), 2),
                "recovered_mb": round(
                    df_recovered_clean.memory_usage(deep=True).sum() / (1024 * 1024), 2
                ),
            },
        }

        return stats

    @staticmethod
    def _generate_usage_instructions(metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Generate usage instructions and notes."""
        return {
            "target_dataset": "clean.csv (recovered clean dataset)",
            "note": "The agent should be trained to match the 'recovered clean' dataset, NOT the perfect synthetic data.",
            "reason": "The recovered clean dataset represents what is realistically achievable given the agent's capabilities (rows may be lost due to outlier/duplicate removal, values may be imputed).",
            "agent_operations_required": [
                op["operation"] for op in metadata.get("cleaning_operations", [])
            ],
            "expected_operation_sequence": [
                op["operation"] for op in metadata.get("cleaning_operations", [])
            ],
            "difficulty_level": metadata.get("corruption_config", {}).get(
                "difficulty", "unknown"
            ),
            "recommended_use": {
                "training": "Use as training example for data cleaning agent",
                "evaluation": "Compare agent output to clean.csv (recovered clean)",
                "baseline": "Agent should achieve similar quality to recovered clean dataset",
            },
        }


# =============================================================================
# Convenience function
# =============================================================================


def generate_metadata(
    df_dirty: pd.DataFrame,
    df_recovered_clean: pd.DataFrame,
    base_metadata: Dict[str, Any],
    df_perfect: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Convenience function to generate comprehensive metadata.

    Args:
        df_dirty: Dirty dataset
        df_recovered_clean: Recovered clean dataset
        base_metadata: Base metadata from pipeline
        df_perfect: Optional perfect synthetic dataset

    Returns:
        Comprehensive metadata dictionary

    Example:
        >>> metadata = generate_metadata(df_dirty, df_clean, base_metadata)
        >>> # Save to file
        >>> import json
        >>> with open("metadata.json", "w") as f:
        ...     json.dump(metadata, f, indent=2)
    """
    generator = MetadataGenerator()
    return generator.generate_comprehensive_metadata(
        df_dirty, df_recovered_clean, base_metadata, df_perfect
    )
