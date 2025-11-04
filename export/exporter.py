"""
Dataset exporter supporting multiple formats.

Exports dirty and clean datasets to CSV, Parquet, and JSON formats
with comprehensive metadata and data quality reports.
"""

import json
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd


class ExportFormat(Enum):
    """Supported export formats."""

    CSV = "csv"
    PARQUET = "parquet"
    JSON = "json"
    ALL = "all"  # Export in all formats


class DatasetExporter:
    """Export datasets in multiple formats with metadata."""

    def __init__(self, output_dir: Union[str, Path]):
        """
        Initialize exporter.

        Args:
            output_dir: Directory to save exported files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export(
        self,
        df_dirty: pd.DataFrame,
        df_recovered_clean: pd.DataFrame,
        metadata: Dict[str, Any],
        formats: Union[ExportFormat, List[ExportFormat]] = ExportFormat.CSV,
        include_perfect: bool = False,
        df_perfect: Optional[pd.DataFrame] = None,
    ) -> Dict[str, List[Path]]:
        """
        Export datasets in specified format(s).

        Args:
            df_dirty: Dirty dataset
            df_recovered_clean: Recovered clean dataset (actual target)
            metadata: Dataset metadata
            formats: Export format(s) to use
            include_perfect: Whether to also export perfect synthetic data
            df_perfect: Perfect synthetic data (required if include_perfect=True)

        Returns:
            Dictionary mapping dataset type to list of file paths

        Example:
            >>> exporter = DatasetExporter("output/dataset_1")
            >>> paths = exporter.export(
            ...     df_dirty, df_clean, metadata,
            ...     formats=[ExportFormat.CSV, ExportFormat.PARQUET]
            ... )
        """
        # Normalize formats to list
        if isinstance(formats, ExportFormat):
            if formats == ExportFormat.ALL:
                format_list = [ExportFormat.CSV, ExportFormat.PARQUET, ExportFormat.JSON]
            else:
                format_list = [formats]
        else:
            format_list = formats

        exported_files = {
            "dirty": [],
            "clean": [],
            "perfect": [],
            "metadata": [],
        }

        # Export datasets in each format
        for fmt in format_list:
            if fmt == ExportFormat.CSV:
                exported_files["dirty"].append(
                    self._export_csv(df_dirty, "dirty.csv")
                )
                exported_files["clean"].append(
                    self._export_csv(df_recovered_clean, "clean.csv")
                )
                if include_perfect and df_perfect is not None:
                    exported_files["perfect"].append(
                        self._export_csv(df_perfect, "perfect_synthetic.csv")
                    )

            elif fmt == ExportFormat.PARQUET:
                exported_files["dirty"].append(
                    self._export_parquet(df_dirty, "dirty.parquet")
                )
                exported_files["clean"].append(
                    self._export_parquet(df_recovered_clean, "clean.parquet")
                )
                if include_perfect and df_perfect is not None:
                    exported_files["perfect"].append(
                        self._export_parquet(df_perfect, "perfect_synthetic.parquet")
                    )

            elif fmt == ExportFormat.JSON:
                exported_files["dirty"].append(
                    self._export_json(df_dirty, "dirty.json")
                )
                exported_files["clean"].append(
                    self._export_json(df_recovered_clean, "clean.json")
                )
                if include_perfect and df_perfect is not None:
                    exported_files["perfect"].append(
                        self._export_json(df_perfect, "perfect_synthetic.json")
                    )

        # Export metadata
        metadata_path = self._export_metadata(metadata)
        exported_files["metadata"].append(metadata_path)

        return exported_files

    def _export_csv(self, df: pd.DataFrame, filename: str) -> Path:
        """Export DataFrame to CSV."""
        path = self.output_dir / filename
        df.to_csv(path, index=False)
        return path

    def _export_parquet(self, df: pd.DataFrame, filename: str) -> Path:
        """Export DataFrame to Parquet."""
        path = self.output_dir / filename
        df.to_parquet(path, index=False, engine="pyarrow")
        return path

    def _export_json(self, df: pd.DataFrame, filename: str) -> Path:
        """Export DataFrame to JSON (records format)."""
        path = self.output_dir / filename
        df.to_json(path, orient="records", indent=2, date_format="iso")
        return path

    def _export_metadata(self, metadata: Dict[str, Any]) -> Path:
        """Export metadata to JSON."""
        path = self.output_dir / "metadata.json"

        # Convert any non-serializable objects to strings
        def convert_to_serializable(obj):
            """Convert numpy/pandas types to Python native types."""
            if isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            elif hasattr(obj, "item"):  # numpy scalar
                return obj.item()
            elif isinstance(obj, (pd.Timestamp, pd.Timedelta)):
                return str(obj)
            else:
                return obj

        serializable_metadata = convert_to_serializable(metadata)

        with open(path, "w") as f:
            json.dump(serializable_metadata, f, indent=2, default=str)

        return path

    def get_export_summary(
        self, exported_files: Dict[str, List[Path]]
    ) -> Dict[str, Any]:
        """
        Generate summary of exported files.

        Args:
            exported_files: Dictionary of exported file paths

        Returns:
            Summary with file info
        """
        summary = {
            "output_directory": str(self.output_dir),
            "files": {},
            "total_files": 0,
            "total_size_bytes": 0,
        }

        for dataset_type, paths in exported_files.items():
            if not paths:
                continue

            file_info = []
            for path in paths:
                if path.exists():
                    size = path.stat().st_size
                    file_info.append({
                        "path": str(path),
                        "filename": path.name,
                        "size_bytes": size,
                        "size_mb": round(size / (1024 * 1024), 2),
                    })
                    summary["total_size_bytes"] += size
                    summary["total_files"] += 1

            if file_info:
                summary["files"][dataset_type] = file_info

        summary["total_size_mb"] = round(
            summary["total_size_bytes"] / (1024 * 1024), 2
        )

        return summary


# =============================================================================
# Convenience functions
# =============================================================================


def export_dataset(
    df_dirty: pd.DataFrame,
    df_clean: pd.DataFrame,
    metadata: Dict[str, Any],
    output_dir: Union[str, Path],
    formats: Union[ExportFormat, List[ExportFormat]] = ExportFormat.CSV,
    include_perfect: bool = False,
    df_perfect: Optional[pd.DataFrame] = None,
    verbose: bool = True,
) -> Dict[str, List[Path]]:
    """
    Convenience function to export a dataset.

    Args:
        df_dirty: Dirty dataset
        df_clean: Recovered clean dataset
        metadata: Dataset metadata
        output_dir: Output directory
        formats: Export format(s)
        include_perfect: Whether to export perfect synthetic data
        df_perfect: Perfect synthetic data
        verbose: Print export summary

    Returns:
        Dictionary of exported file paths

    Example:
        >>> paths = export_dataset(
        ...     df_dirty, df_clean, metadata,
        ...     output_dir="output/my_dataset",
        ...     formats=[ExportFormat.CSV, ExportFormat.PARQUET]
        ... )
    """
    exporter = DatasetExporter(output_dir)

    exported_files = exporter.export(
        df_dirty=df_dirty,
        df_recovered_clean=df_clean,
        metadata=metadata,
        formats=formats,
        include_perfect=include_perfect,
        df_perfect=df_perfect,
    )

    if verbose:
        summary = exporter.get_export_summary(exported_files)
        print(f"\n✓ Exported to: {summary['output_directory']}")
        print(f"  Total files: {summary['total_files']}")
        print(f"  Total size: {summary['total_size_mb']} MB")

        for dataset_type, files in summary["files"].items():
            if files:
                print(f"\n  {dataset_type.capitalize()} files:")
                for file_info in files:
                    print(f"    - {file_info['filename']} ({file_info['size_mb']} MB)")

    return exported_files
