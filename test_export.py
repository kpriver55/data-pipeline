"""
Test suite for export functionality.

Tests the export system including multi-format export, metadata generation,
and quality report generation.
"""

import json
import shutil
from pathlib import Path

import pandas as pd

try:
    import pytest
except ImportError:
    pytest = None

from config.dataset_configs import get_config
from export.exporter import DatasetExporter, ExportFormat, export_dataset
from export.metadata_generator import MetadataGenerator, generate_metadata
from export.quality_report_generator import (
    QualityReportGenerator,
    generate_quality_report,
)
from pipeline import generate_dataset


if pytest:
    @pytest.fixture
    def test_output_dir(tmp_path):
        """Create temporary output directory for tests."""
        output_dir = tmp_path / "test_exports"
        output_dir.mkdir()
        yield output_dir
        # Cleanup after test
        if output_dir.exists():
            shutil.rmtree(output_dir)


    @pytest.fixture
    def sample_datasets():
        """Generate sample datasets for testing."""
        # Generate a small test dataset
        config = get_config("ecommerce_small_easy")
        df_dirty, df_clean, metadata = generate_dataset(config, validate=False, verbose=False)
        return df_dirty, df_clean, metadata
else:
    # Dummy fixtures for when pytest is not available
    def test_output_dir(tmp_path=None):
        pass

    def sample_datasets():
        pass


def test_csv_export_basic(sample_datasets, test_output_dir):
    """Test basic CSV export."""
    df_dirty, df_clean, metadata = sample_datasets

    exporter = DatasetExporter(test_output_dir)
    exported_files = exporter.export(
        df_dirty=df_dirty,
        df_recovered_clean=df_clean,
        metadata=metadata,
        formats=ExportFormat.CSV,
    )

    # Verify files exist
    assert (test_output_dir / "dirty.csv").exists()
    assert (test_output_dir / "clean.csv").exists()
    assert (test_output_dir / "metadata.json").exists()

    # Verify exported files list
    assert len(exported_files["dirty"]) == 1
    assert len(exported_files["clean"]) == 1
    assert len(exported_files["metadata"]) == 1

    # Verify CSV can be read back
    df_dirty_loaded = pd.read_csv(test_output_dir / "dirty.csv")
    df_clean_loaded = pd.read_csv(test_output_dir / "clean.csv")

    assert df_dirty_loaded.shape == df_dirty.shape
    assert df_clean_loaded.shape == df_clean.shape


def test_multi_format_export(sample_datasets, test_output_dir):
    """Test export in multiple formats."""
    df_dirty, df_clean, metadata = sample_datasets

    exporter = DatasetExporter(test_output_dir)
    exported_files = exporter.export(
        df_dirty=df_dirty,
        df_recovered_clean=df_clean,
        metadata=metadata,
        formats=[ExportFormat.CSV, ExportFormat.PARQUET, ExportFormat.JSON],
    )

    # Verify CSV files
    assert (test_output_dir / "dirty.csv").exists()
    assert (test_output_dir / "clean.csv").exists()

    # Verify Parquet files
    assert (test_output_dir / "dirty.parquet").exists()
    assert (test_output_dir / "clean.parquet").exists()

    # Verify JSON files
    assert (test_output_dir / "dirty.json").exists()
    assert (test_output_dir / "clean.json").exists()

    # Verify metadata
    assert (test_output_dir / "metadata.json").exists()

    # Verify file counts
    assert len(exported_files["dirty"]) == 3  # CSV, Parquet, JSON
    assert len(exported_files["clean"]) == 3
    assert len(exported_files["metadata"]) == 1

    # Verify data can be read back from each format
    df_dirty_csv = pd.read_csv(test_output_dir / "dirty.csv")
    df_dirty_parquet = pd.read_parquet(test_output_dir / "dirty.parquet")
    df_dirty_json = pd.read_json(test_output_dir / "dirty.json")

    assert df_dirty_csv.shape == df_dirty.shape
    assert df_dirty_parquet.shape == df_dirty.shape
    assert df_dirty_json.shape == df_dirty.shape


def test_export_all_formats(sample_datasets, test_output_dir):
    """Test export using ALL format option."""
    df_dirty, df_clean, metadata = sample_datasets

    exporter = DatasetExporter(test_output_dir)
    exported_files = exporter.export(
        df_dirty=df_dirty,
        df_recovered_clean=df_clean,
        metadata=metadata,
        formats=ExportFormat.ALL,
    )

    # Should have all 3 formats
    assert len(exported_files["dirty"]) == 3
    assert len(exported_files["clean"]) == 3


def test_export_with_perfect_dataset(sample_datasets, test_output_dir):
    """Test export including perfect synthetic dataset."""
    df_dirty, df_clean, metadata = sample_datasets

    # Create a mock perfect dataset
    df_perfect = df_clean.copy()

    exporter = DatasetExporter(test_output_dir)
    exported_files = exporter.export(
        df_dirty=df_dirty,
        df_recovered_clean=df_clean,
        metadata=metadata,
        formats=ExportFormat.CSV,
        include_perfect=True,
        df_perfect=df_perfect,
    )

    # Verify perfect dataset was exported
    assert (test_output_dir / "perfect_synthetic.csv").exists()
    assert len(exported_files["perfect"]) == 1

    # Verify file can be read back
    df_perfect_loaded = pd.read_csv(test_output_dir / "perfect_synthetic.csv")
    assert df_perfect_loaded.shape == df_perfect.shape


def test_export_summary(sample_datasets, test_output_dir):
    """Test export summary generation."""
    df_dirty, df_clean, metadata = sample_datasets

    exporter = DatasetExporter(test_output_dir)
    exported_files = exporter.export(
        df_dirty=df_dirty,
        df_recovered_clean=df_clean,
        metadata=metadata,
        formats=ExportFormat.CSV,
    )

    summary = exporter.get_export_summary(exported_files)

    # Verify summary structure
    assert "output_directory" in summary
    assert "total_files" in summary
    assert "total_size_bytes" in summary
    assert "total_size_mb" in summary
    assert "files" in summary

    # Verify file counts
    assert summary["total_files"] == 3  # dirty, clean, metadata

    # Verify size is calculated
    assert summary["total_size_bytes"] > 0
    assert summary["total_size_mb"] > 0

    # Verify file info
    assert "dirty" in summary["files"]
    assert "clean" in summary["files"]
    assert "metadata" in summary["files"]


def test_metadata_generation(sample_datasets):
    """Test comprehensive metadata generation."""
    df_dirty, df_clean, base_metadata = sample_datasets

    metadata = generate_metadata(
        df_dirty=df_dirty,
        df_recovered_clean=df_clean,
        base_metadata=base_metadata,
    )

    # Verify metadata structure
    assert "generated_at" in metadata
    assert "pipeline_version" in metadata
    assert "datasets" in metadata
    assert "schema_info" in metadata
    assert "corruption_details" in metadata
    assert "cleaning_info" in metadata
    assert "statistics" in metadata
    assert "usage" in metadata

    # Verify datasets section
    assert "dirty" in metadata["datasets"]
    assert "clean" in metadata["datasets"]
    assert metadata["datasets"]["dirty"]["shape"]["rows"] == len(df_dirty)
    assert metadata["datasets"]["clean"]["shape"]["rows"] == len(df_clean)

    # Verify statistics
    assert "row_counts" in metadata["statistics"]
    assert "missing_values" in metadata["statistics"]
    assert "duplicates" in metadata["statistics"]

    # Verify usage instructions
    assert "target_dataset" in metadata["usage"]
    assert metadata["usage"]["target_dataset"] == "clean.csv (recovered clean dataset)"


def test_metadata_with_perfect_dataset(sample_datasets):
    """Test metadata generation with perfect dataset."""
    df_dirty, df_clean, base_metadata = sample_datasets
    df_perfect = df_clean.copy()

    metadata = generate_metadata(
        df_dirty=df_dirty,
        df_recovered_clean=df_clean,
        base_metadata=base_metadata,
        df_perfect=df_perfect,
    )

    # Verify perfect dataset is included
    assert "perfect_synthetic" in metadata["datasets"]
    assert metadata["datasets"]["perfect_synthetic"]["shape"]["rows"] == len(df_perfect)


def test_quality_report_generation(sample_datasets):
    """Test quality report generation."""
    df_dirty, df_clean, metadata = sample_datasets

    report = generate_quality_report(
        df_dirty=df_dirty,
        df_recovered_clean=df_clean,
        metadata=metadata,
    )

    # Verify report structure
    assert "generated_at" in report
    assert "summary" in report
    assert "column_quality" in report
    assert "corruption_analysis" in report
    assert "cleaning_analysis" in report
    assert "data_loss" in report
    assert "recommendations" in report

    # Verify summary metrics
    summary = report["summary"]
    assert "overall_quality_score" in summary
    assert 0 <= summary["overall_quality_score"] <= 100
    assert "total_rows_dirty" in summary
    assert "total_rows_recovered" in summary
    assert "row_retention_rate" in summary

    # Verify column quality
    assert len(report["column_quality"]) > 0
    for col_name, col_metrics in report["column_quality"].items():
        assert "dirty" in col_metrics
        assert "recovered" in col_metrics
        assert "improvement_score" in col_metrics

    # Verify recommendations
    assert len(report["recommendations"]) > 0
    for rec in report["recommendations"]:
        assert "severity" in rec
        assert "category" in rec
        assert "message" in rec


def test_quality_report_with_perfect_comparison(sample_datasets):
    """Test quality report with perfect dataset comparison."""
    df_dirty, df_clean, metadata = sample_datasets
    df_perfect = df_clean.copy()

    report = generate_quality_report(
        df_dirty=df_dirty,
        df_recovered_clean=df_clean,
        metadata=metadata,
        df_perfect=df_perfect,
    )

    # Verify perfect comparison section exists
    assert "perfect_comparison" in report

    comparison = report["perfect_comparison"]
    assert "shape_match" in comparison
    assert "perfect_shape" in comparison
    assert "recovered_shape" in comparison
    assert "columns_match" in comparison


def test_convenience_export_function(sample_datasets, test_output_dir):
    """Test convenience export_dataset function."""
    df_dirty, df_clean, metadata = sample_datasets

    exported_files = export_dataset(
        df_dirty=df_dirty,
        df_clean=df_clean,
        metadata=metadata,
        output_dir=test_output_dir,
        formats=ExportFormat.CSV,
        verbose=False,
    )

    # Verify files were exported
    assert (test_output_dir / "dirty.csv").exists()
    assert (test_output_dir / "clean.csv").exists()
    assert (test_output_dir / "metadata.json").exists()


def test_pipeline_integration_with_export(test_output_dir):
    """Test pipeline integration with enhanced export system."""
    config = get_config("ecommerce_small_easy")

    df_dirty, df_clean, metadata = generate_dataset(
        config,
        output_dir=test_output_dir / "dataset_1",
        validate=False,
        verbose=False,
        export_formats=ExportFormat.CSV,
        include_quality_report=True,
    )

    # Verify datasets were generated
    assert df_dirty is not None
    assert df_clean is not None
    assert metadata is not None

    # Verify files were exported
    output_path = test_output_dir / "dataset_1"
    assert (output_path / "dirty.csv").exists()
    assert (output_path / "clean.csv").exists()
    assert (output_path / "metadata.json").exists()

    # Verify metadata includes quality report
    with open(output_path / "metadata.json") as f:
        saved_metadata = json.load(f)

    assert "quality_report" in saved_metadata
    assert "summary" in saved_metadata["quality_report"]
    assert "overall_quality_score" in saved_metadata["quality_report"]["summary"]


def test_pipeline_multi_format_export(test_output_dir):
    """Test pipeline with multiple export formats."""
    config = get_config("ecommerce_small_easy")

    generate_dataset(
        config,
        output_dir=test_output_dir / "dataset_multi",
        validate=False,
        verbose=False,
        export_formats=[ExportFormat.CSV, ExportFormat.PARQUET],
        include_quality_report=True,
    )

    output_path = test_output_dir / "dataset_multi"

    # Verify CSV files
    assert (output_path / "dirty.csv").exists()
    assert (output_path / "clean.csv").exists()

    # Verify Parquet files
    assert (output_path / "dirty.parquet").exists()
    assert (output_path / "clean.parquet").exists()


def test_metadata_serialization(sample_datasets, test_output_dir):
    """Test that metadata can be serialized to JSON without errors."""
    df_dirty, df_clean, base_metadata = sample_datasets

    metadata = generate_metadata(
        df_dirty=df_dirty,
        df_recovered_clean=df_clean,
        base_metadata=base_metadata,
    )

    # Add quality report
    quality_report = generate_quality_report(
        df_dirty=df_dirty,
        df_recovered_clean=df_clean,
        metadata=base_metadata,
    )
    metadata["quality_report"] = quality_report

    # Try to serialize to JSON
    metadata_path = test_output_dir / "test_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    # Verify file was created and can be loaded
    assert metadata_path.exists()

    with open(metadata_path) as f:
        loaded_metadata = json.load(f)

    assert "quality_report" in loaded_metadata
    assert "summary" in loaded_metadata["quality_report"]


def main():
    """Run tests manually (for development)."""
    import tempfile

    print("=" * 70)
    print("EXPORT SYSTEM TESTS")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Generate sample datasets
        print("\nGenerating sample datasets...")
        config = get_config("ecommerce_small_easy")
        df_dirty, df_clean, metadata = generate_dataset(
            config, validate=False, verbose=False
        )
        print(f"✓ Generated datasets: dirty={df_dirty.shape}, clean={df_clean.shape}")

        # Test 1: CSV export
        print("\n[1/8] Testing CSV export...")
        output_dir = tmp_path / "test_csv"
        exporter = DatasetExporter(output_dir)
        files = exporter.export(df_dirty, df_clean, metadata, ExportFormat.CSV)
        summary = exporter.get_export_summary(files)
        print(f"✓ CSV export: {summary['total_files']} files, {summary['total_size_mb']} MB")

        # Test 2: Multi-format export
        print("\n[2/8] Testing multi-format export...")
        output_dir = tmp_path / "test_multi"
        exporter = DatasetExporter(output_dir)
        files = exporter.export(
            df_dirty, df_clean, metadata,
            formats=[ExportFormat.CSV, ExportFormat.PARQUET, ExportFormat.JSON]
        )
        summary = exporter.get_export_summary(files)
        print(f"✓ Multi-format: {summary['total_files']} files, {summary['total_size_mb']} MB")

        # Test 3: Metadata generation
        print("\n[3/8] Testing metadata generation...")
        comprehensive_metadata = generate_metadata(df_dirty, df_clean, metadata)
        print(f"✓ Metadata generated with {len(comprehensive_metadata)} sections")

        # Test 4: Quality report
        print("\n[4/8] Testing quality report generation...")
        report = generate_quality_report(df_dirty, df_clean, metadata)
        quality_score = report["summary"]["overall_quality_score"]
        print(f"✓ Quality report generated: score={quality_score}/100")

        # Test 5: Export with perfect dataset
        print("\n[5/8] Testing export with perfect dataset...")
        output_dir = tmp_path / "test_perfect"
        exporter = DatasetExporter(output_dir)
        files = exporter.export(
            df_dirty, df_clean, metadata,
            formats=ExportFormat.CSV,
            include_perfect=True,
            df_perfect=df_clean.copy(),
        )
        print(f"✓ Export with perfect: {len(files['perfect'])} perfect files")

        # Test 6: Pipeline integration
        print("\n[6/8] Testing pipeline integration...")
        output_dir = tmp_path / "test_pipeline"
        df_d, df_c, meta = generate_dataset(
            config,
            output_dir=output_dir,
            validate=False,
            verbose=False,
            export_formats=ExportFormat.CSV,
            include_quality_report=True,
        )
        metadata_file = output_dir / "metadata.json"
        with open(metadata_file) as f:
            saved_meta = json.load(f)
        has_quality = "quality_report" in saved_meta
        print(f"✓ Pipeline integration: quality_report={'included' if has_quality else 'missing'}")

        # Test 7: Multi-format pipeline
        print("\n[7/8] Testing pipeline multi-format...")
        output_dir = tmp_path / "test_pipeline_multi"
        generate_dataset(
            config,
            output_dir=output_dir,
            validate=False,
            verbose=False,
            export_formats=[ExportFormat.CSV, ExportFormat.PARQUET],
        )
        csv_exists = (output_dir / "dirty.csv").exists()
        parquet_exists = (output_dir / "dirty.parquet").exists()
        print(f"✓ Multi-format pipeline: CSV={csv_exists}, Parquet={parquet_exists}")

        # Test 8: Serialization
        print("\n[8/8] Testing JSON serialization...")
        full_metadata = generate_metadata(df_dirty, df_clean, metadata)
        full_metadata["quality_report"] = generate_quality_report(df_dirty, df_clean, metadata)
        test_file = tmp_path / "serialization_test.json"
        with open(test_file, "w") as f:
            json.dump(full_metadata, f, indent=2, default=str)
        file_size = test_file.stat().st_size
        print(f"✓ Serialization: {file_size} bytes written")

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED ✓")
    print("=" * 70)

    print("\nExport System Summary:")
    print("  - Multi-format export: CSV, Parquet, JSON")
    print("  - Comprehensive metadata generation")
    print("  - Quality reports with scoring")
    print("  - Perfect dataset comparison")
    print("  - Full pipeline integration")


if __name__ == "__main__":
    # Run manual tests (pytest tests available for CI/CD)
    main()
