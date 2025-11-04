"""
Test script for Phase 5: Pipeline Orchestration

Tests the end-to-end pipeline for generating dataset pairs.

Run with: python3 test_pipeline.py
"""

from pathlib import Path
import shutil

from config.dataset_configs import (
    DatasetConfig,
    get_config,
    list_configs,
    ECOMMERCE_MEDIUM,
)
from pipeline import (
    SyntheticDataPipeline,
    generate_dataset,
    generate_batch_datasets,
    generate_training_set,
)


def test_single_dataset_generation():
    """Test generating a single dataset using preset config."""
    print("=" * 70)
    print("Testing Single Dataset Generation (Preset Config)")
    print("=" * 70)

    # Generate using preset config name
    df_dirty, df_clean, metadata = generate_dataset(
        "ecommerce_medium", verbose=True
    )

    # Verify outputs
    assert df_dirty.shape[0] > 0, "Dirty dataset should have rows"
    assert df_clean.shape[0] > 0, "Clean dataset should have rows"
    assert len(metadata["cleaning_operations"]) > 0, "Should have cleaning operations"

    print("\n✓ Single dataset generation test passed")
    print(f"  Dirty shape: {df_dirty.shape}")
    print(f"  Clean shape: {df_clean.shape}")
    print(f"  Operations: {len(metadata['cleaning_operations'])}")


def test_custom_config():
    """Test generating dataset with custom configuration."""
    print("\n" + "=" * 70)
    print("Testing Custom Configuration")
    print("=" * 70)

    # Create custom config
    custom_config = DatasetConfig(
        name="custom_test",
        domain="finance",
        num_rows=200,
        difficulty="easy",
        seed=99,
        description="Custom test configuration",
    )

    # Generate
    df_dirty, df_clean, metadata = generate_dataset(custom_config, verbose=True)

    # Verify
    assert df_dirty.shape[0] > 0, "Should generate data"
    assert metadata["dataset_config"]["domain"] == "finance"
    assert metadata["dataset_config"]["num_rows"] == 200

    print("\n✓ Custom configuration test passed")


def test_specific_strategies():
    """Test dataset with specific corruption strategies."""
    print("\n" + "=" * 70)
    print("Testing Specific Corruption Strategies")
    print("=" * 70)

    # Text-only corruption
    config = DatasetConfig(
        name="text_only_test",
        domain="ecommerce",
        num_rows=150,
        difficulty="custom",
        strategies=["text_whitespace", "text_case"],
        seed=100,
    )

    df_dirty, df_clean, metadata = generate_dataset(config, verbose=True)

    # Verify only text strategies were applied
    applied = set(metadata["applied_strategies"])
    expected = {"text_whitespace", "text_case"}

    assert applied == expected, f"Expected {expected}, got {applied}"

    print("\n✓ Specific strategies test passed")
    print(f"  Applied strategies: {metadata['applied_strategies']}")


def test_batch_generation():
    """Test generating multiple datasets in batch."""
    print("\n" + "=" * 70)
    print("Testing Batch Dataset Generation")
    print("=" * 70)

    configs = [
        "ecommerce_small_easy",
        "healthcare_small_easy",
        "finance_small_easy",
    ]

    results = generate_batch_datasets(configs, verbose=True)

    # Verify we got 3 datasets
    assert len(results) == 3, "Should generate 3 datasets"

    for i, (df_dirty, df_clean, metadata) in enumerate(results):
        assert df_dirty.shape[0] > 0, f"Dataset {i} should have dirty data"
        assert df_clean.shape[0] > 0, f"Dataset {i} should have clean data"

    print("\n✓ Batch generation test passed")
    print(f"  Generated {len(results)} datasets")


def test_file_export():
    """Test exporting datasets to files."""
    print("\n" + "=" * 70)
    print("Testing File Export")
    print("=" * 70)

    output_dir = Path("output/test_pipeline_export")

    # Clean up if exists
    if output_dir.exists():
        shutil.rmtree(output_dir)

    try:
        # Generate with export
        df_dirty, df_clean, metadata = generate_dataset(
            "ecommerce_small_easy",
            output_dir=output_dir,
            verbose=True,
        )

        # Verify files were created
        assert (output_dir / "dirty.csv").exists(), "Should create dirty.csv"
        assert (output_dir / "clean.csv").exists(), "Should create clean.csv"
        assert (output_dir / "metadata.json").exists(), "Should create metadata.json"

        # Verify we can load the files
        import pandas as pd
        import json

        df_dirty_loaded = pd.read_csv(output_dir / "dirty.csv")
        df_clean_loaded = pd.read_csv(output_dir / "clean.csv")

        with open(output_dir / "metadata.json") as f:
            metadata_loaded = json.load(f)

        assert df_dirty_loaded.shape == df_dirty.shape, "Dirty data should match"
        assert df_clean_loaded.shape == df_clean.shape, "Clean data should match"
        assert "cleaning_operations" in metadata_loaded, "Metadata should be complete"

        print("\n✓ File export test passed")
        print(f"  Files created in: {output_dir}")
        print(f"    - dirty.csv: {df_dirty_loaded.shape}")
        print(f"    - clean.csv: {df_clean_loaded.shape}")
        print(f"    - metadata.json: {len(metadata_loaded)} keys")

    finally:
        # Clean up
        if output_dir.exists():
            shutil.rmtree(output_dir)
            print(f"  Cleaned up test directory")


def test_training_set_generation():
    """Test generating a complete training set."""
    print("\n" + "=" * 70)
    print("Testing Training Set Generation")
    print("=" * 70)

    output_dir = Path("output/test_training_set")

    # Clean up if exists
    if output_dir.exists():
        shutil.rmtree(output_dir)

    try:
        # Generate small training set
        results = generate_training_set(
            domain="ecommerce",
            num_easy=2,
            num_medium=2,
            num_hard=2,
            rows_per_dataset=100,
            output_dir=output_dir,
            base_seed=42,
        )

        # Verify we got 6 datasets (2 + 2 + 2)
        assert len(results) == 6, "Should generate 6 datasets"

        # Check difficulty distribution
        difficulties = []
        for _, _, metadata in results:
            config = metadata["dataset_config"]
            difficulties.append(config["difficulty"])

        assert difficulties.count("easy") == 2, "Should have 2 easy"
        assert difficulties.count("medium") == 2, "Should have 2 medium"
        assert difficulties.count("hard") == 2, "Should have 2 hard"

        # Verify files were created
        subdirs = list(output_dir.glob("*"))
        assert len(subdirs) == 6, "Should create 6 subdirectories"

        print("\n✓ Training set generation test passed")
        print(f"  Generated {len(results)} training examples")
        print(f"  Difficulty distribution: {dict(zip(['easy', 'medium', 'hard'], [2, 2, 2]))}")

    finally:
        # Clean up
        if output_dir.exists():
            shutil.rmtree(output_dir)
            print(f"  Cleaned up test directory")


def test_config_registry():
    """Test configuration registry functions."""
    print("\n" + "=" * 70)
    print("Testing Configuration Registry")
    print("=" * 70)

    # Test listing all configs
    all_configs = list_configs()
    assert len(all_configs) > 0, "Should have pre-defined configs"
    print(f"\n  Total configs: {len(all_configs)}")

    # Test getting specific config
    config = get_config("ecommerce_medium")
    assert config.domain == "ecommerce", "Should be ecommerce domain"
    assert config.difficulty == "medium", "Should be medium difficulty"
    print(f"  Retrieved config: {config.name}")

    # Test invalid config name
    try:
        get_config("nonexistent_config")
        assert False, "Should raise KeyError for invalid config"
    except KeyError as e:
        print(f"  ✓ Correctly raised error for invalid config")

    print("\n✓ Configuration registry test passed")


def test_pipeline_class():
    """Test using the pipeline class directly."""
    print("\n" + "=" * 70)
    print("Testing Pipeline Class API")
    print("=" * 70)

    # Create pipeline with custom settings
    pipeline = SyntheticDataPipeline(validate=True, verbose=True)

    # Generate dataset
    df_dirty, df_clean, metadata = pipeline.generate("healthcare_small_easy")

    # Verify
    assert df_dirty.shape[0] > 0, "Should generate data"
    assert metadata["dataset_config"]["domain"] == "healthcare"

    print("\n✓ Pipeline class test passed")


def test_validation_integration():
    """Test that validation catches issues."""
    print("\n" + "=" * 70)
    print("Testing Validation Integration")
    print("=" * 70)

    # Normal generation should pass validation
    try:
        df_dirty, df_clean, metadata = generate_dataset(
            "ecommerce_small_easy",
            validate=True,  # Validation enabled
            verbose=False,
        )
        print("  ✓ Validation passed for valid dataset")
    except RuntimeError:
        assert False, "Validation should pass for properly generated datasets"

    print("\n✓ Validation integration test passed")


def main():
    """Run all pipeline tests."""
    print("\n" + "=" * 70)
    print("PHASE 5: PIPELINE ORCHESTRATION TESTS")
    print("=" * 70)

    try:
        # Test basic functionality
        test_single_dataset_generation()
        test_custom_config()
        test_specific_strategies()

        # Test batch operations
        test_batch_generation()
        test_training_set_generation()

        # Test file I/O
        test_file_export()

        # Test configuration system
        test_config_registry()

        # Test API variations
        test_pipeline_class()
        test_validation_integration()

        print("\n" + "=" * 70)
        print("ALL PIPELINE TESTS PASSED ✓")
        print("=" * 70)
        print("\nPhase 5 (Pipeline Orchestration) is complete!")
        print("\nKey Features:")
        print("  - End-to-end dataset generation pipeline")
        print("  - Pre-defined configurations for common use cases")
        print("  - Batch generation capabilities")
        print("  - Training set generation helper")
        print("  - Automatic validation integration")
        print("  - File export with metadata")

        return 0

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
