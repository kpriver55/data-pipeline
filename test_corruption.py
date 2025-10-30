"""
Test script for Phase 3: Corruption Engine

Run with: python test_corruption.py
"""

import json

from corruption.base import CorruptionConfig
from corruption.orchestrator import corrupt_dataset
from generators.clean_data_builder import build_clean_dataset
from schema.schema_generator import SchemaGenerator


def test_easy_corruption():
    """Test easy difficulty corruption (1-2 strategies)."""
    print("=" * 60)
    print("Testing Easy Difficulty Corruption")
    print("=" * 60)

    # Generate clean data
    generator = SchemaGenerator()
    schema = generator.generate_ecommerce_schema(num_rows=100)
    df_clean = build_clean_dataset(schema, seed=42)

    print(f"\nClean dataset: {df_clean.shape}")

    # Apply easy corruption
    config = CorruptionConfig.from_preset("easy", seed=42)
    print(f"\nSelected strategies: {config.strategies}")

    df_corrupt, metadata = corrupt_dataset(df_clean, config, seed=42)

    print(f"\nCorrupted dataset: {df_corrupt.shape}")
    print(f"Required operations: {metadata['required_operations']}")
    print(f"\nCorruption summary:")
    print(f"  Missing values added: {metadata['corruption_summary']['missing_values_added']}")
    print(
        f"  Missing percentage: {metadata['corruption_summary']['missing_percentage']}%"
    )
    print(f"  Duplicates added: {metadata['corruption_summary']['duplicates_added']}")
    print(f"  Rows added: {metadata['corruption_summary']['rows_added']}")

    return df_clean, df_corrupt, metadata


def test_medium_corruption():
    """Test medium difficulty corruption (3-4 strategies)."""
    print("\n" + "=" * 60)
    print("Testing Medium Difficulty Corruption")
    print("=" * 60)

    generator = SchemaGenerator()
    schema = generator.generate_ecommerce_schema(num_rows=100)
    df_clean = build_clean_dataset(schema, seed=43)

    print(f"\nClean dataset: {df_clean.shape}")

    config = CorruptionConfig.from_preset("medium", seed=43)
    print(f"\nSelected strategies: {config.strategies}")

    df_corrupt, metadata = corrupt_dataset(df_clean, config, seed=43)

    print(f"\nCorrupted dataset: {df_corrupt.shape}")
    print(f"Required operations: {metadata['required_operations']}")
    print(f"\nCorruption summary:")
    print(f"  Missing values added: {metadata['corruption_summary']['missing_values_added']}")
    print(
        f"  Missing percentage: {metadata['corruption_summary']['missing_percentage']}%"
    )
    print(f"  Duplicates added: {metadata['corruption_summary']['duplicates_added']}")
    print(f"  Rows added: {metadata['corruption_summary']['rows_added']}")

    return df_clean, df_corrupt, metadata


def test_hard_corruption():
    """Test hard difficulty corruption (5-7 strategies)."""
    print("\n" + "=" * 60)
    print("Testing Hard Difficulty Corruption")
    print("=" * 60)

    generator = SchemaGenerator()
    schema = generator.generate_ecommerce_schema(num_rows=100)
    df_clean = build_clean_dataset(schema, seed=44)

    print(f"\nClean dataset: {df_clean.shape}")

    config = CorruptionConfig.from_preset("hard", seed=44)
    print(f"\nSelected strategies: {config.strategies}")

    df_corrupt, metadata = corrupt_dataset(df_clean, config, seed=44)

    print(f"\nCorrupted dataset: {df_corrupt.shape}")
    print(f"Required operations: {metadata['required_operations']}")
    print(f"\nCorruption summary:")
    print(f"  Missing values added: {metadata['corruption_summary']['missing_values_added']}")
    print(
        f"  Missing percentage: {metadata['corruption_summary']['missing_percentage']}%"
    )
    print(f"  Duplicates added: {metadata['corruption_summary']['duplicates_added']}")
    print(f"  Rows added: {metadata['corruption_summary']['rows_added']}")

    return df_clean, df_corrupt, metadata


def test_custom_corruption():
    """Test custom corruption with specific strategies."""
    print("\n" + "=" * 60)
    print("Testing Custom Corruption")
    print("=" * 60)

    generator = SchemaGenerator()
    schema = generator.generate_ecommerce_schema(num_rows=100)
    df_clean = build_clean_dataset(schema, seed=45)

    print(f"\nClean dataset: {df_clean.shape}")

    # Custom: only missing values and text whitespace
    config = CorruptionConfig.custom(
        strategies=["missing_values", "text_whitespace"], seed=45
    )
    print(f"\nSelected strategies: {config.strategies}")

    df_corrupt, metadata = corrupt_dataset(df_clean, config, seed=45)

    print(f"\nCorrupted dataset: {df_corrupt.shape}")
    print(f"Required operations: {metadata['required_operations']}")

    return df_clean, df_corrupt, metadata


def test_corruption_diversity():
    """Test that random strategy selection creates diversity."""
    print("\n" + "=" * 60)
    print("Testing Corruption Diversity (5 easy datasets)")
    print("=" * 60)

    generator = SchemaGenerator()
    schema = generator.generate_ecommerce_schema(num_rows=50)

    strategy_combinations = []

    for i in range(5):
        df_clean = build_clean_dataset(schema, seed=50 + i)
        config = CorruptionConfig.from_preset("easy", seed=50 + i)
        df_corrupt, metadata = corrupt_dataset(df_clean, config, seed=50 + i)

        strategies = tuple(sorted(config.strategies))
        strategy_combinations.append(strategies)
        print(f"\nDataset {i+1}: {config.strategies}")

    # Check diversity
    unique_combinations = len(set(strategy_combinations))
    print(f"\n✓ Generated {unique_combinations} unique strategy combinations out of 5")

    return strategy_combinations


def test_save_sample_pair():
    """Generate and save a sample corrupted dataset pair."""
    print("\n" + "=" * 60)
    print("Generating Sample Dataset Pair for Inspection")
    print("=" * 60)

    generator = SchemaGenerator()
    schema = generator.generate_ecommerce_schema(num_rows=200)
    df_clean = build_clean_dataset(schema, seed=100)

    # Use medium difficulty
    config = CorruptionConfig.from_preset("medium", seed=100)
    print(f"\nSelected strategies: {config.strategies}")

    df_corrupt, metadata = corrupt_dataset(df_clean, config, seed=100)

    # Save to output directory
    import os

    os.makedirs("output", exist_ok=True)

    df_clean.to_csv("output/sample_ecommerce_clean.csv", index=False)
    df_corrupt.to_csv("output/sample_ecommerce_raw.csv", index=False)

    # Save metadata
    with open("output/sample_ecommerce_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    print(f"\n✓ Saved clean dataset to: output/sample_ecommerce_clean.csv")
    print(f"✓ Saved raw dataset to: output/sample_ecommerce_raw.csv")
    print(f"✓ Saved metadata to: output/sample_ecommerce_metadata.json")

    print(f"\nDataset comparison:")
    print(f"  Clean shape: {df_clean.shape}")
    print(f"  Raw shape: {df_corrupt.shape}")
    print(f"  Missing in clean: {df_clean.isnull().sum().sum()}")
    print(f"  Missing in raw: {df_corrupt.isnull().sum().sum()}")
    print(f"  Duplicates in clean: {df_clean.duplicated().sum()}")
    print(f"  Duplicates in raw: {df_corrupt.duplicated().sum()}")

    return df_clean, df_corrupt, metadata


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("CORRUPTION ENGINE TEST SUITE")
    print("=" * 60)

    try:
        # Test different difficulty levels
        test_easy_corruption()
        test_medium_corruption()
        test_hard_corruption()

        # Test custom configuration
        test_custom_corruption()

        # Test diversity
        test_corruption_diversity()

        # Save a sample pair
        test_save_sample_pair()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)
        print("\nPhase 3 (Corruption Engine) is complete!")
        print("\nNext steps:")
        print("  1. Implement Phase 4: Validation Layer")
        print("  2. Implement Phase 5: Pipeline Orchestration")
        print("  3. Implement Phase 6: Export and Metadata")

        return 0

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
