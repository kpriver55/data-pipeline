"""
Test script for Phase 4: Validation Layer

Tests both consistency and solvability checkers to ensure:
1. Recovered clean datasets are consistent with dirty datasets
2. All corruptions are solvable by the cleaning agent

Run with: python3 test_validation.py
"""

from corruption.base import CorruptionConfig
from corruption.orchestrator import corrupt_dataset
from generators.clean_data_builder import build_clean_dataset
from schema.schema_generator import SchemaGenerator
from validation.consistency_checker import validate_dataset_pair
from validation.solvability_checker import validate_solvability


def test_consistency_validation():
    """Test that consistency validation works correctly."""
    print("=" * 60)
    print("Testing Consistency Validation")
    print("=" * 60)

    # Generate a dataset pair
    generator = SchemaGenerator()
    schema = generator.generate_ecommerce_schema(num_rows=100)
    df_perfect = build_clean_dataset(schema, seed=50)

    config = CorruptionConfig.from_preset("medium", seed=50)
    df_dirty, df_recovered, metadata = corrupt_dataset(df_perfect, config, seed=50)

    print(f"\nDataset shapes:")
    print(f"  Perfect: {df_perfect.shape}")
    print(f"  Dirty: {df_dirty.shape}")
    print(f"  Recovered: {df_recovered.shape}")

    # Validate consistency
    is_valid = validate_dataset_pair(df_dirty, df_recovered, metadata, verbose=True)

    assert is_valid, "Consistency validation should pass for properly generated datasets"

    print("\n✓ Consistency validation test passed")


def test_solvability_validation():
    """Test that solvability validation works correctly."""
    print("\n" + "=" * 60)
    print("Testing Solvability Validation")
    print("=" * 60)

    # Generate a dataset with multiple corruption types
    generator = SchemaGenerator()
    schema = generator.generate_ecommerce_schema(num_rows=100)
    df_perfect = build_clean_dataset(schema, seed=51)

    config = CorruptionConfig.from_preset("hard", seed=51)
    df_dirty, df_recovered, metadata = corrupt_dataset(df_perfect, config, seed=51)

    print(f"\nApplied strategies: {metadata['applied_strategies']}")
    print(f"Cleaning operations: {len(metadata['cleaning_operations'])}")

    # Validate solvability
    is_solvable = validate_solvability(df_dirty, metadata, verbose=True)

    assert is_solvable, "Solvability validation should pass for agent-compatible corruptions"

    print("\n✓ Solvability validation test passed")


def test_validation_across_difficulties():
    """Test validation across all difficulty levels."""
    print("\n" + "=" * 60)
    print("Testing Validation Across Difficulty Levels")
    print("=" * 60)

    generator = SchemaGenerator()
    schema = generator.generate_ecommerce_schema(num_rows=100)

    for difficulty in ["easy", "medium", "hard"]:
        print(f"\n--- {difficulty.upper()} Difficulty ---")

        df_perfect = build_clean_dataset(schema, seed=52 + ord(difficulty[0]))
        config = CorruptionConfig.from_preset(difficulty, seed=52 + ord(difficulty[0]))

        df_dirty, df_recovered, metadata = corrupt_dataset(
            df_perfect, config, seed=52
        )

        # Validate both consistency and solvability
        print(f"\nConsistency check:")
        is_consistent = validate_dataset_pair(
            df_dirty, df_recovered, metadata, verbose=False
        )

        print(f"\nSolvability check:")
        is_solvable = validate_solvability(df_dirty, metadata, verbose=False)

        print(f"\nResults: Consistent={is_consistent}, Solvable={is_solvable}")

        assert is_consistent, f"{difficulty} difficulty should pass consistency validation"
        assert is_solvable, f"{difficulty} difficulty should pass solvability validation"

    print("\n✓ All difficulty levels passed validation")


def test_metadata_format():
    """Test that metadata is in the correct format for the cleaning agent."""
    print("\n" + "=" * 60)
    print("Testing Metadata Format")
    print("=" * 60)

    # Generate a dataset
    generator = SchemaGenerator()
    schema = generator.generate_ecommerce_schema(num_rows=100)
    df_perfect = build_clean_dataset(schema, seed=53)

    config = CorruptionConfig.from_preset("medium", seed=53)
    df_dirty, df_recovered, metadata = corrupt_dataset(df_perfect, config, seed=53)

    # Check metadata structure
    print("\nMetadata keys:")
    for key in metadata.keys():
        print(f"  - {key}")

    assert "cleaning_operations" in metadata, "Metadata should have cleaning_operations"
    assert "recovery_summary" in metadata, "Metadata should have recovery_summary"

    # Check cleaning operations format (agent-compatible)
    print("\nCleaning operations format:")
    for op in metadata["cleaning_operations"]:
        print(f"\n  Operation: {op['operation']}")
        print(f"    Rationale: {op.get('rationale', 'N/A')}")

        # Check that all operations have required fields
        assert "operation" in op, "Each operation should have 'operation' field"
        assert "rationale" in op, "Each operation should have 'rationale' field"

        # Parameters should be flattened (not in a 'parameters' sub-dict)
        assert "parameters" not in op, "Parameters should be flattened, not in a sub-dict"

    print("\n✓ Metadata format test passed")


def main():
    """Run all validation tests."""
    print("\n" + "=" * 60)
    print("PHASE 4: VALIDATION LAYER TESTS")
    print("=" * 60)

    try:
        # Test consistency validation
        test_consistency_validation()

        # Test solvability validation
        test_solvability_validation()

        # Test across difficulty levels
        test_validation_across_difficulties()

        # Test metadata format
        test_metadata_format()

        print("\n" + "=" * 60)
        print("ALL VALIDATION TESTS PASSED ✓")
        print("=" * 60)
        print("\nPhase 4 (Validation Layer) is complete!")
        print("\nKey Features:")
        print("  - Consistency checker verifies dirty→recovered clean integrity")
        print("  - Solvability checker ensures all corruptions are agent-fixable")
        print("  - Metadata format matches cleaning agent expectations")

        return 0

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
