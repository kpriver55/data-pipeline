"""
Test script for schema system.

Run with: python test_schema.py
"""

import json

from schema.schema import ColumnType, NumericDistribution
from schema.schema_generator import SchemaGenerator


def test_ecommerce_schema():
    """Test e-commerce schema generation."""
    print("=" * 60)
    print("Testing E-commerce Schema Generation")
    print("=" * 60)

    generator = SchemaGenerator()
    schema = generator.generate_ecommerce_schema(num_rows=5000)

    print(f"\nDataset: {schema.name}")
    print(f"Domain: {schema.domain}")
    print(f"Number of rows: {schema.num_rows}")
    print(f"Number of columns: {len(schema.columns)}")

    print("\nColumns:")
    for col in schema.columns:
        print(f"  - {col.name} ({col.type.value})")
        if col.description:
            print(f"    Description: {col.description}")

    # Test column retrieval
    order_amount = schema.get_column("order_amount")
    assert order_amount is not None
    assert order_amount.type == ColumnType.NUMERIC_FLOAT
    print(f"\n✓ Column retrieval test passed")

    # Test columns by type
    numeric_cols = schema.get_columns_by_type(ColumnType.NUMERIC_INT)
    print(f"✓ Found {len(numeric_cols)} integer columns")

    categorical_cols = schema.get_columns_by_type(ColumnType.CATEGORICAL)
    print(f"✓ Found {len(categorical_cols)} categorical columns")

    datetime_cols = schema.get_columns_by_type(ColumnType.DATETIME)
    print(f"✓ Found {len(datetime_cols)} datetime columns")

    return schema


def test_healthcare_schema():
    """Test healthcare schema generation."""
    print("\n" + "=" * 60)
    print("Testing Healthcare Schema Generation")
    print("=" * 60)

    generator = SchemaGenerator()
    schema = generator.generate_healthcare_schema(num_rows=3000)

    print(f"\nDataset: {schema.name}")
    print(f"Domain: {schema.domain}")
    print(f"Number of columns: {len(schema.columns)}")

    print("\nSample columns:")
    for col in schema.columns[:5]:
        print(f"  - {col.name} ({col.type.value})")

    return schema


def test_finance_schema():
    """Test finance schema generation."""
    print("\n" + "=" * 60)
    print("Testing Finance Schema Generation")
    print("=" * 60)

    generator = SchemaGenerator()
    schema = generator.generate_finance_schema(num_rows=8000)

    print(f"\nDataset: {schema.name}")
    print(f"Domain: {schema.domain}")
    print(f"Number of columns: {len(schema.columns)}")

    return schema


def test_custom_schema():
    """Test custom schema generation."""
    print("\n" + "=" * 60)
    print("Testing Custom Schema Generation")
    print("=" * 60)

    generator = SchemaGenerator()

    column_specs = [
        {
            "name": "user_id",
            "type": "numeric_int",
            "min": 1000,
            "max": 9999,
            "distribution": "uniform",
        },
        {
            "name": "username",
            "type": "categorical",
            "values": ["alice", "bob", "charlie", "diana", "eve"],
        },
        {
            "name": "score",
            "type": "numeric_float",
            "min": 0.0,
            "max": 100.0,
            "mean": 75.0,
            "std": 15.0,
            "distribution": "normal",
            "decimals": 1,
        },
        {
            "name": "signup_date",
            "type": "datetime",
            "start_date": "2024-01-01",
            "end_date": "2024-12-31",
            "include_time": False,
        },
        {
            "name": "account_type",
            "type": "categorical",
            "values": ["free", "premium", "enterprise"],
            "frequencies": [0.6, 0.3, 0.1],
        },
    ]

    schema = generator.generate_custom_schema(
        name="user_accounts",
        domain="saas",
        column_specs=column_specs,
        num_rows=2000,
        description="Custom SaaS user accounts dataset",
    )

    print(f"\nDataset: {schema.name}")
    print(f"Domain: {schema.domain}")
    print(f"Description: {schema.description}")
    print(f"Number of columns: {len(schema.columns)}")

    print("\nColumns:")
    for col in schema.columns:
        print(f"  - {col.name} ({col.type.value})")

    return schema


def test_schema_serialization():
    """Test schema serialization to/from dict."""
    print("\n" + "=" * 60)
    print("Testing Schema Serialization")
    print("=" * 60)

    generator = SchemaGenerator()
    original_schema = generator.generate_ecommerce_schema(num_rows=1000)

    # Convert to dict
    schema_dict = original_schema.to_dict()
    print(f"\n✓ Converted schema to dictionary")
    print(f"  Keys: {list(schema_dict.keys())}")

    # Convert back to schema
    restored_schema = type(original_schema).from_dict(schema_dict)
    print(f"✓ Restored schema from dictionary")
    print(f"  Name: {restored_schema.name}")
    print(f"  Columns: {len(restored_schema.columns)}")

    # Verify they match
    assert original_schema.name == restored_schema.name
    assert len(original_schema.columns) == len(restored_schema.columns)
    print(f"✓ Serialization test passed")

    # Test JSON serialization
    json_str = json.dumps(schema_dict, indent=2)
    print(f"\n✓ Schema can be serialized to JSON ({len(json_str)} bytes)")

    return schema_dict


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("SCHEMA SYSTEM TEST SUITE")
    print("=" * 60)

    try:
        # Test pre-built schemas
        ecommerce_schema = test_ecommerce_schema()
        healthcare_schema = test_healthcare_schema()
        finance_schema = test_finance_schema()

        # Test custom schema
        custom_schema = test_custom_schema()

        # Test serialization
        schema_dict = test_schema_serialization()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)
        print("\nPhase 1 (Schema Definition System) is complete!")
        print("\nNext steps:")
        print("  1. Implement Phase 2: Clean Data Generation")
        print("  2. Use these schemas to generate clean datasets")

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
