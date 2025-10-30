"""
Test script for Phase 2: Clean Data Generation

Run with: python test_clean_generation.py
"""

import pandas as pd

from generators.clean_data_builder import build_clean_dataset
from schema.schema_generator import SchemaGenerator


def test_ecommerce_generation():
    """Test clean data generation for e-commerce dataset."""
    print("=" * 60)
    print("Testing E-commerce Clean Data Generation")
    print("=" * 60)

    # Generate schema
    generator = SchemaGenerator()
    schema = generator.generate_ecommerce_schema(num_rows=1000)

    # Build clean dataset
    print(f"\nGenerating {schema.num_rows} rows...")
    df = build_clean_dataset(schema, seed=42)

    print(f"\nDataset shape: {df.shape}")
    print(f"\nFirst 5 rows:")
    print(df.head())

    print(f"\nData types:")
    print(df.dtypes)

    print(f"\nMissing values:")
    print(df.isnull().sum())

    print(f"\nBasic statistics for numeric columns:")
    print(df.describe())

    # Check categorical value distribution
    print(f"\nProduct category distribution:")
    print(df["product_category"].value_counts(normalize=True))

    print(f"\nOrder status distribution:")
    print(df["order_status"].value_counts(normalize=True))

    return df


def test_healthcare_generation():
    """Test clean data generation for healthcare dataset."""
    print("\n" + "=" * 60)
    print("Testing Healthcare Clean Data Generation")
    print("=" * 60)

    generator = SchemaGenerator()
    schema = generator.generate_healthcare_schema(num_rows=500)

    print(f"\nGenerating {schema.num_rows} rows...")
    df = build_clean_dataset(schema, seed=42)

    print(f"\nDataset shape: {df.shape}")
    print(f"\nFirst 5 rows:")
    print(df.head())

    print(f"\nAge statistics:")
    print(df["age"].describe())

    print(f"\nGender distribution:")
    print(df["gender"].value_counts(normalize=True))

    print(f"\nDiagnosis distribution:")
    print(df["diagnosis"].value_counts(normalize=True))

    # Check datetime generation
    print(f"\nVisit date range:")
    print(f"  Min: {df['visit_date'].min()}")
    print(f"  Max: {df['visit_date'].max()}")

    # Check business hours constraint
    print(f"\nVisit hours distribution (should be 9-16 for business hours):")
    print(df["visit_date"].dt.hour.value_counts().sort_index())

    # Check weekdays constraint
    print(f"\nVisit days of week (0=Mon, 6=Sun, should not have 5 or 6):")
    print(df["visit_date"].dt.dayofweek.value_counts().sort_index())

    return df


def test_finance_generation():
    """Test clean data generation for finance dataset."""
    print("\n" + "=" * 60)
    print("Testing Finance Clean Data Generation")
    print("=" * 60)

    generator = SchemaGenerator()
    schema = generator.generate_finance_schema(num_rows=800)

    print(f"\nGenerating {schema.num_rows} rows...")
    df = build_clean_dataset(schema, seed=42)

    print(f"\nDataset shape: {df.shape}")
    print(f"\nFirst 5 rows:")
    print(df.head())

    print(f"\nTransaction amount statistics:")
    print(df["transaction_amount"].describe())

    print(f"\nTransaction type distribution:")
    print(df["transaction_type"].value_counts(normalize=True))

    print(f"\nTransaction status distribution:")
    print(df["transaction_status"].value_counts(normalize=True))

    return df


def test_custom_generation():
    """Test clean data generation with custom schema."""
    print("\n" + "=" * 60)
    print("Testing Custom Schema Clean Data Generation")
    print("=" * 60)

    generator = SchemaGenerator()

    column_specs = [
        {
            "name": "employee_id",
            "type": "numeric_int",
            "min": 1000,
            "max": 9999,
            "distribution": "uniform",
        },
        {
            "name": "department",
            "type": "categorical",
            "values": ["Engineering", "Sales", "Marketing", "HR", "Finance"],
            "frequencies": [0.35, 0.25, 0.20, 0.10, 0.10],
        },
        {
            "name": "salary",
            "type": "numeric_float",
            "min": 40000.0,
            "max": 200000.0,
            "mean": 75000.0,
            "std": 25000.0,
            "distribution": "normal",
            "decimals": 2,
        },
        {
            "name": "hire_date",
            "type": "datetime",
            "start_date": "2015-01-01",
            "end_date": "2024-12-31",
            "include_time": False,
        },
        {
            "name": "performance_rating",
            "type": "categorical",
            "values": ["Excellent", "Good", "Average", "Below Average"],
            "frequencies": [0.20, 0.40, 0.30, 0.10],
        },
    ]

    schema = generator.generate_custom_schema(
        name="employee_data",
        domain="hr",
        column_specs=column_specs,
        num_rows=300,
    )

    print(f"\nGenerating {schema.num_rows} rows...")
    df = build_clean_dataset(schema, seed=42)

    print(f"\nDataset shape: {df.shape}")
    print(f"\nFirst 5 rows:")
    print(df.head())

    print(f"\nDepartment distribution:")
    print(df["department"].value_counts(normalize=True))

    print(f"\nSalary statistics:")
    print(df["salary"].describe())

    print(f"\nPerformance rating distribution:")
    print(df["performance_rating"].value_counts(normalize=True))

    return df


def test_data_quality():
    """Test that generated data meets quality standards."""
    print("\n" + "=" * 60)
    print("Testing Data Quality Standards")
    print("=" * 60)

    generator = SchemaGenerator()
    schema = generator.generate_ecommerce_schema(num_rows=1000)
    df = build_clean_dataset(schema, seed=42)

    # Test 1: No missing values
    missing_count = df.isnull().sum().sum()
    assert missing_count == 0, f"Found {missing_count} missing values (should be 0)"
    print("✓ No missing values")

    # Test 2: Correct shape
    assert df.shape == (1000, 10), f"Shape mismatch: {df.shape}"
    print("✓ Correct shape")

    # Test 3: Numeric ranges
    assert df["customer_age"].min() >= 18, "Age below minimum"
    assert df["customer_age"].max() <= 80, "Age above maximum"
    print("✓ Numeric values within range")

    # Test 4: Categorical values
    valid_statuses = {"New", "Regular", "Premium", "VIP"}
    assert set(df["customer_status"].unique()).issubset(
        valid_statuses
    ), "Invalid categorical values"
    print("✓ Categorical values valid")

    # Test 5: Datetime range
    assert df["order_date"].min() >= pd.Timestamp("2023-01-01"), "Date below minimum"
    assert df["order_date"].max() <= pd.Timestamp("2024-12-31"), "Date above maximum"
    print("✓ Datetime values within range")

    print("\n✓ All quality checks passed!")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("CLEAN DATA GENERATION TEST SUITE")
    print("=" * 60)

    try:
        # Test different domain schemas
        ecommerce_df = test_ecommerce_generation()
        healthcare_df = test_healthcare_generation()
        finance_df = test_finance_generation()
        custom_df = test_custom_generation()

        # Test data quality
        test_data_quality()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)
        print("\nPhase 2 (Clean Data Generation) is complete!")
        print("\nNext steps:")
        print("  1. Implement Phase 3: Corruption Engine")
        print("  2. Generate dirty versions of clean datasets")

        return 0

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
