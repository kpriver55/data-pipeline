"""
Generate a sample clean dataset for inspection.

This script generates a small e-commerce dataset and saves it to CSV.
"""

import os

from generators.clean_data_builder import build_clean_dataset
from schema.schema_generator import SchemaGenerator


def main():
    print("Generating sample e-commerce dataset...")
    print("=" * 60)

    # Create output directory if needed
    os.makedirs("output", exist_ok=True)

    # Generate schema (default 1000 rows)
    generator = SchemaGenerator()
    schema = generator.generate_ecommerce_schema()

    print(f"Schema: {schema.name}")
    print(f"Domain: {schema.domain}")
    print(f"Rows: {schema.num_rows}")
    print(f"Columns: {len(schema.columns)}")

    # Generate clean data
    print("\nGenerating clean data...")
    df = build_clean_dataset(schema, seed=42)

    # Save to CSV
    output_path = "output/sample_ecommerce_clean.csv"
    df.to_csv(output_path, index=False)
    print(f"\n✓ Saved to: {output_path}")

    # Display summary
    print("\n" + "=" * 60)
    print("Dataset Summary")
    print("=" * 60)
    print(f"\nShape: {df.shape}")
    print(f"\nFirst 10 rows:")
    print(df.head(10).to_string())

    print(f"\nData types:")
    print(df.dtypes)

    print(f"\nMissing values:")
    print(df.isnull().sum())

    print(f"\nNumeric column statistics:")
    print(df.describe())

    print(f"\nCategorical distributions:")
    for col in df.select_dtypes(include=["object"]).columns:
        print(f"\n{col}:")
        print(df[col].value_counts().head(10))

    print("\n" + "=" * 60)
    print("✓ Sample generation complete!")
    print(f"Review the file at: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
