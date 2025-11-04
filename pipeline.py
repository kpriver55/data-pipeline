"""
Main pipeline orchestrator for synthetic dataset generation.

Provides high-level API for generating dirty/clean dataset pairs with
proper validation and metadata generation.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

from config.dataset_configs import DatasetConfig, get_config
from corruption.base import CorruptionConfig
from corruption.orchestrator import corrupt_dataset
from generators.clean_data_builder import build_clean_dataset
from schema.schema import DatasetSchema
from schema.schema_generator import SchemaGenerator
from validation.consistency_checker import validate_dataset_pair
from validation.solvability_checker import validate_solvability


class SyntheticDataPipeline:
    """
    Main pipeline for generating synthetic dirty/clean dataset pairs.

    This orchestrator ties together all phases:
    - Schema generation
    - Clean data generation
    - Corruption and recovered clean generation
    - Validation
    - Export (metadata and files)
    """

    def __init__(self, validate: bool = True, verbose: bool = True):
        """
        Initialize pipeline.

        Args:
            validate: Whether to run validation checks
            verbose: Whether to print progress messages
        """
        self.validate = validate
        self.verbose = verbose
        self.schema_generator = SchemaGenerator()

    def generate(
        self,
        config: Union[str, DatasetConfig, Dict[str, Any]],
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        """
        Generate a dirty/recovered clean dataset pair.

        Args:
            config: Configuration (name, DatasetConfig object, or dict)
            output_dir: Optional directory to save outputs

        Returns:
            Tuple of (df_dirty, df_recovered_clean, metadata)
        """
        # Parse configuration
        if isinstance(config, str):
            dataset_config = get_config(config)
        elif isinstance(config, dict):
            dataset_config = DatasetConfig.from_dict(config)
        else:
            dataset_config = config

        if self.verbose:
            print("\n" + "=" * 70)
            print(f"GENERATING DATASET: {dataset_config.name}")
            print("=" * 70)
            print(f"Domain: {dataset_config.domain}")
            print(f"Rows: {dataset_config.num_rows}")
            print(f"Difficulty: {dataset_config.difficulty}")
            if dataset_config.description:
                print(f"Description: {dataset_config.description}")

        # Step 1: Generate schema
        schema = self._generate_schema(dataset_config)

        # Step 2: Generate perfect synthetic data
        df_perfect = self._generate_clean_data(schema, dataset_config)

        # Step 3: Apply corruption and generate recovered clean
        df_dirty, df_recovered, metadata = self._apply_corruption(
            df_perfect, dataset_config
        )

        # Add dataset config to metadata
        metadata["dataset_config"] = dataset_config.to_dict()

        # Step 4: Validate (if enabled)
        if self.validate:
            self._validate_datasets(df_dirty, df_recovered, metadata)

        # Step 5: Export (if output_dir specified)
        if output_dir:
            self._export_datasets(
                df_dirty, df_recovered, metadata, dataset_config, output_dir
            )

        if self.verbose:
            print("\n" + "=" * 70)
            print("✓ DATASET GENERATION COMPLETE")
            print("=" * 70)

        return df_dirty, df_recovered, metadata

    def generate_batch(
        self,
        configs: List[Union[str, DatasetConfig, Dict[str, Any]]],
        output_dir: Optional[Union[str, Path]] = None,
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]]:
        """
        Generate multiple dataset pairs in batch.

        Args:
            configs: List of configurations
            output_dir: Optional base directory for outputs

        Returns:
            List of (df_dirty, df_recovered_clean, metadata) tuples
        """
        results = []

        if self.verbose:
            print(f"\n{'=' * 70}")
            print(f"BATCH GENERATION: {len(configs)} datasets")
            print("=" * 70)

        for i, config in enumerate(configs, 1):
            if self.verbose:
                print(f"\n[{i}/{len(configs)}]")

            # Create subdirectory for each dataset if output_dir specified
            if output_dir:
                if isinstance(config, str):
                    config_name = config
                elif isinstance(config, dict):
                    config_name = config.get("name", f"dataset_{i}")
                else:
                    config_name = config.name

                dataset_output_dir = Path(output_dir) / config_name
            else:
                dataset_output_dir = None

            result = self.generate(config, output_dir=dataset_output_dir)
            results.append(result)

        if self.verbose:
            print(f"\n{'=' * 70}")
            print(f"✓ BATCH GENERATION COMPLETE: {len(results)} datasets")
            print("=" * 70)

        return results

    def _generate_schema(self, config: DatasetConfig) -> DatasetSchema:
        """Generate schema based on domain."""
        if self.verbose:
            print(f"\n[1/5] Generating schema...")

        # Use pre-built schema generators
        if config.domain == "ecommerce":
            schema = self.schema_generator.generate_ecommerce_schema(
                num_rows=config.num_rows
            )
        elif config.domain == "healthcare":
            schema = self.schema_generator.generate_healthcare_schema(
                num_rows=config.num_rows
            )
        elif config.domain == "finance":
            schema = self.schema_generator.generate_finance_schema(
                num_rows=config.num_rows
            )
        else:
            raise ValueError(
                f"Unknown domain '{config.domain}'. "
                "Supported: ecommerce, healthcare, finance"
            )

        if self.verbose:
            print(f"  ✓ Schema created: {len(schema.columns)} columns")

        return schema

    def _generate_clean_data(
        self, schema: DatasetSchema, config: DatasetConfig
    ) -> pd.DataFrame:
        """Generate perfect synthetic clean data."""
        if self.verbose:
            print(f"\n[2/5] Generating perfect synthetic data...")

        df_perfect = build_clean_dataset(schema, seed=config.seed)

        if self.verbose:
            print(f"  ✓ Perfect data generated: {df_perfect.shape}")

        return df_perfect

    def _apply_corruption(
        self, df_perfect: pd.DataFrame, config: DatasetConfig
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        """Apply corruption and generate recovered clean dataset."""
        if self.verbose:
            print(f"\n[3/5] Applying corruption...")

        # Create corruption config
        if config.strategies:
            # Custom strategies specified
            corruption_config = CorruptionConfig.custom(
                strategies=config.strategies, seed=config.seed
            )
        else:
            # Use difficulty preset
            corruption_config = CorruptionConfig.from_preset(
                config.difficulty, seed=config.seed
            )

        # Apply corruption
        df_dirty, df_recovered, metadata = corrupt_dataset(
            df_perfect, corruption_config, seed=config.seed
        )

        if self.verbose:
            print(f"  ✓ Corruption applied")
            print(f"    Strategies: {metadata['applied_strategies']}")
            print(f"    Dirty shape: {df_dirty.shape}")
            print(f"    Recovered shape: {df_recovered.shape}")

        return df_dirty, df_recovered, metadata

    def _validate_datasets(
        self,
        df_dirty: pd.DataFrame,
        df_recovered: pd.DataFrame,
        metadata: Dict[str, Any],
    ):
        """Run validation checks."""
        if self.verbose:
            print(f"\n[4/5] Running validation...")

        # Consistency validation
        is_consistent = validate_dataset_pair(
            df_dirty, df_recovered, metadata, verbose=False
        )

        # Solvability validation
        is_solvable = validate_solvability(
            df_dirty, metadata, verbose=False
        )

        if self.verbose:
            if is_consistent and is_solvable:
                print(f"  ✓ Validation passed")
            else:
                if not is_consistent:
                    print(f"  ✗ Consistency validation failed")
                if not is_solvable:
                    print(f"  ✗ Solvability validation failed")

        if not (is_consistent and is_solvable):
            # Run with verbose to show details
            print("\nValidation Details:")
            validate_dataset_pair(df_dirty, df_recovered, metadata, verbose=True)
            validate_solvability(df_dirty, metadata, verbose=True)

            raise RuntimeError(
                "Validation failed. See details above."
            )

    def _export_datasets(
        self,
        df_dirty: pd.DataFrame,
        df_recovered: pd.DataFrame,
        metadata: Dict[str, Any],
        config: DatasetConfig,
        output_dir: Union[str, Path],
    ):
        """Export datasets and metadata to files."""
        if self.verbose:
            print(f"\n[5/5] Exporting to {output_dir}...")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save dirty dataset
        dirty_path = output_path / "dirty.csv"
        df_dirty.to_csv(dirty_path, index=False)

        # Save recovered clean dataset (the actual target)
        clean_path = output_path / "clean.csv"
        df_recovered.to_csv(clean_path, index=False)

        # Save metadata
        import json

        metadata_path = output_path / "metadata.json"

        # Metadata already has dataset_config added earlier
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        if self.verbose:
            print(f"  ✓ Files saved:")
            print(f"    - {dirty_path}")
            print(f"    - {clean_path}")
            print(f"    - {metadata_path}")


# =============================================================================
# Convenience Functions
# =============================================================================


def generate_dataset(
    config: Union[str, DatasetConfig, Dict[str, Any]],
    output_dir: Optional[Union[str, Path]] = None,
    validate: bool = True,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Convenience function to generate a single dataset pair.

    Args:
        config: Configuration (name, DatasetConfig object, or dict)
        output_dir: Optional directory to save outputs
        validate: Whether to run validation checks
        verbose: Whether to print progress

    Returns:
        Tuple of (df_dirty, df_recovered_clean, metadata)

    Example:
        >>> df_dirty, df_clean, metadata = generate_dataset("ecommerce_medium")
        >>> df_dirty, df_clean, metadata = generate_dataset(
        ...     "ecommerce_large_hard",
        ...     output_dir="output/datasets/ecommerce_1"
        ... )
    """
    pipeline = SyntheticDataPipeline(validate=validate, verbose=verbose)
    return pipeline.generate(config, output_dir=output_dir)


def generate_batch_datasets(
    configs: List[Union[str, DatasetConfig, Dict[str, Any]]],
    output_dir: Optional[Union[str, Path]] = None,
    validate: bool = True,
    verbose: bool = True,
) -> List[Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]]:
    """
    Convenience function to generate multiple dataset pairs.

    Args:
        configs: List of configurations
        output_dir: Optional base directory for outputs
        validate: Whether to run validation checks
        verbose: Whether to print progress

    Returns:
        List of (df_dirty, df_recovered_clean, metadata) tuples

    Example:
        >>> results = generate_batch_datasets([
        ...     "ecommerce_small_easy",
        ...     "ecommerce_medium",
        ...     "ecommerce_large_hard"
        ... ], output_dir="output/training_set")
    """
    pipeline = SyntheticDataPipeline(validate=validate, verbose=verbose)
    return pipeline.generate_batch(configs, output_dir=output_dir)


def generate_training_set(
    domain: str = "ecommerce",
    num_easy: int = 5,
    num_medium: int = 5,
    num_hard: int = 5,
    rows_per_dataset: int = 1000,
    output_dir: Optional[Union[str, Path]] = None,
    base_seed: int = 42,
) -> List[Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]]:
    """
    Generate a training set with multiple difficulty levels.

    Args:
        domain: Domain for datasets (ecommerce, healthcare, finance)
        num_easy: Number of easy datasets
        num_medium: Number of medium datasets
        num_hard: Number of hard datasets
        rows_per_dataset: Rows per dataset
        output_dir: Optional output directory
        base_seed: Base random seed (incremented for each dataset)

    Returns:
        List of (df_dirty, df_recovered_clean, metadata) tuples

    Example:
        >>> training_set = generate_training_set(
        ...     domain="ecommerce",
        ...     num_easy=10,
        ...     num_medium=10,
        ...     num_hard=10,
        ...     output_dir="output/training_data"
        ... )
    """
    configs = []
    seed = base_seed

    # Easy datasets
    for i in range(num_easy):
        configs.append(
            DatasetConfig(
                name=f"{domain}_easy_{i+1}",
                domain=domain,
                num_rows=rows_per_dataset,
                difficulty="easy",
                seed=seed,
                description=f"Easy training example {i+1}",
            )
        )
        seed += 1

    # Medium datasets
    for i in range(num_medium):
        configs.append(
            DatasetConfig(
                name=f"{domain}_medium_{i+1}",
                domain=domain,
                num_rows=rows_per_dataset,
                difficulty="medium",
                seed=seed,
                description=f"Medium training example {i+1}",
            )
        )
        seed += 1

    # Hard datasets
    for i in range(num_hard):
        configs.append(
            DatasetConfig(
                name=f"{domain}_hard_{i+1}",
                domain=domain,
                num_rows=rows_per_dataset,
                difficulty="hard",
                seed=seed,
                description=f"Hard training example {i+1}",
            )
        )
        seed += 1

    return generate_batch_datasets(configs, output_dir=output_dir)
