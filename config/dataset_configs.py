"""
Pre-defined dataset configurations for common use cases.

These configurations provide ready-to-use settings for generating
dataset pairs across different domains and difficulty levels.
"""

from typing import Any, Dict, List, Optional


class DatasetConfig:
    """Configuration for dataset generation."""

    def __init__(
        self,
        name: str,
        domain: str,
        num_rows: int = 1000,
        difficulty: str = "medium",
        strategies: Optional[List[str]] = None,
        seed: Optional[int] = None,
        description: Optional[str] = None,
    ):
        """
        Initialize dataset configuration.

        Args:
            name: Configuration name
            domain: Domain for schema generation (ecommerce, healthcare, finance)
            num_rows: Number of rows to generate
            difficulty: Corruption difficulty (easy, medium, hard)
            strategies: Specific corruption strategies (None for random selection)
            seed: Random seed for reproducibility
            description: Human-readable description
        """
        self.name = name
        self.domain = domain
        self.num_rows = num_rows
        self.difficulty = difficulty
        self.strategies = strategies
        self.seed = seed
        self.description = description

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "domain": self.domain,
            "num_rows": self.num_rows,
            "difficulty": self.difficulty,
            "strategies": self.strategies,
            "seed": self.seed,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetConfig":
        """Create from dictionary."""
        return cls(**data)


# =============================================================================
# E-Commerce Configurations
# =============================================================================

ECOMMERCE_SMALL_EASY = DatasetConfig(
    name="ecommerce_small_easy",
    domain="ecommerce",
    num_rows=500,
    difficulty="easy",
    description="Small e-commerce dataset with 1-2 simple cleaning operations",
)

ECOMMERCE_MEDIUM = DatasetConfig(
    name="ecommerce_medium",
    domain="ecommerce",
    num_rows=1000,
    difficulty="medium",
    description="Medium e-commerce dataset with 3-4 cleaning operations",
)

ECOMMERCE_LARGE_HARD = DatasetConfig(
    name="ecommerce_large_hard",
    domain="ecommerce",
    num_rows=5000,
    difficulty="hard",
    description="Large e-commerce dataset with 5-7 complex cleaning operations",
)

# Specific corruption types for targeted testing
ECOMMERCE_TEXT_ONLY = DatasetConfig(
    name="ecommerce_text_only",
    domain="ecommerce",
    num_rows=1000,
    difficulty="custom",
    strategies=["text_whitespace", "text_case", "text_special_chars"],
    description="E-commerce dataset with only text corruption issues",
)

ECOMMERCE_NUMERIC_ONLY = DatasetConfig(
    name="ecommerce_numeric_only",
    domain="ecommerce",
    num_rows=1000,
    difficulty="custom",
    strategies=["missing_values", "outliers", "type_corruption"],
    description="E-commerce dataset with only numeric/missing value issues",
)

# =============================================================================
# Healthcare Configurations
# =============================================================================

HEALTHCARE_SMALL_EASY = DatasetConfig(
    name="healthcare_small_easy",
    domain="healthcare",
    num_rows=500,
    difficulty="easy",
    description="Small healthcare dataset with 1-2 simple cleaning operations",
)

HEALTHCARE_MEDIUM = DatasetConfig(
    name="healthcare_medium",
    domain="healthcare",
    num_rows=1000,
    difficulty="medium",
    description="Medium healthcare dataset with 3-4 cleaning operations",
)

HEALTHCARE_LARGE_HARD = DatasetConfig(
    name="healthcare_large_hard",
    domain="healthcare",
    num_rows=5000,
    difficulty="hard",
    description="Large healthcare dataset with 5-7 complex cleaning operations",
)

# =============================================================================
# Finance Configurations
# =============================================================================

FINANCE_SMALL_EASY = DatasetConfig(
    name="finance_small_easy",
    domain="finance",
    num_rows=500,
    difficulty="easy",
    description="Small finance dataset with 1-2 simple cleaning operations",
)

FINANCE_MEDIUM = DatasetConfig(
    name="finance_medium",
    domain="finance",
    num_rows=1000,
    difficulty="medium",
    description="Medium finance dataset with 3-4 cleaning operations",
)

FINANCE_LARGE_HARD = DatasetConfig(
    name="finance_large_hard",
    domain="finance",
    num_rows=5000,
    difficulty="hard",
    description="Large finance dataset with 5-7 complex cleaning operations",
)

# =============================================================================
# Training Set Configurations (for agent training)
# =============================================================================

TRAINING_SET_SMALL = DatasetConfig(
    name="training_set_small",
    domain="ecommerce",  # Can be overridden
    num_rows=100,
    difficulty="easy",
    description="Small training examples for quick iteration",
)

TRAINING_SET_DIVERSE = DatasetConfig(
    name="training_set_diverse",
    domain="ecommerce",  # Can be overridden
    num_rows=1000,
    difficulty="medium",
    description="Diverse training examples with varied corruptions",
)

# =============================================================================
# Configuration Registry
# =============================================================================

# All pre-defined configurations
ALL_CONFIGS = {
    # E-commerce
    "ecommerce_small_easy": ECOMMERCE_SMALL_EASY,
    "ecommerce_medium": ECOMMERCE_MEDIUM,
    "ecommerce_large_hard": ECOMMERCE_LARGE_HARD,
    "ecommerce_text_only": ECOMMERCE_TEXT_ONLY,
    "ecommerce_numeric_only": ECOMMERCE_NUMERIC_ONLY,
    # Healthcare
    "healthcare_small_easy": HEALTHCARE_SMALL_EASY,
    "healthcare_medium": HEALTHCARE_MEDIUM,
    "healthcare_large_hard": HEALTHCARE_LARGE_HARD,
    # Finance
    "finance_small_easy": FINANCE_SMALL_EASY,
    "finance_medium": FINANCE_MEDIUM,
    "finance_large_hard": FINANCE_LARGE_HARD,
    # Training sets
    "training_set_small": TRAINING_SET_SMALL,
    "training_set_diverse": TRAINING_SET_DIVERSE,
}


def get_config(name: str) -> DatasetConfig:
    """
    Get a pre-defined configuration by name.

    Args:
        name: Configuration name

    Returns:
        DatasetConfig object

    Raises:
        KeyError: If configuration name not found
    """
    if name not in ALL_CONFIGS:
        available = ", ".join(sorted(ALL_CONFIGS.keys()))
        raise KeyError(
            f"Configuration '{name}' not found. Available: {available}"
        )

    return ALL_CONFIGS[name]


def list_configs() -> List[str]:
    """
    List all available configuration names.

    Returns:
        List of configuration names
    """
    return sorted(ALL_CONFIGS.keys())


def list_configs_by_domain(domain: str) -> List[str]:
    """
    List configurations for a specific domain.

    Args:
        domain: Domain name (ecommerce, healthcare, finance)

    Returns:
        List of configuration names for that domain
    """
    return sorted(
        [
            name
            for name, config in ALL_CONFIGS.items()
            if config.domain == domain
        ]
    )
