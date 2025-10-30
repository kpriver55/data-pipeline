"""
Schema definitions for synthetic dataset generation.

This module defines the structure and constraints for datasets to be generated.
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator


class ColumnType(str, Enum):
    """Supported column data types."""

    NUMERIC_INT = "numeric_int"
    NUMERIC_FLOAT = "numeric_float"
    DATETIME = "datetime"
    CATEGORICAL = "categorical"


class NumericDistribution(str, Enum):
    """Distribution types for numeric data."""

    NORMAL = "normal"
    UNIFORM = "uniform"
    EXPONENTIAL = "exponential"
    POISSON = "poisson"
    BINOMIAL = "binomial"


class NumericConfig(BaseModel):
    """Configuration for numeric column generation."""

    distribution: NumericDistribution = NumericDistribution.NORMAL
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    mean: Optional[float] = None
    std: Optional[float] = None
    decimals: int = Field(default=2, ge=0, le=10)  # For float types
    allow_negative: bool = True

    @field_validator("decimals")
    @classmethod
    def validate_decimals(cls, v):
        if v < 0 or v > 10:
            raise ValueError("Decimals must be between 0 and 10")
        return v


class DatetimeConfig(BaseModel):
    """Configuration for datetime column generation."""

    start_date: str = "2020-01-01"  # ISO format
    end_date: str = "2024-12-31"  # ISO format
    include_time: bool = False
    business_hours_only: bool = False  # If True, generate times between 9am-5pm
    weekdays_only: bool = False  # If True, skip weekends
    format: str = "%Y-%m-%d"  # strftime format


class CategoricalConfig(BaseModel):
    """Configuration for categorical column generation."""

    values: List[str] = Field(min_length=1)
    frequencies: Optional[List[float]] = None  # Relative frequencies (must sum to 1)
    allow_custom_distribution: bool = True

    @field_validator("frequencies")
    @classmethod
    def validate_frequencies(cls, v, info):
        if v is not None:
            if len(v) != len(info.data.get("values", [])):
                raise ValueError("Frequencies must match number of values")
            if not abs(sum(v) - 1.0) < 0.001:
                raise ValueError("Frequencies must sum to 1.0")
        return v


class ColumnSchema(BaseModel):
    """Schema definition for a single column."""

    name: str = Field(min_length=1)
    type: ColumnType
    description: Optional[str] = None

    # Type-specific configurations
    numeric_config: Optional[NumericConfig] = None
    datetime_config: Optional[DatetimeConfig] = None
    categorical_config: Optional[CategoricalConfig] = None

    # Constraints
    nullable: bool = False  # For clean data, should be False
    unique: bool = False  # Whether values should be unique

    @field_validator("numeric_config")
    @classmethod
    def validate_numeric_config(cls, v, info):
        col_type = info.data.get("type")
        if col_type in [ColumnType.NUMERIC_INT, ColumnType.NUMERIC_FLOAT]:
            if v is None:
                raise ValueError(f"numeric_config required for {col_type}")
        elif v is not None:
            raise ValueError(f"numeric_config not allowed for {col_type}")
        return v

    @field_validator("datetime_config")
    @classmethod
    def validate_datetime_config(cls, v, info):
        col_type = info.data.get("type")
        if col_type == ColumnType.DATETIME:
            if v is None:
                raise ValueError("datetime_config required for datetime columns")
        elif v is not None:
            raise ValueError(f"datetime_config not allowed for {col_type}")
        return v

    @field_validator("categorical_config")
    @classmethod
    def validate_categorical_config(cls, v, info):
        col_type = info.data.get("type")
        if col_type == ColumnType.CATEGORICAL:
            if v is None:
                raise ValueError("categorical_config required for categorical columns")
        elif v is not None:
            raise ValueError(f"categorical_config not allowed for {col_type}")
        return v


class ColumnRelationship(BaseModel):
    """Define relationships between columns (e.g., price correlates with category)."""

    type: str  # "correlation", "conditional", "derived"
    source_columns: List[str]
    target_column: str
    parameters: Dict[str, Any] = {}

    # Examples:
    # - Correlation: {"type": "correlation", "source": ["category"], "target": "price", "params": {"strength": 0.7}}
    # - Conditional: {"type": "conditional", "source": ["status"], "target": "end_date", "params": {"if_value": "completed"}}


class DatasetSchema(BaseModel):
    """Complete schema for a dataset."""

    name: str = Field(min_length=1)
    domain: str = Field(
        default="general", description="Domain context (e-commerce, healthcare, etc.)"
    )
    description: Optional[str] = None

    columns: List[ColumnSchema] = Field(min_length=1)
    relationships: List[ColumnRelationship] = Field(default_factory=list)

    # Dataset-level settings
    num_rows: int = Field(default=1000, ge=1)

    @field_validator("columns")
    @classmethod
    def validate_unique_names(cls, v):
        names = [col.name for col in v]
        if len(names) != len(set(names)):
            raise ValueError("Column names must be unique")
        return v

    def get_column(self, name: str) -> Optional[ColumnSchema]:
        """Get column schema by name."""
        for col in self.columns:
            if col.name == name:
                return col
        return None

    def get_columns_by_type(self, col_type: ColumnType) -> List[ColumnSchema]:
        """Get all columns of a specific type."""
        return [col for col in self.columns if col.type == col_type]

    def to_dict(self) -> Dict[str, Any]:
        """Convert schema to dictionary."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetSchema":
        """Create schema from dictionary."""
        return cls(**data)


# Helper functions for creating common column types


def create_numeric_column(
    name: str,
    is_float: bool = False,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    mean: Optional[float] = None,
    std: Optional[float] = None,
    distribution: NumericDistribution = NumericDistribution.NORMAL,
    decimals: int = 2,
    description: Optional[str] = None,
) -> ColumnSchema:
    """Helper to create a numeric column schema."""
    return ColumnSchema(
        name=name,
        type=ColumnType.NUMERIC_FLOAT if is_float else ColumnType.NUMERIC_INT,
        description=description,
        numeric_config=NumericConfig(
            distribution=distribution,
            min_value=min_value,
            max_value=max_value,
            mean=mean,
            std=std,
            decimals=decimals,
        ),
    )


def create_datetime_column(
    name: str,
    start_date: str = "2020-01-01",
    end_date: str = "2024-12-31",
    include_time: bool = False,
    business_hours_only: bool = False,
    weekdays_only: bool = False,
    description: Optional[str] = None,
) -> ColumnSchema:
    """Helper to create a datetime column schema."""
    return ColumnSchema(
        name=name,
        type=ColumnType.DATETIME,
        description=description,
        datetime_config=DatetimeConfig(
            start_date=start_date,
            end_date=end_date,
            include_time=include_time,
            business_hours_only=business_hours_only,
            weekdays_only=weekdays_only,
        ),
    )


def create_categorical_column(
    name: str,
    values: List[str],
    frequencies: Optional[List[float]] = None,
    description: Optional[str] = None,
) -> ColumnSchema:
    """Helper to create a categorical column schema."""
    return ColumnSchema(
        name=name,
        type=ColumnType.CATEGORICAL,
        description=description,
        categorical_config=CategoricalConfig(values=values, frequencies=frequencies),
    )
