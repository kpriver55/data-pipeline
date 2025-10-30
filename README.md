# Synthetic Data Pipeline for Data Cleaning Agent Training

A Python pipeline for generating synthetic paired datasets (clean and dirty) to train and optimize data cleaning agents. This pipeline generates realistic data quality issues that are solvable using standard data cleaning operations.

## Overview

This pipeline is specifically designed to generate training data for the data cleaning agent in `../data-cleaner`. It uses a **clean-first generation** approach: generate perfect data, then introduce controlled corruptions to create realistic dirty datasets.

### Key Features

- **Randomized corruption strategies** to prevent pattern learning
- **Difficulty levels** (easy, medium, hard) based on number of operations required
- **Agent-aware corruptions** - only introduces issues the cleaning agent can fix
- **Schema-based generation** with support for numeric, datetime, and categorical data
- **Multiple distributions** (normal, uniform, exponential, poisson, binomial)
- **Comprehensive metadata** tracking what corruptions were applied
- **Streamlit web interface** (planned) for easy dataset generation

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Current Implementation Status

✅ **Phase 1: Schema Definition System** (Complete)
- Pydantic-based schema definitions
- Pre-built schemas for e-commerce, healthcare, finance
- Custom schema builder

✅ **Phase 2: Clean Data Generation** (Complete)
- Numeric data generator (5 distribution types)
- Datetime generator with business hours/weekday constraints
- Categorical generator with frequency distributions

✅ **Phase 3: Corruption Engine** (Complete)
- 7 corruption strategies mapped to agent capabilities
- Randomized difficulty presets
- Orchestrator for applying multiple corruptions

⏳ **Phase 4: Validation Layer** (Planned)
- Consistency checking
- Solvability verification

⏳ **Phase 5: Pipeline Orchestration** (Planned)
- End-to-end dataset pair generation
- Batch processing

⏳ **Phase 6: Export and Metadata** (Planned)
- Multi-format export (CSV, Parquet, JSON)
- Comprehensive metadata generation
- Data quality reports

⏳ **Phase 7: Streamlit App Wrapper** (Planned)
- Web-based UI for non-technical users
- Interactive schema builder
- Real-time data preview
- Batch generation interface

## Quick Start

### Generate a Clean Dataset

```python
from schema.schema_generator import SchemaGenerator
from generators.clean_data_builder import build_clean_dataset

# Use pre-built schema
generator = SchemaGenerator()
schema = generator.generate_ecommerce_schema(num_rows=1000)

# Generate clean data
df_clean = build_clean_dataset(schema, seed=42)
df_clean.to_csv("output/clean_data.csv", index=False)
```

### Generate a Corrupted Dataset Pair

```python
from schema.schema_generator import SchemaGenerator
from generators.clean_data_builder import build_clean_dataset
from corruption.base import CorruptionConfig
from corruption.orchestrator import corrupt_dataset

# Generate clean data
generator = SchemaGenerator()
schema = generator.generate_ecommerce_schema(num_rows=1000)
df_clean = build_clean_dataset(schema, seed=42)

# Apply corruption (medium difficulty = 3-4 random strategies)
config = CorruptionConfig.from_preset("medium", seed=42)
df_dirty, metadata = corrupt_dataset(df_clean, config, seed=42)

# Save both versions
df_clean.to_csv("output/clean.csv", index=False)
df_dirty.to_csv("output/dirty.csv", index=False)
```

### Custom Corruption

```python
from corruption.base import CorruptionConfig

# Specify exact strategies to apply
config = CorruptionConfig.custom(
    strategies=["missing_values", "text_whitespace", "duplicates"],
    seed=42
)

df_dirty, metadata = corrupt_dataset(df_clean, config, seed=42)
```

## Corruption Strategies

All corruption strategies are designed to be solvable by the data cleaning agent:

| Strategy | Description | Agent Operation |
|----------|-------------|-----------------|
| `missing_values` | NULL, NaN, empty strings, sentinel values (-999, 0) | `handle_missing_values` |
| `text_whitespace` | Leading/trailing/internal spaces | `clean_text_columns` |
| `text_case` | Mixed case (UPPER, lower, Title, rAnDoM) | `clean_text_columns` |
| `text_special_chars` | Added punctuation/symbols | `clean_text_columns` |
| `type_corruption` | Wrong but parseable types (123 → "123") | `convert_data_types` |
| `outliers` | IQR-detectable extreme values | `remove_outliers` |
| `duplicates` | Exact duplicate rows | `remove_duplicates` |

## Difficulty Levels

Difficulty is based on **number of operations needed**, not corruption severity:

- **Easy**: 1-2 random strategies → 1-2 operations required
- **Medium**: 3-4 random strategies → 3-4 operations required
- **Hard**: 5-7 random strategies → 5-7 operations required

Each run randomly selects strategies to prevent the agent from learning fixed patterns.

## Project Structure

```
data-pipeline/
├── schema/                    # Schema definitions
│   ├── schema.py             # Pydantic schema models
│   └── schema_generator.py   # Pre-built and custom schemas
├── generators/               # Clean data generation
│   ├── numeric_generator.py
│   ├── datetime_generator.py
│   ├── categorical_generator.py
│   └── clean_data_builder.py
├── corruption/               # Corruption strategies
│   ├── base.py              # Base classes and config
│   ├── missing_values.py
│   ├── text_issues.py       # Whitespace, case, special chars
│   ├── numeric_issues.py    # Outliers
│   ├── type_corruption.py
│   ├── duplicates.py
│   └── orchestrator.py      # Coordinates all strategies
├── app.py                    # Streamlit web interface (Phase 7)
├── output/                   # Generated datasets (gitignored)
├── test_schema.py           # Phase 1 tests
├── test_clean_generation.py # Phase 2 tests
├── test_corruption.py       # Phase 3 tests
└── generate_sample.py       # Generate sample clean dataset
```

## Design Principles

### 1. Clean-First Generation
Generate perfect data first, then corrupt it. This ensures:
- Perfect consistency between clean and dirty versions
- Clean data is always derivable from dirty data
- Easy validation

### 2. Agent-Aware Corruptions
Only introduce issues the agent can fix:
- ✅ Whitespace, case, special chars (agent has text cleaning)
- ✅ Type mismatches with parseable formats (agent has type conversion)
- ✅ Outliers detectable by IQR/z-score (agent has outlier removal)
- ❌ Typos/misspellings (agent has no fuzzy matching)
- ❌ Near-duplicates (agent only handles exact duplicates)

### 3. Randomized Difficulty
Random strategy selection prevents overfitting:
- Each "medium" dataset has different 3-4 strategies
- Agent learns general cleaning patterns, not fixed sequences
- More diverse training data

### 4. Controlled Distribution Shift
- Outlier rate capped at 5% to prevent distribution change
- Safety margin (1.2x) ensures outliers remain detectable
- Integer outliers properly rounded

## Examples

Run the test scripts to see the pipeline in action:

```bash
# Test schema generation
python test_schema.py

# Test clean data generation
python test_clean_generation.py

# Test corruption engine
python test_corruption.py

# Generate a sample clean dataset
python generate_sample.py
```

## Configuration

### Default Corruption Rates

```python
DEFAULT_RATES = {
    "missing_rate": 0.15,        # 15% of cells
    "whitespace_rate": 0.20,     # 20% of text cells
    "case_rate": 0.25,           # 25% of text cells
    "special_char_rate": 0.10,   # 10% of text cells
    "duplicate_rate": 0.05,      # 5% of rows
    "outlier_rate": 0.03,        # 3% of numeric rows (capped at 5%)
    "outlier_severity": 2.5,     # IQR multiplier (standard is 1.5)
}
```

### Default Dataset Size

```python
# Changed from 10,000 to 1,000 for agent training efficiency
schema = generator.generate_ecommerce_schema(num_rows=1000)
```

## Data Cleaning Agent Integration

This pipeline generates data specifically for the agent in `../data-cleaner`:

**Agent Operations Available:**
- `handle_missing_values` - Imputation strategies
- `remove_duplicates` - Exact duplicate removal
- `remove_outliers` - IQR/z-score based
- `clean_text_columns` - Strip, case, special chars
- `convert_data_types` - Type conversion with coercion
- `inspect_data` - Data quality analysis

**Agent Limitations:**
- No fuzzy matching/spell correction
- No near-duplicate detection
- Limited date parsing (pandas defaults only)

## Roadmap

### Completed
- ✅ Schema definition system
- ✅ Clean data generation
- ✅ Corruption engine with 7 strategies

### In Progress
- 🚧 Validation layer
- 🚧 Pipeline orchestration
- 🚧 Export and metadata

### Planned
- 📋 Streamlit web interface
- 📋 Batch generation capabilities
- 📋 Data quality visualizations
- 📋 LLM-assisted schema generation (optional)

## Contributing

This pipeline is designed to evolve with the data cleaning agent's capabilities. When new cleaning operations are added to the agent, corresponding corruption strategies should be added here.

## License

[To be determined]
