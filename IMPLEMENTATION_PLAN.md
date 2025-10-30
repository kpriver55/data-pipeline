# Synthetic Dirty-Clean Dataset Pair Generation Pipeline

## Overview

This pipeline generates pairs of raw (dirty) and clean (silver-level) datasets for training and testing data cleaning agents. The key insight is to generate clean data first, then introduce controlled data quality issues to create the raw version, ensuring perfect consistency.

**IMPORTANT**: This pipeline is designed to generate data specifically for training the data cleaning agent in `../data-cleaner`. All corruptions are carefully designed to be solvable using only the agent's available tools.

## Data Cleaning Agent Capabilities

The target cleaning agent (in `../data-cleaner`) has the following tools:

### Available Cleaning Operations

1. **handle_missing_values**: Impute missing values
   - Numeric strategies: mean, median, mode, interpolate, zero, knn
   - Categorical strategies: most_frequent, unknown, empty_string, missing, forward_fill, back_fill, constant

2. **remove_duplicates**: Remove exact duplicate rows
   - Can specify subset of columns
   - Keep options: first, last, False

3. **remove_outliers**: Remove outlier rows
   - Methods: IQR, z-score
   - Configurable threshold

4. **clean_text_columns**: Clean text data
   - Operations: strip (whitespace), lower, upper, remove_special, remove_numbers

5. **convert_data_types**: Convert column types
   - Target types: int, float, string, datetime, boolean, category
   - Uses coercion (errors='coerce')

6. **inspect_data**: Analyze data quality (read-only)

### Agent Limitations (Corruptions to AVOID)

The agent **cannot** handle:
- **Typos/misspellings**: No fuzzy matching or spell correction
- **Complex date format parsing**: Limited to pd.to_datetime() automatic parsing
- **Near-duplicates**: Can only detect exact duplicates
- **Semantic corrections**: Cannot infer intended values
- **Cross-column validation**: No relational constraint checking

## Architecture

### Core Principle: Clean-First Generation

```
Schema Definition → Clean Data Generation → Controlled Corruption → Raw Data
                                        ↓
                                   Validation
```

### Key Components

1. **Schema Generator**: Defines dataset structure (columns, types, distributions)
2. **Clean Data Generator**: Creates silver-level clean datasets
3. **Corruption Engine**: Applies controlled data quality issues
4. **Validation Layer**: Ensures raw→clean transformation is feasible
5. **Export Module**: Saves both versions with metadata

## Implementation Plan

### Phase 1: Schema Definition System

**Purpose**: Define the structure and characteristics of datasets to generate

**Components**:
- `schema.py`: Core schema data structures
- `schema_generator.py`: LLM-powered schema generation from domain descriptions

**Schema Elements**:
- Column definitions (name, type, constraints)
- Data types: numeric (int/float), datetime, categorical
- Distribution parameters (for numerics)
- Categorical value sets (generated via LLM)
- Relationships/constraints between columns

**LLM Usage**:
- Generate realistic column names for a domain (e.g., e-commerce, healthcare, finance)
- Create realistic categorical value sets
- Suggest appropriate data distributions

### Phase 2: Clean Data Generation

**Purpose**: Generate silver-level clean datasets

**Components**:
- `generators/numeric_generator.py`: Generate numeric data with specified distributions
- `generators/datetime_generator.py`: Generate datetime values with patterns
- `generators/categorical_generator.py`: Generate categorical values (using LLM)
- `clean_data_builder.py`: Orchestrates clean data generation

**Features**:
- Numeric: Normal, uniform, exponential distributions; respect ranges
- Datetime: Regular intervals, business hours, seasonal patterns
- Categorical: Realistic values via LLM, proper frequency distributions
- Inter-column relationships (e.g., price correlates with category)

**Silver-Level Characteristics**:
- No missing values
- Consistent formatting
- Valid data types
- No duplicates (unless intentional)
- No obvious outliers or anomalies

### Phase 3: Corruption Engine

**Purpose**: Transform clean data into realistic dirty data that the agent can fix

**Components**:
- `corruption/base.py`: Abstract corruption strategy interface
- `corruption/missing_values.py`: Introduce NULLs, empty strings, sentinel values
- `corruption/text_issues.py`: Whitespace, case, special characters
- `corruption/numeric_issues.py`: Outliers, type mixing
- `corruption/duplicates.py`: Introduce exact duplicate rows
- `corruption/type_corruption.py`: Convert columns to wrong types
- `corruption/orchestrator.py`: Apply multiple corruption strategies with configuration

**Corruption Strategies** (Mapped to Agent Tools):

1. **Missing Values** (10-30% of cells) → `handle_missing_values`
   - Replace with NULL/NaN
   - Empty strings for text columns
   - Sentinel values for numeric columns (-999, -1, 0 depending on context)
   - Mixed missing indicators within same column
   - **Why solvable**: Agent has comprehensive imputation strategies

2. **Text Issues** → `clean_text_columns`
   - **Whitespace**: Leading/trailing spaces, multiple internal spaces
   - **Case inconsistencies**: Mix "active", "Active", "ACTIVE"
   - **Special characters**: Add punctuation/symbols to categorical values
   - **Mixed formatting**: Numbers embedded in text (e.g., "Product123")
   - **Why solvable**: Agent can strip, normalize case, remove special chars/numbers

3. **Numeric Outliers** (1-5% of rows) → `remove_outliers`
   - Extreme values detectable by IQR (e.g., age=999, salary=9999999)
   - Values far from distribution mean (detectable by z-score)
   - Keep outliers realistic enough to be caught by threshold=1.5-3.0
   - **Why solvable**: Agent has IQR and z-score detection with configurable thresholds

4. **Type Corruption** → `convert_data_types`
   - Store numbers as strings: "123" instead of 123
   - Store datetimes as strings in parseable format: "2024-01-15"
   - Mix numeric types: integers with decimals stored as strings
   - **Why solvable**: Agent can convert with coercion (errors='coerce')

5. **Exact Duplicates** (2-10% duplicate rate) → `remove_duplicates`
   - Create exact row duplicates
   - Create duplicates on subset of columns
   - **Why solvable**: Agent can detect and remove exact duplicates
   - **NOTE**: NO near-duplicates with variations (agent cannot handle)

6. **Datetime Issues** → `convert_data_types` + `handle_missing_values`
   - Convert proper datetimes to string format
   - Use pandas-parseable formats only: "2024-01-15", "2024-01-15 10:30:00"
   - Introduce NULL values
   - **Why solvable**: Agent can convert back using pd.to_datetime() with coercion
   - **NOTE**: Do NOT mix multiple date formats (agent has limited parsing)

**Corruption Configuration**:
```python
corruption_config = {
    # Rates determine what percentage of applicable cells/rows are corrupted
    "missing_rate": 0.15,           # 15% of cells → NULL/empty/sentinel
    "text_whitespace_rate": 0.20,   # 20% of text cells get whitespace
    "text_case_variation_rate": 0.25,  # 25% of text cells get case changes
    "text_special_char_rate": 0.10,    # 10% of text cells get special chars
    "type_corruption_rate": 0.30,   # 30% of columns get type corruption
    "duplicate_rate": 0.05,         # 5% of rows duplicated
    "outlier_rate": 0.03,           # 3% of numeric rows become outliers
    "outlier_method": "iqr",        # "iqr" or "zscore" or "both"
    "outlier_severity": 2.5         # How extreme (IQR multiplier or z-score threshold)
}
```

**LLM Usage** (Reduced scope):
- Generate realistic categorical values (for clean data)
- Suggest appropriate sentinel values for missing data
- **NOT used for**: Typo generation (removed)

### Phase 3.5: Recovered Clean Dataset Generation

**Purpose**: Generate the "recovered clean" dataset - the actual achievable target for the agent

**Key Insight**: The perfect synthetic data generated in Phase 2 is **unrealistically clean**. In practice, no matter how good a cleaning agent is, some information cannot be recovered after corruption:
- Outliers removed → row count changes
- Duplicates removed → row count changes
- Type conversion with `errors='coerce'` → may create new NaN values
- Missing value imputation → approximations, not exact recovery

**Solution**: Auto-clean the dirty dataset using the same operations the agent would perform. This produces the "recovered clean" dataset, which is the **actual target** the agent should achieve.

**Components**:
- `corruption/cleaning_script_generator.py`: Maps corruption metadata to cleaning operations
- `auto_clean_dataset()`: Automatically applies cleaning steps in proper order

**Cleaning Operation Order** (Critical):
1. `clean_text_columns` - FIRST: Clean text so type conversion can work
2. `convert_data_types` - SECOND: Fix types after text is clean
3. `handle_missing_values` - THIRD: Handle missing after types are correct
4. `remove_outliers` - FOURTH: Remove outliers (needs correct types)
5. `remove_duplicates` - LAST: Remove duplicates

**Updated Orchestrator Output**:
```python
# OLD: Returns 2 values
df_dirty, metadata = corrupt_dataset(df_clean, config)

# NEW: Returns 3 values
df_dirty, df_recovered_clean, metadata = corrupt_dataset(df_perfect, config)
```

**Three Dataset Types**:
1. **Perfect Synthetic** (df_perfect): Unrealistically clean, generated in Phase 2
2. **Dirty** (df_dirty): Corrupted version with data quality issues
3. **Recovered Clean** (df_recovered_clean): **ACTUAL TARGET** - what the agent should achieve

**Metadata Enhancements**:
- `cleaning_operations`: List of operations applied to recover clean data
- `recovery_summary`: Stats comparing perfect → dirty → recovered
  - Row counts (shows rows lost in outlier/duplicate removal)
  - Missing value counts (may increase due to type coercion)
  - Duplicate counts

**Why This Matters**:
- Training the agent to match perfect synthetic data would be impossible
- The recovered clean dataset represents realistic, achievable cleaning
- Agent evaluation should compare to recovered clean, not perfect clean
- More accurate difficulty assessment based on what's actually achievable

### Phase 4: Validation Layer

**Purpose**: Ensure raw data can be cleaned to the **recovered clean version** using agent's tools

**Components**:
- `validation/consistency_checker.py`: Verify raw→clean mappings
- `validation/solvability_checker.py`: Ensure all corruptions are fixable by agent

**Checks**:
1. **Consistency Checks**:
   - Same number of logical records (accounting for duplicates and outliers)
   - All clean values derivable from raw values
   - No information added that wasn't in raw data

2. **Solvability Checks** (Agent-Specific):
   - **Missing values**: Can be imputed or filled with reasonable strategies
   - **Text issues**: Whitespace, case, and special chars match clean version after text cleaning
   - **Type issues**: All type conversions succeed with pd.to_numeric/to_datetime coercion
   - **Duplicates**: All duplicates are exact (no near-duplicates)
   - **Outliers**: All outliers detectable with IQR threshold ≤ 3.0 or z-score ≤ 3.0
   - **Datetimes**: All datetime strings parseable by pd.to_datetime()

3. **Metadata Generation**:
   - Document which agent operations are required to clean the data
   - Provide expected operation sequence
   - Include difficulty level (based on corruption complexity)

### Phase 5: Pipeline Orchestration

**Components**:
- `pipeline.py`: Main pipeline orchestrator
- `config/dataset_configs.py`: Pre-defined dataset configurations
- `llm/ollama_client.py`: Interface to local Ollama LLMs

**Pipeline Flow**:
```python
# 1. Define or generate schema
schema = generate_schema(domain="e-commerce", num_columns=10)

# 2. Generate perfect synthetic data
perfect_df = generate_clean_data(schema, num_rows=1000)

# 3. Apply corruption and generate recovered clean target
raw_df, recovered_clean_df, metadata = apply_corruption(perfect_df, corruption_config)

# 4. Validate consistency
validate_pair(raw_df, recovered_clean_df)

# 5. Export (use recovered_clean as the target, not perfect)
export_pair(raw_df, recovered_clean_df, metadata, output_dir="./output")
```

### Phase 6: Export and Metadata

**Components**:
- `export/exporter.py`: Export to CSV, Parquet, JSON
- `export/metadata_generator.py`: Generate documentation

**Outputs**:
- `raw_dataset.csv`: Dirty dataset
- `clean_dataset.csv`: Recovered clean dataset (**actual achievable target**)
- `metadata.json`: Complete dataset documentation
  - Schema definition
  - Corruption types and rates applied
  - Statistics (missing %, duplicate %, outlier %)
  - Required agent operations to clean
  - Cleaning operations applied during recovery
  - Expected operation sequence
  - Difficulty level (easy/medium/hard)
  - Recovery summary (perfect → dirty → recovered stats)
- `data_quality_report.json`: Detailed issue inventory
  - Per-column quality metrics
  - Specific corruption locations
  - Solvability verification results

**Note**: The "clean" dataset is the **recovered clean** version, not the perfect synthetic. This represents what the agent can realistically achieve.

### Phase 7: Streamlit App Wrapper

**Purpose**: Provide a user-friendly web interface for the pipeline

**Components**:
- `app.py`: Main Streamlit application
- `app/components/`: Reusable UI components
- `app/pages/`: Multi-page application structure

**Features**:
1. **Schema Builder**:
   - Select pre-built schemas (e-commerce, healthcare, finance)
   - Custom schema creation with interactive forms
   - Preview schema structure

2. **Data Generation**:
   - Configure dataset size
   - Set random seed for reproducibility
   - Generate and preview clean data
   - Download clean dataset

3. **Corruption Configuration**:
   - Select difficulty level (easy/medium/hard)
   - Custom strategy selection with checkboxes
   - Adjust corruption rates
   - Preview corruption config

4. **Dataset Generation**:
   - Generate corrupted dataset
   - Side-by-side comparison (clean vs dirty)
   - Statistics and visualizations
   - Download both datasets + metadata

5. **Batch Generation**:
   - Generate multiple dataset pairs at once
   - Specify number of datasets per difficulty level
   - Bulk download as ZIP

6. **Metadata Viewer**:
   - Display corruption summary
   - Show required cleaning operations
   - Visualize data quality metrics

**UI Flow**:
```
1. Select/Create Schema → 2. Configure Corruption → 3. Generate Data → 4. Download
```

**Benefits**:
- No coding required for non-technical users
- Visual feedback during generation
- Easy experimentation with parameters
- Quick iteration for testing agent training data

## LLM Integration with Ollama

### Use Cases

1. **Schema Generation**:
   ```python
   prompt = f"Generate 10 realistic column names for a {domain} dataset"
   columns = ollama.generate(model="llama2", prompt=prompt)
   ```

2. **Categorical Value Generation**:
   ```python
   prompt = f"Generate 20 realistic values for a '{column_name}' category in {domain}"
   values = ollama.generate(model="llama2", prompt=prompt)
   ```

3. **Sentinel Value Suggestions**:
   ```python
   prompt = f"What are common missing value indicators for {column_name} in {domain}?"
   sentinels = ollama.generate(model="llama2", prompt=prompt)
   # e.g., age: -1, 999; salary: -999, 0; status: "Unknown", "N/A"
   ```

### Ollama Client
```python
class OllamaClient:
    def generate(self, prompt: str, model: str = "llama2") -> str:
        # Call ollama API via HTTP
        pass

    def generate_list(self, prompt: str, count: int) -> List[str]:
        # Parse structured outputs (e.g., JSON, comma-separated)
        pass
```

**Note**: LLM usage is optional and primarily for generating realistic domain-specific values. The corruption engine does NOT use LLMs (no typo generation).

## Project Structure

```
data-pipeline/
├── README.md
├── IMPLEMENTATION_PLAN.md
├── requirements.txt
├── app.py                       # Streamlit app entry point (Phase 7)
├── config/
│   ├── __init__.py
│   └── dataset_configs.py       # Pre-defined dataset configs
├── schema/
│   ├── __init__.py
│   ├── schema.py                # Schema data structures
│   └── schema_generator.py      # LLM-powered schema generation (optional)
├── generators/
│   ├── __init__.py
│   ├── numeric_generator.py
│   ├── datetime_generator.py
│   ├── categorical_generator.py
│   └── clean_data_builder.py
├── corruption/
│   ├── __init__.py
│   ├── base.py                  # Abstract corruption strategy
│   ├── missing_values.py        # NULL, sentinel values, empty strings
│   ├── text_issues.py           # Whitespace, case, special chars
│   ├── numeric_issues.py        # Outliers (IQR/z-score detectable)
│   ├── type_corruption.py       # Convert to wrong types (but parseable)
│   ├── duplicates.py            # Exact duplicates only
│   ├── cleaning_script_generator.py  # Maps corruptions to cleaning operations (Phase 3.5)
│   └── orchestrator.py          # Coordinate all corruption strategies
├── validation/
│   ├── __init__.py
│   ├── consistency_checker.py   # Verify raw→clean consistency
│   └── solvability_checker.py   # Verify agent can solve all issues
├── llm/
│   ├── __init__.py
│   └── ollama_client.py         # Optional: for schema/value generation
├── export/
│   ├── __init__.py
│   ├── exporter.py
│   └── metadata_generator.py    # Include required operations
├── app/                         # Streamlit app components (Phase 7)
│   ├── components/              # Reusable UI components
│   └── pages/                   # Multi-page app structure
├── pipeline.py                  # Main orchestrator
└── examples/
    ├── generate_ecommerce.py
    ├── generate_healthcare.py
    └── generate_finance.py
```

## Dependencies

```
pandas
numpy
faker  # For generating realistic base data
ollama  # Python client for Ollama
pydantic  # For schema validation
python-dateutil
pyarrow  # For Parquet support
streamlit  # For web UI (Phase 7)
```

## Implementation Priority

1. **Phase 1**: Schema definition system (foundation) ✅ **COMPLETE**
2. **Phase 2**: Clean data generation (core functionality) ✅ **COMPLETE**
3. **Phase 3**: Corruption engine (key differentiator) ✅ **COMPLETE**
4. **Phase 3.5**: Recovered clean dataset generation (critical insight) ✅ **COMPLETE**
5. **Phase 4**: Validation layer (ensures quality)
6. **Phase 5**: Pipeline orchestration (ties everything together)
7. **Phase 6**: Export and metadata (polish)
8. **Phase 7**: Streamlit app wrapper (user interface)

## Success Criteria

1. **Consistency**: Clean version is fully derivable from raw version
2. **Realism**: Data quality issues mirror real-world problems
3. **Configurability**: Easy to adjust corruption levels and types
4. **Scalability**: Can generate datasets from 1K to 1M+ rows
5. **Diversity**: Support multiple domains and use cases
6. **Validation**: Automated checks ensure quality pairs

## Example Usage

```python
from pipeline import SyntheticDataPipeline
from config.dataset_configs import ECOMMERCE_CONFIG

# Initialize pipeline
pipeline = SyntheticDataPipeline(
    model="llama2",
    ollama_host="localhost:11434"
)

# Generate dataset trio (perfect, dirty, recovered_clean)
raw_df, recovered_clean_df, metadata = pipeline.generate(
    config=ECOMMERCE_CONFIG,
    num_rows=50000,
    corruption_level="medium"  # easy, medium, hard
)

# Export (using recovered_clean as the actual target)
pipeline.export(
    raw_df=raw_df,
    clean_df=recovered_clean_df,  # This is the achievable target
    metadata=metadata,
    output_dir="./output/ecommerce_dataset"
)
```

## Corruption-to-Tool Mapping

This table shows how each corruption type maps to specific agent tools:

| Corruption Type | Example Issues | Agent Tool(s) | Notes |
|----------------|----------------|---------------|-------|
| Missing Values | NULL, NaN, empty strings, sentinel values (-999) | `handle_missing_values` | Use various imputation strategies |
| Text Whitespace | Leading/trailing spaces, multiple spaces | `clean_text_columns` (strip) | Can be combined with case normalization |
| Case Inconsistencies | "Active", "ACTIVE", "active" | `clean_text_columns` (lower/upper) | Normalize to single case |
| Special Characters | "Product#123", "user@@@" | `clean_text_columns` (remove_special) | Strip non-alphanumeric |
| Numbers in Text | "Category123", "Status999" | `clean_text_columns` (remove_numbers) | Remove numeric characters |
| Type Mismatches | "123" as string, dates as strings | `convert_data_types` | Use appropriate target type |
| Numeric Outliers | age=999, salary=99999999 | `remove_outliers` (iqr/zscore) | Ensure detectable with threshold ≤ 3.0 |
| Exact Duplicates | Identical rows | `remove_duplicates` | Can specify subset of columns |
| Mixed Issues | Multiple corruptions per cell/row | Multiple tools in sequence | Requires proper operation ordering |

## Design Decisions

### Why Clean-First?
Generating clean data first and then corrupting it ensures perfect consistency. The alternative (generating dirty data and cleaning it) risks introducing artifacts or inconsistencies. This approach also makes validation trivial: we know exactly what the clean version should look like.

### Why Agent-Specific Corruptions?
The pipeline is designed specifically for the data-cleaner agent. Every corruption is carefully chosen to be solvable with the agent's available tools. This ensures:
- Training data quality: The agent can always reach the target clean state
- Meaningful optimization: Failures indicate agent limitations, not impossible tasks
- Realistic difficulty: Issues are challenging but solvable

### Why No Typos/Fuzzy Matching?
The agent lacks fuzzy matching or spell correction capabilities. Including typos would create unsolvable problems, leading to poor training outcomes. We focus on corruptions the agent can definitively fix.

### Why Local LLMs (Optional)?
- Privacy: No external data sharing
- Cost: Free inference
- Control: Can fine-tune or swap models
- Speed: Fast iteration during development
- **Note**: LLM usage is optional; corruptions don't require LLMs

### Why Silver-Level Clean Data?
Matches the typical input for machine learning pipelines. Going further (gold-level with feature engineering) would make the problem too domain-specific.

### Corruption Realism
Focus on common data quality issues that appear in real datasets AND that the agent can fix. The goal is to train the agent on realistic but solvable problems.

## Future Enhancements

1. **Multi-table datasets**: Generate related tables with foreign keys
2. **Time-series specific corruptions**: Gaps, irregular intervals
3. **Domain-specific patterns**: Healthcare coding errors, financial rounding
4. **Cleaning script generation**: Auto-generate pandas cleaning code
5. **Difficulty levels**: Easy/medium/hard cleaning challenges
6. **Benchmark suite**: Standard test cases for data cleaning tools
