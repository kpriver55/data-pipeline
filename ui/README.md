# Synthetic Data Pipeline UI

A web-based interface for generating dirty/clean dataset pairs for training data cleaning agents.

## Features

### 🤖 Natural Language Generation
- Describe your dataset needs in plain English
- LLM automatically generates appropriate schema configuration
- Customizable row count, column count, and difficulty
- Requires Ollama to be running locally

### 📋 Pre-configured Templates
- Choose from 13+ pre-built dataset configurations
- Covers multiple domains: e-commerce, healthcare, finance
- Multiple difficulty levels: easy, medium, hard
- Instant generation with proven configurations

### ⚙️ Custom Configuration
- Build fully customized dataset configurations
- Choose domain, size, and difficulty
- Select specific corruption strategies
- Fine-tune all parameters

### 📦 Batch Generation
- Generate multiple datasets at once
- Training set mode: automatically creates easy/medium/hard variations
- Multi-config mode: generate from multiple pre-configured templates
- Progress tracking for large batches

### 📊 Results & Analytics
- Interactive dataset preview
- Quality report with scoring (0-100)
- Per-column quality metrics
- Data loss analysis and recommendations
- Export datasets in CSV, Parquet, or JSON
- Download comprehensive metadata

## Running the UI

### Start the Application

From the `data-pipeline` directory:

```bash
streamlit run ui/app.py
```

The UI will open automatically in your browser at `http://localhost:8501`

### With Ollama (for LLM features)

To use the natural language generation feature, start Ollama first:

```bash
# In a separate terminal
ollama serve
```

Then start the Streamlit app as usual. The UI will automatically detect if Ollama is running.

## Usage

### 1. Natural Language Generation

1. Navigate to the "🤖 Natural Language" tab
2. Describe your dataset in the text area:
   ```
   E-commerce transaction data with customer information,
   order details, and payment methods
   ```
3. Set the number of rows and difficulty level
4. Click "Generate Dataset"
5. The LLM will create a schema and generate the datasets

### 2. Pre-configured Templates

1. Navigate to the "📋 Pre-configured" tab
2. Select a configuration from the dropdown (e.g., "ecommerce_medium")
3. Review the configuration details
4. Click "Generate Dataset"

### 3. Custom Configuration

1. Navigate to the "⚙️ Custom Config" tab
2. Fill in the configuration form:
   - Name your configuration
   - Choose domain (ecommerce/healthcare/finance)
   - Set number of rows and difficulty
   - Optionally select specific corruption strategies
3. Click "Generate Dataset"

### 4. Batch Generation

1. Navigate to the "📦 Batch Generation" tab
2. Choose batch mode:
   - **Training Set**: Generate multiple difficulty levels automatically
   - **Multiple Configs**: Select specific pre-configured templates
3. Configure the batch settings
4. Click "Generate X Datasets"
5. View progress and summary

## Configuration Options

### Sidebar Settings

**Export Format**
- CSV: Comma-separated values (default)
- Parquet: Columnar storage format (efficient)
- JSON: JavaScript Object Notation
- All: Export in all formats

**Include Quality Report**
- Enabled by default
- Generates comprehensive quality metrics
- Adds quality report to metadata

**Validate Datasets**
- Enabled by default
- Runs consistency and solvability checks
- Ensures datasets meet quality standards

## Understanding Results

### Quality Score (0-100)
- Based on three factors:
  - Missing value reduction (40% weight)
  - Duplicate removal (30% weight)
  - Data retention (30% weight)
- Higher scores indicate better quality

### Datasets
- **Dirty Dataset**: Corrupted data requiring cleaning
- **Clean Dataset**: Recovered clean data (achievable target)
- Clean dataset is what the agent should be able to recover, not perfect synthetic data

### Quality Report Sections

1. **Summary Metrics**: Overall quality score, retention rate, improvement metrics
2. **Data Statistics**: Row counts, missing values, duplicates for both datasets
3. **Recommendations**: Automated suggestions based on quality thresholds
4. **Column Quality**: Detailed per-column analysis with improvement scores

## Tips

- Start with pre-configured templates to understand the system
- Use natural language generation for custom use cases
- Enable validation to ensure dataset quality
- Check quality reports to verify generation results
- Use batch generation for creating training sets
- Export in Parquet format for large datasets (more efficient)

## Troubleshooting

### Ollama not connected
```bash
# Start Ollama service
ollama serve

# Verify it's running
curl http://localhost:11434/api/tags
```

### UI not loading
```bash
# Reinstall streamlit if needed
pip install streamlit

# Clear streamlit cache
streamlit cache clear
```

### Import errors
```bash
# Ensure you're running from the data-pipeline directory
cd /path/to/data-pipeline
streamlit run ui/app.py
```

## Examples

### Generate E-commerce Dataset with LLM
1. Tab: Natural Language
2. Description: "Online retail sales data with product categories, customer demographics, and purchase history"
3. Rows: 5000
4. Difficulty: medium
5. Generate

### Create Training Set
1. Tab: Batch Generation
2. Mode: Training Set
3. Domain: ecommerce
4. Easy: 10, Medium: 10, Hard: 10
5. Rows per dataset: 1000
6. Generate

### Quick Test
1. Tab: Pre-configured
2. Select: "ecommerce_small_easy"
3. Generate
4. Review quality report and download

## Advanced Features

### Custom Corruption Strategies
In the Custom Config tab, select specific strategies:
- `missing_values`: Introduce missing data
- `duplicate_rows`: Add duplicate records
- `type_corruption`: Corrupt data types
- `outliers_numeric`: Add numeric outliers
- `text_whitespace`: Add whitespace issues
- `text_case_variation`: Add case inconsistencies
- `text_special_chars`: Add special characters

### Export All Formats
Set export format to "All" to get:
- `dirty.csv`, `clean.csv`
- `dirty.parquet`, `clean.parquet`
- `dirty.json`, `clean.json`
- `metadata.json` (with quality report)
