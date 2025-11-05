"""
Streamlit UI for Synthetic Data Pipeline.

A web interface for generating dirty/clean dataset pairs using natural language,
pre-configured templates, or custom configurations.
"""

import io
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.dataset_configs import ALL_CONFIGS, DatasetConfig, get_config
from export.exporter import ExportFormat
from llm.ollama_client import OllamaClient
from pipeline import generate_dataset, generate_batch_datasets
from schema.schema_generator import SchemaGenerator


# =============================================================================
# Page Configuration
# =============================================================================

st.set_page_config(
    page_title="Synthetic Data Pipeline",
    page_icon="🔄",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# Session State Initialization
# =============================================================================

if "generated_datasets" not in st.session_state:
    st.session_state.generated_datasets = []

if "current_quality_report" not in st.session_state:
    st.session_state.current_quality_report = None

if "current_metadata" not in st.session_state:
    st.session_state.current_metadata = None


# =============================================================================
# Helper Functions
# =============================================================================

def check_ollama_connection():
    """Check if Ollama is running and return status."""
    try:
        client = OllamaClient()
        return True, "Connected"
    except Exception as e:
        return False, str(e)


def format_file_size(size_bytes):
    """Format file size in bytes to human-readable format."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


def create_download_button(df, filename, label, key):
    """Create a download button for a dataframe."""
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_data = csv_buffer.getvalue()

    st.download_button(
        label=label,
        data=csv_data,
        file_name=filename,
        mime="text/csv",
        key=key,
    )


def display_quality_report(report):
    """Display quality report in a nice format."""
    if not report:
        return

    st.subheader("📊 Quality Report")

    # Summary metrics in columns
    summary = report["summary"]
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Overall Quality Score",
            f"{summary['overall_quality_score']}/100",
            delta=None,
        )

    with col2:
        st.metric(
            "Row Retention",
            f"{summary['row_retention_rate']:.1f}%",
            delta=None,
        )

    with col3:
        st.metric(
            "Missing Values Reduced",
            f"{summary['missing_reduction_rate']:.1f}%",
            delta=None,
        )

    with col4:
        st.metric(
            "Duplicates Removed",
            f"{summary['duplicate_reduction_rate']:.1f}%",
            delta=None,
        )

    # Data statistics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("**Dirty Dataset**")
        st.write(f"Rows: {summary['total_rows_dirty']:,}")
        st.write(f"Missing: {summary['missing_values_dirty']:,}")
        st.write(f"Duplicates: {summary['duplicates_dirty']:,}")

    with col2:
        st.write("**Recovered Dataset**")
        st.write(f"Rows: {summary['total_rows_recovered']:,}")
        st.write(f"Missing: {summary['missing_values_recovered']:,}")
        st.write(f"Duplicates: {summary['duplicates_recovered']:,}")

    with col3:
        st.write("**Processing**")
        st.write(f"Strategies: {summary['strategies_applied']}")
        st.write(f"Operations: {summary['cleaning_operations']}")
        st.write(f"Difficulty: {summary['difficulty']}")

    # Recommendations
    if "recommendations" in report and report["recommendations"]:
        st.write("**Recommendations**")
        for rec in report["recommendations"]:
            severity = rec["severity"]
            if severity == "warning":
                st.warning(f"⚠️ {rec['message']}")
            elif severity == "info":
                st.info(f"ℹ️ {rec['message']}")
            elif severity == "success":
                st.success(f"✓ {rec['message']}")

    # Detailed column quality (expandable)
    with st.expander("📋 Column Quality Details"):
        if "column_quality" in report:
            col_quality_data = []
            for col_name, metrics in report["column_quality"].items():
                row = {
                    "Column": col_name,
                    "Type": metrics.get("type", ""),
                    "Dirty Missing %": metrics["dirty"]["missing_pct"],
                    "Recovered Missing %": metrics["recovered"]["missing_pct"],
                    "Improvement Score": metrics["improvement_score"],
                }
                col_quality_data.append(row)

            df_quality = pd.DataFrame(col_quality_data)
            st.dataframe(df_quality, use_container_width=True)


def display_metadata(metadata):
    """Display metadata in expandable sections."""
    if not metadata:
        return

    with st.expander("📄 Dataset Metadata"):
        # Dataset config
        if "dataset_config" in metadata:
            st.write("**Configuration**")
            config = metadata["dataset_config"]
            st.json(config)

        # Corruption details
        if "applied_strategies" in metadata:
            st.write("**Applied Strategies**")
            st.write(", ".join(metadata["applied_strategies"]))

        # Cleaning operations
        if "cleaning_operations" in metadata:
            st.write("**Cleaning Operations**")
            for i, op in enumerate(metadata["cleaning_operations"], 1):
                st.write(f"{i}. {op['operation']}")


# =============================================================================
# Main UI
# =============================================================================

def main():
    st.title("🔄 Synthetic Data Pipeline")
    st.markdown(
        "Generate dirty/clean dataset pairs for training data cleaning agents"
    )

    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Ollama status
        st.subheader("LLM Status")
        ollama_connected, ollama_status = check_ollama_connection()
        if ollama_connected:
            st.success(f"✓ Ollama: {ollama_status}")
        else:
            st.error(f"✗ Ollama: Not connected")
            st.caption("LLM features require Ollama to be running")

        st.divider()

        # Export settings
        st.subheader("Export Settings")
        export_format = st.selectbox(
            "Export Format",
            options=["CSV", "Parquet", "JSON", "All"],
            index=0,
        )

        format_map = {
            "CSV": ExportFormat.CSV,
            "Parquet": ExportFormat.PARQUET,
            "JSON": ExportFormat.JSON,
            "All": ExportFormat.ALL,
        }
        selected_format = format_map[export_format]

        include_quality_report = st.checkbox(
            "Include Quality Report", value=True
        )

        st.divider()

        # Validation settings
        st.subheader("Validation")
        validate_datasets = st.checkbox("Validate Datasets", value=True)

    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🤖 Natural Language",
        "📋 Pre-configured",
        "⚙️ Custom Config",
        "📦 Batch Generation"
    ])

    # =============================================================================
    # Tab 1: Natural Language Generation (LLM-powered)
    # =============================================================================
    with tab1:
        st.header("Generate from Natural Language Description")
        st.markdown(
            "Describe your use case in natural language, and the LLM will "
            "generate an appropriate schema configuration."
        )

        if not ollama_connected:
            st.warning(
                "⚠️ Ollama is not running. Please start Ollama to use this feature."
            )
            st.code("ollama serve", language="bash")
        else:
            with st.form("nlp_form"):
                description = st.text_area(
                    "Dataset Description",
                    placeholder="Example: E-commerce transaction data with customer information, "
                    "order details, and payment methods",
                    height=100,
                )

                col1, col2 = st.columns(2)
                with col1:
                    num_rows = st.number_input(
                        "Number of Rows",
                        min_value=10,
                        max_value=100000,
                        value=1000,
                        step=100,
                    )

                with col2:
                    num_columns = st.number_input(
                        "Number of Columns (optional)",
                        min_value=0,
                        max_value=50,
                        value=0,
                        step=1,
                        help="Leave at 0 for automatic column count",
                    )

                difficulty = st.selectbox(
                    "Corruption Difficulty",
                    options=["easy", "medium", "hard"],
                    index=1,
                )

                seed = st.number_input(
                    "Random Seed (optional)",
                    min_value=0,
                    value=42,
                    help="For reproducible results",
                )

                generate_btn = st.form_submit_button("🚀 Generate Dataset", type="primary")

                if generate_btn:
                    if not description.strip():
                        st.error("Please provide a dataset description")
                    else:
                        with st.spinner("Generating schema from description..."):
                            try:
                                # Generate schema using LLM
                                schema_gen = SchemaGenerator()
                                schema = schema_gen.generate_schema_from_description(
                                    description=description,
                                    num_rows=num_rows,
                                    num_columns=num_columns if num_columns > 0 else None,
                                )

                                st.success(f"✓ Schema generated: {len(schema.columns)} columns")

                                # Show generated schema
                                with st.expander("📋 View Generated Schema"):
                                    for col in schema.columns:
                                        st.write(f"**{col.name}** ({col.type.value})")
                                        if col.description:
                                            st.caption(col.description)

                                # Create config from schema
                                config = DatasetConfig(
                                    name="llm_generated",
                                    domain="custom",
                                    num_rows=num_rows,
                                    difficulty=difficulty,
                                    seed=seed,
                                    description=description,
                                )

                                # Generate dataset
                                with st.spinner("Generating dataset..."):
                                    df_dirty, df_clean, metadata = generate_dataset(
                                        config=config,
                                        validate=validate_datasets,
                                        verbose=False,
                                        export_formats=selected_format,
                                        include_quality_report=include_quality_report,
                                    )

                                    st.session_state.generated_datasets = [
                                        ("dirty", df_dirty),
                                        ("clean", df_clean),
                                    ]
                                    st.session_state.current_metadata = metadata
                                    st.session_state.current_quality_report = (
                                        metadata.get("quality_report")
                                    )

                                    st.success("✓ Dataset generated successfully!")
                                    st.rerun()

                            except Exception as e:
                                st.error(f"Error generating dataset: {str(e)}")
                                st.exception(e)

    # =============================================================================
    # Tab 2: Pre-configured Datasets
    # =============================================================================
    with tab2:
        st.header("Generate from Pre-configured Template")
        st.markdown("Choose from pre-configured dataset templates")

        # Show available configs
        config_options = list(ALL_CONFIGS.keys())

        selected_config_name = st.selectbox(
            "Select Configuration",
            options=config_options,
            index=0,
        )

        # Show config details
        selected_config = get_config(selected_config_name)
        col1, col2, col3 = st.columns(3)

        with col1:
            st.write(f"**Domain:** {selected_config.domain}")
            st.write(f"**Rows:** {selected_config.num_rows:,}")

        with col2:
            st.write(f"**Difficulty:** {selected_config.difficulty}")
            if selected_config.seed:
                st.write(f"**Seed:** {selected_config.seed}")

        with col3:
            if selected_config.description:
                st.write(f"**Description:** {selected_config.description}")

        if st.button("🚀 Generate Dataset", key="preconfig_generate", type="primary"):
            with st.spinner(f"Generating {selected_config_name}..."):
                try:
                    df_dirty, df_clean, metadata = generate_dataset(
                        config=selected_config_name,
                        validate=validate_datasets,
                        verbose=False,
                        export_formats=selected_format,
                        include_quality_report=include_quality_report,
                    )

                    st.session_state.generated_datasets = [
                        ("dirty", df_dirty),
                        ("clean", df_clean),
                    ]
                    st.session_state.current_metadata = metadata
                    st.session_state.current_quality_report = (
                        metadata.get("quality_report")
                    )

                    st.success("✓ Dataset generated successfully!")
                    st.rerun()

                except Exception as e:
                    st.error(f"Error generating dataset: {str(e)}")
                    st.exception(e)

    # =============================================================================
    # Tab 3: Custom Configuration
    # =============================================================================
    with tab3:
        st.header("Generate with Custom Configuration")
        st.markdown("Create a fully customized dataset configuration")

        with st.form("custom_config_form"):
            config_name = st.text_input("Configuration Name", value="custom_dataset")

            domain = st.selectbox(
                "Domain",
                options=["ecommerce", "healthcare", "finance"],
                index=0,
            )

            col1, col2 = st.columns(2)

            with col1:
                num_rows = st.number_input(
                    "Number of Rows",
                    min_value=10,
                    max_value=100000,
                    value=1000,
                    step=100,
                )

            with col2:
                difficulty = st.selectbox(
                    "Difficulty",
                    options=["easy", "medium", "hard"],
                    index=1,
                )

            description = st.text_area(
                "Description (optional)",
                placeholder="Optional description of this dataset configuration",
            )

            seed = st.number_input(
                "Random Seed (optional)",
                min_value=0,
                value=42,
            )

            # Strategy selection
            st.write("**Corruption Strategies** (leave empty for automatic)")
            strategy_options = [
                "missing_values",
                "duplicate_rows",
                "type_corruption",
                "outliers_numeric",
                "text_whitespace",
                "text_case_variation",
                "text_special_chars",
            ]

            selected_strategies = st.multiselect(
                "Select Strategies",
                options=strategy_options,
                default=[],
            )

            custom_generate_btn = st.form_submit_button(
                "🚀 Generate Dataset",
                type="primary"
            )

            if custom_generate_btn:
                with st.spinner("Generating custom dataset..."):
                    try:
                        config = DatasetConfig(
                            name=config_name,
                            domain=domain,
                            num_rows=num_rows,
                            difficulty=difficulty,
                            strategies=selected_strategies if selected_strategies else None,
                            seed=seed,
                            description=description if description else None,
                        )

                        df_dirty, df_clean, metadata = generate_dataset(
                            config=config,
                            validate=validate_datasets,
                            verbose=False,
                            export_formats=selected_format,
                            include_quality_report=include_quality_report,
                        )

                        st.session_state.generated_datasets = [
                            ("dirty", df_dirty),
                            ("clean", df_clean),
                        ]
                        st.session_state.current_metadata = metadata
                        st.session_state.current_quality_report = (
                            metadata.get("quality_report")
                        )

                        st.success("✓ Dataset generated successfully!")
                        st.rerun()

                    except Exception as e:
                        st.error(f"Error generating dataset: {str(e)}")
                        st.exception(e)

    # =============================================================================
    # Tab 4: Batch Generation
    # =============================================================================
    with tab4:
        st.header("Batch Dataset Generation")
        st.markdown("Generate multiple datasets at once")

        batch_mode = st.radio(
            "Batch Mode",
            options=["Training Set", "Multiple Configs"],
            index=0,
        )

        if batch_mode == "Training Set":
            st.subheader("Generate Training Set")
            st.markdown("Generate multiple datasets with different difficulty levels")

            with st.form("training_set_form"):
                domain = st.selectbox(
                    "Domain",
                    options=["ecommerce", "healthcare", "finance"],
                    index=0,
                    key="batch_domain",
                )

                col1, col2, col3 = st.columns(3)

                with col1:
                    num_easy = st.number_input("Easy Datasets", min_value=0, value=5)

                with col2:
                    num_medium = st.number_input("Medium Datasets", min_value=0, value=5)

                with col3:
                    num_hard = st.number_input("Hard Datasets", min_value=0, value=5)

                rows_per_dataset = st.number_input(
                    "Rows per Dataset",
                    min_value=10,
                    max_value=10000,
                    value=1000,
                    step=100,
                )

                base_seed = st.number_input(
                    "Base Seed",
                    min_value=0,
                    value=42,
                )

                batch_generate_btn = st.form_submit_button(
                    f"🚀 Generate {num_easy + num_medium + num_hard} Datasets",
                    type="primary"
                )

                if batch_generate_btn:
                    total_datasets = num_easy + num_medium + num_hard

                    if total_datasets == 0:
                        st.error("Please specify at least one dataset to generate")
                    else:
                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        try:
                            from pipeline import generate_training_set

                            with st.spinner(f"Generating {total_datasets} datasets..."):
                                results = generate_training_set(
                                    domain=domain,
                                    num_easy=num_easy,
                                    num_medium=num_medium,
                                    num_hard=num_hard,
                                    rows_per_dataset=rows_per_dataset,
                                    base_seed=base_seed,
                                    export_formats=selected_format,
                                    include_quality_report=include_quality_report,
                                )

                            progress_bar.progress(1.0)
                            status_text.text(f"✓ Generated {len(results)} datasets")

                            st.success(f"✓ Successfully generated {len(results)} datasets!")

                            # Show summary
                            st.write("**Summary:**")
                            for i, (df_dirty, df_clean, metadata) in enumerate(results, 1):
                                config = metadata.get("dataset_config", {})
                                quality = metadata.get("quality_report", {}).get("summary", {})
                                quality_score = quality.get("overall_quality_score", "N/A")

                                st.write(
                                    f"{i}. {config.get('name', 'unknown')} - "
                                    f"Rows: {len(df_dirty)} → {len(df_clean)}, "
                                    f"Quality: {quality_score}/100"
                                )

                        except Exception as e:
                            st.error(f"Error generating batch: {str(e)}")
                            st.exception(e)

        else:  # Multiple Configs
            st.subheader("Generate from Multiple Configs")
            st.markdown("Select multiple pre-configured datasets to generate")

            config_options = list(ALL_CONFIGS.keys())
            selected_configs = st.multiselect(
                "Select Configurations",
                options=config_options,
                default=[],
            )

            if st.button(
                f"🚀 Generate {len(selected_configs)} Datasets",
                key="multi_config_generate",
                type="primary",
                disabled=len(selected_configs) == 0,
            ):
                progress_bar = st.progress(0)
                status_text = st.empty()

                try:
                    results = []
                    for i, config_name in enumerate(selected_configs, 1):
                        status_text.text(f"Generating {config_name}... ({i}/{len(selected_configs)})")

                        df_dirty, df_clean, metadata = generate_dataset(
                            config=config_name,
                            validate=validate_datasets,
                            verbose=False,
                            export_formats=selected_format,
                            include_quality_report=include_quality_report,
                        )

                        results.append((df_dirty, df_clean, metadata))
                        progress_bar.progress(i / len(selected_configs))

                    status_text.text(f"✓ Generated {len(results)} datasets")

                    st.success(f"✓ Successfully generated {len(results)} datasets!")

                    # Show summary
                    st.write("**Summary:**")
                    for i, (df_dirty, df_clean, metadata) in enumerate(results, 1):
                        config = metadata.get("dataset_config", {})
                        quality = metadata.get("quality_report", {}).get("summary", {})
                        quality_score = quality.get("overall_quality_score", "N/A")

                        st.write(
                            f"{i}. {config.get('name', 'unknown')} - "
                            f"Rows: {len(df_dirty)} → {len(df_clean)}, "
                            f"Quality: {quality_score}/100"
                        )

                except Exception as e:
                    st.error(f"Error generating batch: {str(e)}")
                    st.exception(e)

    # =============================================================================
    # Results Display (shown when datasets are generated)
    # =============================================================================

    if st.session_state.generated_datasets:
        st.divider()
        st.header("📊 Generated Datasets")

        # Dataset tabs
        dataset_tab1, dataset_tab2 = st.tabs(["Dirty Dataset", "Clean Dataset"])

        dirty_df = st.session_state.generated_datasets[0][1]
        clean_df = st.session_state.generated_datasets[1][1]

        with dataset_tab1:
            st.subheader("Dirty Dataset")
            st.write(f"Shape: {dirty_df.shape[0]:,} rows × {dirty_df.shape[1]} columns")

            # Preview
            st.dataframe(dirty_df.head(100), use_container_width=True)

            # Download button
            create_download_button(
                dirty_df,
                "dirty_dataset.csv",
                "📥 Download Dirty Dataset",
                "download_dirty",
            )

        with dataset_tab2:
            st.subheader("Clean Dataset (Recovered)")
            st.write(f"Shape: {clean_df.shape[0]:,} rows × {clean_df.shape[1]} columns")

            # Preview
            st.dataframe(clean_df.head(100), use_container_width=True)

            # Download button
            create_download_button(
                clean_df,
                "clean_dataset.csv",
                "📥 Download Clean Dataset",
                "download_clean",
            )

        # Quality report
        if st.session_state.current_quality_report:
            st.divider()
            display_quality_report(st.session_state.current_quality_report)

        # Metadata
        if st.session_state.current_metadata:
            st.divider()
            display_metadata(st.session_state.current_metadata)

            # Download metadata
            if st.button("📥 Download Metadata JSON"):
                metadata_json = json.dumps(
                    st.session_state.current_metadata,
                    indent=2,
                    default=str
                )

                st.download_button(
                    label="Download metadata.json",
                    data=metadata_json,
                    file_name="metadata.json",
                    mime="application/json",
                    key="download_metadata",
                )


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    main()
