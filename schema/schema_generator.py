"""
Schema generation utilities for creating dataset schemas.

Supports both manual schema creation and LLM-assisted generation.
"""

from typing import List, Optional

from schema.schema import (
    CategoricalConfig,
    ColumnSchema,
    ColumnType,
    DatasetSchema,
    DatetimeConfig,
    NumericConfig,
    NumericDistribution,
    create_categorical_column,
    create_datetime_column,
    create_numeric_column,
)


class SchemaGenerator:
    """Generate dataset schemas with or without LLM assistance."""

    def __init__(self, llm_client=None):
        """
        Initialize schema generator.

        Args:
            llm_client: Optional LLM client for generating suggestions (from llm/ollama_client.py)
        """
        self.llm_client = llm_client

    def create_simple_schema(
        self, name: str, domain: str, num_rows: int = 10000
    ) -> DatasetSchema:
        """Create a basic dataset schema without LLM assistance."""
        return DatasetSchema(name=name, domain=domain, columns=[], num_rows=num_rows)

    def generate_ecommerce_schema(self, num_rows: int = 1000) -> DatasetSchema:
        """Generate a typical e-commerce dataset schema."""
        columns = [
            create_numeric_column(
                name="order_id",
                is_float=False,
                min_value=1000,
                max_value=999999,
                distribution=NumericDistribution.UNIFORM,
                description="Unique order identifier",
            ),
            create_categorical_column(
                name="product_category",
                values=["Electronics", "Clothing", "Home & Garden", "Sports", "Books"],
                frequencies=[0.3, 0.25, 0.2, 0.15, 0.1],
                description="Product category",
            ),
            create_categorical_column(
                name="customer_status",
                values=["New", "Regular", "Premium", "VIP"],
                frequencies=[0.4, 0.35, 0.15, 0.1],
                description="Customer loyalty status",
            ),
            create_numeric_column(
                name="order_amount",
                is_float=True,
                min_value=10.0,
                max_value=5000.0,
                mean=150.0,
                std=200.0,
                decimals=2,
                distribution=NumericDistribution.EXPONENTIAL,
                description="Order total in USD",
            ),
            create_numeric_column(
                name="quantity",
                is_float=False,
                min_value=1,
                max_value=20,
                mean=3,
                std=2,
                distribution=NumericDistribution.POISSON,
                description="Number of items ordered",
            ),
            create_datetime_column(
                name="order_date",
                start_date="2023-01-01",
                end_date="2024-12-31",
                include_time=True,
                description="Date and time of order",
            ),
            create_categorical_column(
                name="shipping_method",
                values=["Standard", "Express", "Overnight", "International"],
                frequencies=[0.5, 0.3, 0.15, 0.05],
                description="Shipping method selected",
            ),
            create_categorical_column(
                name="payment_method",
                values=["Credit Card", "PayPal", "Debit Card", "Bank Transfer"],
                frequencies=[0.45, 0.30, 0.20, 0.05],
                description="Payment method used",
            ),
            create_numeric_column(
                name="customer_age",
                is_float=False,
                min_value=18,
                max_value=80,
                mean=35,
                std=12,
                distribution=NumericDistribution.NORMAL,
                description="Customer age",
            ),
            create_categorical_column(
                name="order_status",
                values=["Pending", "Processing", "Shipped", "Delivered", "Cancelled"],
                frequencies=[0.05, 0.10, 0.15, 0.65, 0.05],
                description="Current order status",
            ),
        ]

        return DatasetSchema(
            name="ecommerce_orders",
            domain="e-commerce",
            description="Synthetic e-commerce order dataset",
            columns=columns,
            num_rows=num_rows,
        )

    def generate_healthcare_schema(self, num_rows: int = 1000) -> DatasetSchema:
        """Generate a typical healthcare dataset schema."""
        columns = [
            create_numeric_column(
                name="patient_id",
                is_float=False,
                min_value=10000,
                max_value=99999,
                distribution=NumericDistribution.UNIFORM,
                description="Unique patient identifier",
            ),
            create_numeric_column(
                name="age",
                is_float=False,
                min_value=0,
                max_value=100,
                mean=45,
                std=20,
                distribution=NumericDistribution.NORMAL,
                description="Patient age",
            ),
            create_categorical_column(
                name="gender",
                values=["Male", "Female", "Other"],
                frequencies=[0.48, 0.48, 0.04],
                description="Patient gender",
            ),
            create_numeric_column(
                name="blood_pressure_systolic",
                is_float=False,
                min_value=90,
                max_value=180,
                mean=120,
                std=15,
                distribution=NumericDistribution.NORMAL,
                description="Systolic blood pressure (mmHg)",
            ),
            create_numeric_column(
                name="blood_pressure_diastolic",
                is_float=False,
                min_value=60,
                max_value=120,
                mean=80,
                std=10,
                distribution=NumericDistribution.NORMAL,
                description="Diastolic blood pressure (mmHg)",
            ),
            create_numeric_column(
                name="heart_rate",
                is_float=False,
                min_value=50,
                max_value=120,
                mean=75,
                std=12,
                distribution=NumericDistribution.NORMAL,
                description="Heart rate (bpm)",
            ),
            create_numeric_column(
                name="temperature",
                is_float=True,
                min_value=36.0,
                max_value=40.0,
                mean=37.0,
                std=0.5,
                decimals=1,
                distribution=NumericDistribution.NORMAL,
                description="Body temperature (°C)",
            ),
            create_categorical_column(
                name="diagnosis",
                values=["Hypertension", "Diabetes", "Asthma", "Healthy", "Other"],
                frequencies=[0.25, 0.20, 0.15, 0.30, 0.10],
                description="Primary diagnosis",
            ),
            create_categorical_column(
                name="treatment_status",
                values=["Admitted", "Outpatient", "Discharged", "Follow-up"],
                frequencies=[0.15, 0.40, 0.30, 0.15],
                description="Current treatment status",
            ),
            create_datetime_column(
                name="visit_date",
                start_date="2023-01-01",
                end_date="2024-12-31",
                include_time=True,
                business_hours_only=True,
                weekdays_only=True,
                description="Date of hospital visit",
            ),
        ]

        return DatasetSchema(
            name="patient_records",
            domain="healthcare",
            description="Synthetic patient health records dataset",
            columns=columns,
            num_rows=num_rows,
        )

    def generate_finance_schema(self, num_rows: int = 1000) -> DatasetSchema:
        """Generate a typical financial transactions dataset schema."""
        columns = [
            create_numeric_column(
                name="transaction_id",
                is_float=False,
                min_value=100000,
                max_value=999999,
                distribution=NumericDistribution.UNIFORM,
                description="Unique transaction identifier",
            ),
            create_categorical_column(
                name="account_type",
                values=["Checking", "Savings", "Credit", "Investment"],
                frequencies=[0.40, 0.30, 0.20, 0.10],
                description="Account type",
            ),
            create_numeric_column(
                name="transaction_amount",
                is_float=True,
                min_value=0.01,
                max_value=50000.0,
                mean=500.0,
                std=2000.0,
                decimals=2,
                distribution=NumericDistribution.EXPONENTIAL,
                description="Transaction amount in USD",
            ),
            create_categorical_column(
                name="transaction_type",
                values=["Deposit", "Withdrawal", "Transfer", "Payment", "Fee"],
                frequencies=[0.25, 0.25, 0.25, 0.20, 0.05],
                description="Type of transaction",
            ),
            create_categorical_column(
                name="merchant_category",
                values=[
                    "Groceries",
                    "Dining",
                    "Gas",
                    "Entertainment",
                    "Bills",
                    "Shopping",
                    "Travel",
                    "Other",
                ],
                frequencies=[0.20, 0.15, 0.10, 0.10, 0.15, 0.15, 0.10, 0.05],
                description="Merchant category",
            ),
            create_datetime_column(
                name="transaction_date",
                start_date="2023-01-01",
                end_date="2024-12-31",
                include_time=True,
                description="Date and time of transaction",
            ),
            create_categorical_column(
                name="transaction_status",
                values=["Completed", "Pending", "Failed", "Reversed"],
                frequencies=[0.85, 0.10, 0.03, 0.02],
                description="Transaction status",
            ),
            create_numeric_column(
                name="account_balance",
                is_float=True,
                min_value=0.0,
                max_value=100000.0,
                mean=5000.0,
                std=10000.0,
                decimals=2,
                distribution=NumericDistribution.EXPONENTIAL,
                description="Account balance after transaction",
            ),
            create_categorical_column(
                name="customer_tier",
                values=["Basic", "Silver", "Gold", "Platinum"],
                frequencies=[0.50, 0.30, 0.15, 0.05],
                description="Customer service tier",
            ),
        ]

        return DatasetSchema(
            name="financial_transactions",
            domain="finance",
            description="Synthetic financial transaction dataset",
            columns=columns,
            num_rows=num_rows,
        )

    def generate_custom_schema(
        self,
        name: str,
        domain: str,
        column_specs: List[dict],
        num_rows: int = 10000,
        description: Optional[str] = None,
    ) -> DatasetSchema:
        """
        Generate a custom schema from column specifications.

        Args:
            name: Dataset name
            domain: Domain context
            column_specs: List of column specification dicts
            num_rows: Number of rows to generate
            description: Optional dataset description

        Example column_specs:
        [
            {
                "name": "age",
                "type": "numeric_int",
                "min": 18,
                "max": 65,
                "distribution": "normal",
                "mean": 35,
                "std": 10
            },
            {
                "name": "status",
                "type": "categorical",
                "values": ["active", "inactive"],
                "frequencies": [0.7, 0.3]
            },
            {
                "name": "created_at",
                "type": "datetime",
                "start_date": "2023-01-01",
                "end_date": "2024-12-31",
                "include_time": True
            }
        ]
        """
        columns = []

        for spec in column_specs:
            col_type = spec.get("type")

            if col_type in ["numeric_int", "numeric_float"]:
                columns.append(
                    create_numeric_column(
                        name=spec["name"],
                        is_float=(col_type == "numeric_float"),
                        min_value=spec.get("min"),
                        max_value=spec.get("max"),
                        mean=spec.get("mean"),
                        std=spec.get("std"),
                        decimals=spec.get("decimals", 2),
                        distribution=NumericDistribution(
                            spec.get("distribution", "normal")
                        ),
                        description=spec.get("description"),
                    )
                )
            elif col_type == "categorical":
                columns.append(
                    create_categorical_column(
                        name=spec["name"],
                        values=spec["values"],
                        frequencies=spec.get("frequencies"),
                        description=spec.get("description"),
                    )
                )
            elif col_type == "datetime":
                columns.append(
                    create_datetime_column(
                        name=spec["name"],
                        start_date=spec.get("start_date", "2020-01-01"),
                        end_date=spec.get("end_date", "2024-12-31"),
                        include_time=spec.get("include_time", False),
                        business_hours_only=spec.get("business_hours_only", False),
                        weekdays_only=spec.get("weekdays_only", False),
                        description=spec.get("description"),
                    )
                )

        return DatasetSchema(
            name=name,
            domain=domain,
            description=description or f"Custom {domain} dataset",
            columns=columns,
            num_rows=num_rows,
        )

    # LLM-assisted methods (optional, require llm_client)

    def generate_categorical_values_with_llm(
        self, column_name: str, domain: str, count: int = 20
    ) -> List[str]:
        """
        Use LLM to generate realistic categorical values.

        Args:
            column_name: Name of the column
            domain: Domain context
            count: Number of values to generate

        Returns:
            List of categorical values
        """
        if self.llm_client is None:
            raise ValueError("LLM client required for LLM-assisted generation")

        prompt = f"""Generate {count} realistic values for a '{column_name}' column in a {domain} dataset.
Return only the values as a comma-separated list, nothing else.
Example format: value1, value2, value3

Values for {column_name}:"""

        response = self.llm_client.generate(prompt)
        # Parse comma-separated values
        values = [v.strip() for v in response.split(",") if v.strip()]
        return values[:count]  # Ensure we don't exceed count

    def generate_schema_with_llm(
        self, domain: str, num_columns: int = 10, num_rows: int = 10000
    ) -> DatasetSchema:
        """
        Use LLM to suggest a complete schema for a domain (legacy method).

        Args:
            domain: Domain context (e.g., "e-commerce", "healthcare")
            num_columns: Number of columns to generate
            num_rows: Number of rows in the dataset

        Returns:
            Generated DatasetSchema
        """
        if self.llm_client is None:
            raise ValueError("LLM client required for LLM-assisted generation")

        # Use the new description-based method
        description = f"A {domain} dataset with typical fields for this domain"
        return self.generate_schema_from_description(
            description, num_rows=num_rows, num_columns=num_columns
        )

    def generate_schema_from_description(
        self,
        description: str,
        num_rows: int = 1000,
        num_columns: Optional[int] = None,
    ) -> DatasetSchema:
        """
        Generate a complete schema from a natural language description.

        This is the PRIMARY LLM use case: User describes their needs,
        LLM generates the schema config, then hardcoded generators produce data.

        Args:
            description: Natural language description of the desired dataset
            num_rows: Number of rows to generate
            num_columns: Optional hint for number of columns

        Returns:
            DatasetSchema that can be used with the pipeline

        Example:
            >>> generator = SchemaGenerator(llm_client=ollama_client)
            >>> schema = generator.generate_schema_from_description(
            ...     "I need a dataset for an e-commerce checkout process"
            ... )
            >>> df = build_clean_dataset(schema)

        Raises:
            ValueError: If no LLM client configured
            RuntimeError: If schema generation or parsing fails
        """
        if self.llm_client is None:
            raise ValueError(
                "LLM client required for description-based generation. "
                "Initialize SchemaGenerator with an OllamaClient."
            )

        # Get schema config from LLM
        try:
            config = self.llm_client.generate_schema_config(
                description, num_rows=num_rows, num_columns=num_columns
            )
        except Exception as e:
            raise RuntimeError(f"Failed to generate schema config: {e}")

        # Parse config into DatasetSchema
        try:
            return self._parse_schema_config(config)
        except Exception as e:
            raise RuntimeError(
                f"Failed to parse schema config: {e}\n"
                f"Config: {config}"
            )

    def _parse_schema_config(self, config: dict) -> DatasetSchema:
        """
        Parse LLM-generated schema config into DatasetSchema object.

        Args:
            config: Schema configuration dictionary from LLM

        Returns:
            DatasetSchema object
        """
        columns = []

        for col_config in config.get("columns", []):
            col_type = col_config.get("type", "").lower()
            col_name = col_config.get("name", "")
            col_desc = col_config.get("description", "")

            if "numeric" in col_type:
                # Parse numeric column
                is_float = "float" in col_type
                distribution_str = col_config.get("distribution", "uniform").upper()

                try:
                    distribution = NumericDistribution[distribution_str]
                except KeyError:
                    distribution = NumericDistribution.UNIFORM  # Default to uniform

                # Set sensible defaults based on column type
                if is_float:
                    default_min, default_max = 0.0, 1000.0
                else:
                    default_min, default_max = 1, 10000

                # Create numeric config with defaults
                numeric_config = NumericConfig(
                    distribution=distribution,
                    min_value=col_config.get("min_value", default_min),
                    max_value=col_config.get("max_value", default_max),
                    mean=col_config.get("mean"),
                    std=col_config.get("std"),
                )

                column = ColumnSchema(
                    name=col_name,
                    type=ColumnType.NUMERIC_FLOAT if is_float else ColumnType.NUMERIC_INT,
                    numeric_config=numeric_config,
                    description=col_desc,
                )

            elif col_type == "categorical":
                # Parse categorical column
                # LLM generates "categories" but our model uses "values"
                values = col_config.get("categories", col_config.get("values", []))
                frequencies = col_config.get("frequencies")

                categorical_config = CategoricalConfig(
                    values=values,
                    frequencies=frequencies,
                )

                column = ColumnSchema(
                    name=col_name,
                    type=ColumnType.CATEGORICAL,
                    categorical_config=categorical_config,
                    description=col_desc,
                )

            elif col_type == "datetime":
                # Parse datetime column
                datetime_config = DatetimeConfig(
                    start_date=col_config.get("start_date", "2023-01-01"),
                    end_date=col_config.get("end_date", "2024-12-31"),
                    business_hours_only=col_config.get("business_hours_only", False),
                    weekdays_only=col_config.get("weekdays_only", False),
                )

                column = ColumnSchema(
                    name=col_name,
                    type=ColumnType.DATETIME,
                    datetime_config=datetime_config,
                    description=col_desc,
                )

            else:
                # Unknown type, skip or default to categorical
                continue

            columns.append(column)

        return DatasetSchema(
            name=config.get("name", "generated_schema"),
            domain=config.get("domain", "general"),
            columns=columns,
            num_rows=config.get("num_rows", 1000),
        )
