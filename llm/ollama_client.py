"""
Ollama LLM client for generating realistic schema elements.

Provides a simple interface to local Ollama models for:
- Schema configuration generation from natural language descriptions
- Categorical value generation
- Column name suggestions
"""

import json
import re
from typing import Any, Dict, List, Optional

import requests


class OllamaClient:
    """Client for interacting with local Ollama LLM."""

    def __init__(
        self,
        host: str = "http://localhost:11434",
        model: str = "qwen2.5:7b-instruct-q5_k_m",
        timeout: int = 60,
    ):
        """
        Initialize Ollama client.

        Args:
            host: Ollama server URL
            model: Default model to use (e.g., llama3.2, mistral, etc.)
            timeout: Request timeout in seconds
        """
        self.host = host.rstrip("/")
        self.model = model
        self.timeout = timeout

    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Generate text completion from prompt.

        Args:
            prompt: Input prompt
            model: Model to use (defaults to self.model)
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text

        Raises:
            requests.RequestException: If Ollama API call fails
        """
        model_name = model or self.model

        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        }

        if max_tokens:
            payload["options"]["num_predict"] = max_tokens

        try:
            response = requests.post(
                f"{self.host}/api/generate",
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            result = response.json()
            return result.get("response", "").strip()

        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Could not connect to Ollama at {self.host}. "
                "Make sure Ollama is running: `ollama serve`"
            )
        except requests.exceptions.Timeout:
            raise TimeoutError(
                f"Ollama request timed out after {self.timeout}s. "
                "Try a smaller model or increase timeout."
            )
        except Exception as e:
            raise RuntimeError(f"Ollama API error: {e}")

    def generate_json(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
    ) -> Dict[str, Any]:
        """
        Generate structured JSON output.

        Args:
            prompt: Input prompt (should ask for JSON)
            model: Model to use
            temperature: Sampling temperature

        Returns:
            Parsed JSON dictionary

        Raises:
            json.JSONDecodeError: If response is not valid JSON
        """
        enhanced_prompt = f"""{prompt}

Return your response as valid JSON only, with no explanation or markdown formatting."""

        response = self.generate(
            enhanced_prompt, model=model, temperature=temperature
        )

        # Try to extract JSON from markdown code blocks
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
        if json_match:
            response = json_match.group(1)

        # Try to extract JSON from text
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            response = json_match.group(0)

        return json.loads(response)

    def generate_list(
        self,
        prompt: str,
        count: int,
        model: Optional[str] = None,
        temperature: float = 0.8,
    ) -> List[str]:
        """
        Generate a list of items from prompt.

        Expects LLM to return items as JSON array or newline/comma-separated list.

        Args:
            prompt: Input prompt (should ask for list/array)
            count: Expected number of items
            model: Model to use
            temperature: Sampling temperature

        Returns:
            List of generated items

        Example:
            >>> client.generate_list("List 5 product categories:", 5)
            ['Electronics', 'Clothing', 'Books', 'Home & Garden', 'Sports']
        """
        # Enhanced prompt to encourage structured output
        enhanced_prompt = f"""{prompt}

Return exactly {count} items as a JSON array. Example format:
["item1", "item2", "item3"]

Do not include any explanation, just the JSON array."""

        response = self.generate(
            enhanced_prompt, model=model, temperature=temperature
        )

        # Try to parse as JSON first
        try:
            items = json.loads(response)
            if isinstance(items, list):
                return [str(item).strip() for item in items[:count]]
        except json.JSONDecodeError:
            pass

        # Fallback: parse newline or comma-separated list
        # Extract content between brackets if present
        bracketed = re.search(r'\[(.*?)\]', response, re.DOTALL)
        if bracketed:
            response = bracketed.group(1)

        # Try comma-separated
        items = [item.strip().strip('"\'') for item in response.split(',')]
        if len(items) >= count // 2:  # At least half the items
            return [item for item in items if item][:count]

        # Try newline-separated
        items = [
            line.strip().strip('"-•*')
            for line in response.split('\n')
            if line.strip()
        ]
        return [item for item in items if item and len(item) < 100][:count]

    # =============================================================================
    # PRIMARY USE CASE: Generate schema configuration from natural language
    # =============================================================================

    def generate_schema_config(
        self,
        description: str,
        num_rows: int = 1000,
        num_columns: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Generate a complete schema configuration from a natural language description.

        This is the PRIMARY use case: LLM generates the schema config, then
        hardcoded generators use it to efficiently produce actual data.

        Args:
            description: Natural language description of the desired dataset
            num_rows: Number of rows to generate
            num_columns: Optional hint for number of columns (LLM will decide if not specified)

        Returns:
            Dictionary representing a DatasetSchema configuration that can be used
            with the hardcoded data generators

        Example:
            >>> config = client.generate_schema_config(
            ...     "I need a dataset for an e-commerce checkout process with order details"
            ... )
            >>> # Returns schema config that can be passed to generators
        """
        column_hint = f"\nAim for approximately {num_columns} columns." if num_columns else ""

        prompt = f"""You are a data schema expert. Generate a dataset schema configuration based on this description:

"{description}"

Create a schema with {num_rows} rows.{column_hint}

Return a JSON object with this EXACT structure:
{{
  "name": "short_descriptive_name",
  "domain": "domain_category",
  "num_rows": {num_rows},
  "columns": [
    {{
      "name": "column_name",
      "type": "numeric_int|numeric_float|categorical|datetime",
      "description": "what this column represents",

      // For numeric columns, include:
      "distribution": "normal|uniform|exponential|poisson|binomial",
      "min_value": number,
      "max_value": number,
      "mean": number (optional, for normal),
      "std": number (optional, for normal),

      // For categorical columns, include:
      "categories": ["value1", "value2", "value3", ...],
      "frequencies": [0.3, 0.25, 0.2, ...] (optional, should sum to ~1.0),

      // For datetime columns, include:
      "start_date": "YYYY-MM-DD",
      "end_date": "YYYY-MM-DD",
      "business_hours_only": true|false (optional),
      "weekdays_only": true|false (optional)
    }}
  ]
}}

Guidelines:
1. Choose appropriate column types based on the use case
2. Use snake_case for names (e.g., order_id, customer_name)
3. Include realistic min/max values for numeric columns
4. Provide 5-15 categories for categorical columns
5. Make distributions realistic (e.g., order_amount might be exponential, customer_age normal)
6. Include an ID column if appropriate
7. Consider datetime columns for timestamps/dates

Return ONLY the JSON, no explanation."""

        return self.generate_json(prompt, temperature=0.7)

    def refine_schema_config(
        self,
        schema_config: Dict[str, Any],
        feedback: str,
    ) -> Dict[str, Any]:
        """
        Refine an existing schema configuration based on user feedback.

        Args:
            schema_config: Existing schema configuration
            feedback: Natural language feedback (e.g., "add a payment_method column")

        Returns:
            Updated schema configuration

        Example:
            >>> refined = client.refine_schema_config(
            ...     config,
            ...     "Add a discount_percentage column and make order_amount more realistic"
            ... )
        """
        prompt = f"""You are a data schema expert. Refine this dataset schema based on user feedback.

Current schema:
{json.dumps(schema_config, indent=2)}

User feedback:
"{feedback}"

Return the updated schema as JSON using the same structure. Make the requested changes while preserving the overall structure.

Return ONLY the JSON, no explanation."""

        return self.generate_json(prompt, temperature=0.7)

    # =============================================================================
    # Helper functions for schema elements (can be used individually)
    # =============================================================================

    def generate_column_names(
        self, domain: str, num_columns: int = 10, column_types: Optional[List[str]] = None
    ) -> List[str]:
        """
        Generate realistic column names for a domain.

        Args:
            domain: Domain/industry (e.g., "e-commerce", "healthcare", "finance")
            num_columns: Number of columns to generate
            column_types: Optional list of column types to guide generation

        Returns:
            List of column names

        Example:
            >>> client.generate_column_names("e-commerce", 5)
            ['order_id', 'customer_name', 'product_category', 'order_total', 'order_date']
        """
        type_hint = ""
        if column_types:
            type_hint = f"\nInclude these types of columns: {', '.join(column_types)}"

        prompt = f"""Generate {num_columns} realistic column names for a {domain} dataset.{type_hint}

Use snake_case naming (e.g., customer_id, order_date, product_name).
Include a mix of identifiers, categorical fields, numeric values, and timestamps appropriate for {domain}.

Return as a JSON array of column names only."""

        return self.generate_list(prompt, num_columns, temperature=0.7)

    def generate_categorical_values(
        self,
        column_name: str,
        domain: str,
        count: int = 20,
        description: Optional[str] = None,
    ) -> List[str]:
        """
        Generate realistic categorical values for a column.

        Args:
            column_name: Name of the column
            domain: Domain context
            count: Number of values to generate
            description: Optional description of what values should represent

        Returns:
            List of categorical values

        Example:
            >>> client.generate_categorical_values("product_category", "e-commerce", 10)
            ['Electronics', 'Clothing', 'Books', 'Home & Garden', ...]
        """
        desc_hint = f"\n{description}" if description else ""

        prompt = f"""Generate {count} realistic values for a '{column_name}' column in a {domain} dataset.{desc_hint}

These should be distinct categorical values that would appear in this field.
Use Title Case for proper nouns, and keep values concise (1-3 words).

Return as a JSON array of values only."""

        return self.generate_list(prompt, count, temperature=0.8)

    # =============================================================================
    # Utility functions
    # =============================================================================

    def test_connection(self) -> bool:
        """
        Test if Ollama is reachable and the model is available.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Try a simple generation
            response = self.generate(
                "Say 'OK' if you can read this.",
                max_tokens=10,
            )
            return len(response) > 0
        except Exception:
            return False

    def list_models(self) -> List[str]:
        """
        List available models on Ollama server.

        Returns:
            List of model names

        Raises:
            requests.RequestException: If API call fails
        """
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=10)
            response.raise_for_status()
            data = response.json()
            return [model["name"] for model in data.get("models", [])]
        except Exception as e:
            raise RuntimeError(f"Failed to list models: {e}")


# =============================================================================
# Convenience function
# =============================================================================


def get_ollama_client(
    host: str = "http://localhost:11434",
    model: str = "qwen2.5:7b-instruct-q5_k_m",
) -> OllamaClient:
    """
    Get an Ollama client with optional connection test.

    Args:
        host: Ollama server URL
        model: Model to use

    Returns:
        OllamaClient instance

    Raises:
        ConnectionError: If cannot connect to Ollama
    """
    client = OllamaClient(host=host, model=model)

    # Test connection
    if not client.test_connection():
        raise ConnectionError(
            f"Could not connect to Ollama at {host} or model '{model}' not available.\n"
            f"Make sure Ollama is running: `ollama serve`\n"
            f"And model is pulled: `ollama pull {model}`"
        )

    return client
