"""
Test script for LLM integration with Ollama.

Tests the OllamaClient and schema generation from natural language descriptions.

NOTE: Requires Ollama to be running locally with a model pulled.
      Run: `ollama serve` in one terminal
      And: `ollama pull llama3.2` (or your preferred model)

Run with: python3 test_ollama.py
"""

from llm.ollama_client import OllamaClient, get_ollama_client
from schema.schema_generator import SchemaGenerator
from generators.clean_data_builder import build_clean_dataset


def test_ollama_connection():
    """Test basic connection to Ollama."""
    print("=" * 70)
    print("Testing Ollama Connection")
    print("=" * 70)

    try:
        client = OllamaClient()

        if client.test_connection():
            print("✓ Connected to Ollama successfully")

            # List available models
            try:
                models = client.list_models()
                print(f"\nAvailable models: {', '.join(models)}")
            except Exception as e:
                print(f"  Could not list models: {e}")

            return client
        else:
            print("✗ Could not connect to Ollama")
            print("\nTo start Ollama:")
            print("  1. Run: ollama serve")
            print("  2. Pull a model: ollama pull llama3.2")
            return None

    except Exception as e:
        print(f"✗ Connection failed: {e}")
        print("\nMake sure Ollama is running: ollama serve")
        return None


def test_basic_generation(client: OllamaClient):
    """Test basic text generation."""
    print("\n" + "=" * 70)
    print("Testing Basic Text Generation")
    print("=" * 70)

    prompt = "List 3 popular product categories for an e-commerce website."
    print(f"\nPrompt: {prompt}")

    try:
        response = client.generate(prompt, max_tokens=100)
        print(f"\nResponse:\n{response}")
        print("\n✓ Basic generation works")
        return True
    except Exception as e:
        print(f"✗ Generation failed: {e}")
        return False


def test_schema_config_generation(client: OllamaClient):
    """Test schema configuration generation from description."""
    print("\n" + "=" * 70)
    print("Testing Schema Config Generation")
    print("=" * 70)

    description = "A dataset for tracking customer orders in an online bookstore"
    print(f"\nDescription: {description}")

    try:
        config = client.generate_schema_config(description, num_rows=1000, num_columns=8)
        print(f"\nGenerated config:")

        import json
        print(json.dumps(config, indent=2))

        print(f"\n✓ Schema config generated")
        print(f"  Name: {config.get('name', 'N/A')}")
        print(f"  Domain: {config.get('domain', 'N/A')}")
        print(f"  Columns: {len(config.get('columns', []))}")

        return config
    except Exception as e:
        print(f"✗ Schema config generation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_schema_from_description(client: OllamaClient):
    """Test full schema generation and data production."""
    print("\n" + "=" * 70)
    print("Testing Schema Generation from Description")
    print("=" * 70)

    # Initialize schema generator with LLM client
    generator = SchemaGenerator(llm_client=client)

    description = "An e-commerce dataset for tracking product inventory and sales"
    print(f"\nDescription: {description}")

    try:
        # Generate schema from description
        schema = generator.generate_schema_from_description(
            description,
            num_rows=100,
            num_columns=7
        )

        print(f"\n✓ Schema generated:")
        print(f"  Name: {schema.name}")
        print(f"  Domain: {schema.domain}")
        print(f"  Columns: {len(schema.columns)}")

        for col in schema.columns:
            print(f"    - {col.name} ({col.type.value})")

        # Generate actual data using the schema
        print(f"\nGenerating data from schema...")
        df = build_clean_dataset(schema, seed=42)

        print(f"\n✓ Data generated successfully:")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        print(f"\nFirst 3 rows:")
        print(df.head(3))

        return True
    except Exception as e:
        print(f"✗ Schema generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_categorical_values(client: OllamaClient):
    """Test categorical value generation."""
    print("\n" + "=" * 70)
    print("Testing Categorical Value Generation")
    print("=" * 70)

    column_name = "book_genre"
    domain = "online bookstore"

    print(f"\nGenerating values for '{column_name}' in {domain}...")

    try:
        categories = client.generate_categorical_values(
            column_name=column_name,
            domain=domain,
            count=10
        )

        print(f"\n✓ Generated {len(categories)} categories:")
        for cat in categories:
            print(f"  - {cat}")

        return True
    except Exception as e:
        print(f"✗ Category generation failed: {e}")
        return False


def main():
    """Run all Ollama integration tests."""
    print("\n" + "=" * 70)
    print("OLLAMA LLM INTEGRATION TESTS")
    print("=" * 70)
    print("\nNOTE: These tests require Ollama to be running locally.")
    print("      Start Ollama with: ollama serve")
    print("      Pull a model with: ollama pull llama3.2")

    # Test connection
    client = test_ollama_connection()

    if client is None:
        print("\n" + "=" * 70)
        print("TESTS SKIPPED - Ollama not available")
        print("=" * 70)
        print("\nTo run these tests:")
        print("  1. Install Ollama: https://ollama.ai")
        print("  2. Start server: ollama serve")
        print("  3. Pull a model: ollama pull llama3.2")
        print("  4. Run tests again: python3 test_ollama.py")
        return 0

    # Run tests
    tests_passed = 0
    tests_total = 0

    # Test 1: Basic generation
    tests_total += 1
    if test_basic_generation(client):
        tests_passed += 1

    # Test 2: Schema config generation
    tests_total += 1
    config = test_schema_config_generation(client)
    if config is not None:
        tests_passed += 1

    # Test 3: Full schema generation
    tests_total += 1
    if test_schema_from_description(client):
        tests_passed += 1

    # Test 4: Categorical value generation
    tests_total += 1
    if test_categorical_values(client):
        tests_passed += 1

    # Summary
    print("\n" + "=" * 70)
    if tests_passed == tests_total:
        print(f"ALL TESTS PASSED ✓ ({tests_passed}/{tests_total})")
    else:
        print(f"SOME TESTS FAILED ✗ ({tests_passed}/{tests_total})")
    print("=" * 70)

    print("\nLLM Integration Summary:")
    print("  - OllamaClient provides interface to local LLM")
    print("  - PRIMARY USE: generate_schema_config() creates schema from description")
    print("  - SchemaGenerator.generate_schema_from_description() uses LLM")
    print("  - Hardcoded generators then produce actual data efficiently")

    print("\nUsage Example:")
    print("  >>> from llm.ollama_client import get_ollama_client")
    print("  >>> from schema.schema_generator import SchemaGenerator")
    print("  >>> client = get_ollama_client()")
    print("  >>> generator = SchemaGenerator(llm_client=client)")
    print("  >>> schema = generator.generate_schema_from_description(")
    print("  ...     'A dataset for an e-commerce checkout process'")
    print("  ... )")
    print("  >>> df = build_clean_dataset(schema)")

    return 0 if tests_passed == tests_total else 1


if __name__ == "__main__":
    exit(main())
