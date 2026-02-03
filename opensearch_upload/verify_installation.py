#!/usr/bin/env python3
"""
Verify Error Classification System Installation

Checks that all components are installed and configured correctly.

Usage:
    python3 verify_installation.py
"""

import os
import sys
from typing import List, Tuple


def check_python_version() -> Tuple[bool, str]:
    """Check Python version >= 3.8."""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        return True, f"✓ Python {version.major}.{version.minor}.{version.micro}"
    return False, f"✗ Python {version.major}.{version.minor}.{version.micro} (need >= 3.8)"


def check_imports() -> List[Tuple[bool, str]]:
    """Check required package imports."""
    results = []

    # Check anthropic
    try:
        import anthropic
        results.append((True, f"✓ anthropic {anthropic.__version__}"))
    except ImportError:
        results.append((False, "✗ anthropic not installed (pip install anthropic>=0.40.0)"))

    # Check opensearch-py
    try:
        import opensearchpy
        results.append((True, f"✓ opensearch-py {opensearchpy.__version__}"))
    except ImportError:
        results.append((False, "✗ opensearch-py not installed (pip install opensearch-py>=2.3.0)"))

    # Check requests
    try:
        import requests
        results.append((True, f"✓ requests {requests.__version__}"))
    except ImportError:
        results.append((False, "✗ requests not installed (pip install requests>=2.28.0)"))

    return results


def check_module_imports() -> List[Tuple[bool, str]]:
    """Check error_classification module imports."""
    results = []

    try:
        from error_classification import Config
        results.append((True, "✓ error_classification.Config"))
    except ImportError as e:
        results.append((False, f"✗ error_classification.Config: {e}"))

    try:
        from error_classification import ErrorClassifier
        results.append((True, "✓ error_classification.ErrorClassifier"))
    except ImportError as e:
        results.append((False, f"✗ error_classification.ErrorClassifier: {e}"))

    try:
        from error_classification import ErrorExtractor
        results.append((True, "✓ error_classification.ErrorExtractor"))
    except ImportError as e:
        results.append((False, f"✗ error_classification.ErrorExtractor: {e}"))

    try:
        from error_classification import ErrorDeduplicator
        results.append((True, "✓ error_classification.ErrorDeduplicator"))
    except ImportError as e:
        results.append((False, f"✗ error_classification.ErrorDeduplicator: {e}"))

    try:
        from error_classification import ClaudeClient
        results.append((True, "✓ error_classification.ClaudeClient"))
    except ImportError as e:
        results.append((False, f"✗ error_classification.ClaudeClient: {e}"))

    return results


def check_environment_variables() -> List[Tuple[bool, str, str]]:
    """Check required environment variables."""
    checks = [
        ("ANTHROPIC_API_KEY", "Required for Claude API", True),
        ("OPENSEARCH_URL", "Required for OpenSearch", False),
        ("ERROR_CLASSIFICATION_INDEX", "Recommended for OpenSearch", False),
        ("GITHUB_TOKEN", "Required for error analysis", False),
    ]

    results = []
    for var_name, description, required in checks:
        value = os.getenv(var_name)
        if value:
            # Mask sensitive values
            if "KEY" in var_name or "TOKEN" in var_name or "PASSWORD" in var_name:
                display_value = value[:10] + "..." if len(value) > 10 else "***"
            else:
                display_value = value

            results.append((True, f"✓ {var_name}", display_value))
        else:
            status = "✗" if required else "⚠️"
            results.append((not required, f"{status} {var_name}", f"Not set - {description}"))

    return results


def check_file_structure() -> List[Tuple[bool, str]]:
    """Check that required files exist."""
    required_files = [
        "error_classification/__init__.py",
        "error_classification/config.py",
        "error_classification/classifier.py",
        "error_classification/claude_client.py",
        "error_classification/deduplicator.py",
        "error_classification/error_extractor.py",
        "error_classification/opensearch_schema.py",
        "error_classification/prompts.py",
        "analyze_recent_errors.py",
        "upload_error_classifications.py",
        "requirements.txt",
    ]

    results = []
    for file_path in required_files:
        full_path = os.path.join(os.path.dirname(__file__), file_path)
        if os.path.exists(full_path):
            results.append((True, f"✓ {file_path}"))
        else:
            results.append((False, f"✗ {file_path} (missing)"))

    return results


def main():
    """Run all verification checks."""
    print("=" * 70)
    print("ERROR CLASSIFICATION SYSTEM - INSTALLATION VERIFICATION")
    print("=" * 70)
    print()

    all_passed = True

    # Python version
    print("1. Python Version")
    print("-" * 70)
    passed, message = check_python_version()
    print(f"   {message}")
    all_passed = all_passed and passed
    print()

    # Package imports
    print("2. Required Packages")
    print("-" * 70)
    results = check_imports()
    for passed, message in results:
        print(f"   {message}")
        all_passed = all_passed and passed
    print()

    # Module imports
    print("3. Error Classification Module")
    print("-" * 70)
    results = check_module_imports()
    for passed, message in results:
        print(f"   {message}")
        all_passed = all_passed and passed
    print()

    # Environment variables
    print("4. Environment Variables")
    print("-" * 70)
    results = check_environment_variables()
    for passed, message, description in results:
        print(f"   {message}: {description}")
        all_passed = all_passed and passed
    print()

    # File structure
    print("5. File Structure")
    print("-" * 70)
    results = check_file_structure()
    for passed, message in results:
        print(f"   {message}")
        all_passed = all_passed and passed
    print()

    # Summary
    print("=" * 70)
    if all_passed:
        print("✅ ALL CHECKS PASSED - System is ready to use!")
        print()
        print("Next steps:")
        print("  1. Run Phase 0 validation: python3 analyze_recent_errors.py --hours 48")
        print("  2. Test batch classification: python3 upload_error_classifications.py --hours 24")
        print("  3. See SETUP_GUIDE.md for full deployment instructions")
    else:
        print("❌ SOME CHECKS FAILED - Please fix the issues above")
        print()
        print("Common fixes:")
        print("  - Install packages: pip install -r requirements.txt")
        print("  - Set environment variables: export ANTHROPIC_API_KEY=...")
        print("  - Check file paths and permissions")
    print("=" * 70)

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
