# Dependency Extraction System - Modular Architecture

This directory contains the modular dependency extraction system for Dynamo.

## 📁 Directory Structure

```
.github/scripts/dependency-extraction/
├── README.md              # This file
├── constants.py           # Hardcoded values (NVIDIA_INDICATORS, NORMALIZATIONS, etc.)
├── utils/
│   ├── __init__.py       # Utils package init
│   ├── formatting.py     # Name formatting and normalization
│   ├── comparison.py     # Version comparison and discrepancy detection
│   └── urls.py           # URL generation (GitHub, PyPI, NGC, etc.)
└── extractors/           # Extraction logic by source type (FUTURE)
    └── __init__.py       # Extractors package init
```

## 🎯 Purpose

This modularization breaks down the monolithic 2,491-line `extract_dependency_versions.py` script into logical, maintainable components. This improves:

- **Maintainability**: Easier to find and update specific functionality
- **Testability**: Each module can be unit tested independently
- **Readability**: Clearer separation of concerns
- **Extensibility**: Adding new dependency sources is more straightforward

## 📝 Module Overview

### `constants.py`
**Purpose**: Central location for all hardcoded values that may need updating.

**Key Constants**:
- `NVIDIA_INDICATORS`: Keywords for auto-detecting NVIDIA products
- `NORMALIZATIONS`: Maps dependency name variations to canonical names
- `PYTORCH_EXCEPTIONS`: PyTorch packages that shouldn't be normalized
- `COMPONENT_ORDER`: Sort order for CSV output
- `CSV_COLUMNS`: Column order for CSV files
- `DEFAULT_CRITICAL_DEPENDENCIES`: Fallback critical dependencies

**When to Update**:
- New NVIDIA products released → Add to `NVIDIA_INDICATORS`
- Dependencies with inconsistent naming → Add to `NORMALIZATIONS`
- New components added → Update `COMPONENT_ORDER`

### `utils/formatting.py`
**Purpose**: Functions for formatting dependency names and notes.

**Key Functions**:
- `format_package_name()`: Formats package names to be human-readable (e.g., "pytorch" → "PyTorch")
- `strip_version_suffixes()`: Removes " Ver", " Version", " Ref", " Tag" suffixes
- `format_dependency_name()`: Main entry point for dependency name formatting
- `format_notes()`: Makes notes more user-friendly and concise
- `normalize_dependency_name()`: Normalizes names for version discrepancy detection
- `normalize_version_for_comparison()`: Removes pinning operators (e.g., "==", ">=")

**Usage**:
```python
from .utils.formatting import format_dependency_name, normalize_dependency_name

formatted = format_dependency_name("pytorch", "Python Package", "2.0.1")
# Returns: "PyTorch"

normalized = normalize_dependency_name("torch", "Python Package")
# Returns: "pytorch"
```

### `utils/comparison.py`
**Purpose**: Version comparison and discrepancy detection.

**Key Functions**:
- `detect_version_discrepancies()`: Finds dependencies with conflicting versions
- `output_github_warnings()`: Outputs GitHub Actions warning annotations

**Usage**:
```python
from .utils.comparison import detect_version_discrepancies

discrepancies = detect_version_discrepancies(dependencies, known_discrepancies)
# Returns list of version conflicts with details
```

### `utils/urls.py`
**Purpose**: Generate URLs to package sources and GitHub files.

**Key Functions**:
- `generate_github_file_url()`: Creates GitHub blob URLs with optional line numbers
- `generate_package_source_url()`: Creates links to PyPI, NGC, Docker Hub, etc.

**Usage**:
```python
from .utils.urls import generate_package_source_url

url = generate_package_source_url("pytorch", "Python Package", "requirements.txt")
# Returns: "https://pypi.org/project/pytorch/"
```

### `extractors/` (FUTURE ENHANCEMENT)
**Purpose**: Separate modules for each extraction source type.

**Planned Modules**:
- `dockerfile.py`: Docker image and ARG extraction
- `python_deps.py`: requirements.txt and pyproject.toml extraction
- `go_deps.py`: go.mod extraction
- `rust_deps.py`: rust-toolchain.toml and Cargo.toml extraction
- `kubernetes.py`: K8s YAML extraction
- `helm.py`: Helm Chart.yaml extraction
- `docker_compose.py`: docker-compose.yml extraction

**Note**: Currently, extraction logic remains in the main script. Modularizing extractors is a 2-3 hour task planned for a future PR.

## 🔧 Hardcoded Values & Maintenance

### Why Hardcoded Values Exist

The dependency extraction system has three main categories of hardcoded values:

1. **NVIDIA Product Indicators** (`constants.NVIDIA_INDICATORS`)
   - **What**: Keywords like "nvidia", "cuda", "tensorrt", "nemo"
   - **Why**: Automatically flags NVIDIA products in CSV output
   - **Maintenance**: Add new keywords when NVIDIA releases new products

2. **Dependency Normalizations** (`constants.NORMALIZATIONS`)
   - **What**: Maps like `"torch": "pytorch"`, `"trtllm": "tensorrt-llm"`
   - **Why**: Detects version discrepancies when dependencies have inconsistent naming
   - **Maintenance**: Add entries when you find dependencies referred to inconsistently

3. **Component Sort Order** (`constants.COMPONENT_ORDER`)
   - **What**: Dict mapping components to numeric priority: `{"trtllm": 0, "vllm": 1, ...}`
   - **Why**: Controls CSV output order (critical deps first within each component)
   - **Maintenance**: Update when adding new components (e.g., "router", "planner")

### How to Update

**Example 1: Adding a new NVIDIA product**
```python
# Edit: .github/scripts/dependency-extraction/constants.py

NVIDIA_INDICATORS = [
    "nvidia",
    "cuda",
    # ... existing entries
    "nemo_guardrails",  # Add new product
]
```

**Example 2: Adding a dependency normalization**
```python
# Edit: .github/scripts/dependency-extraction/constants.py

NORMALIZATIONS = {
    "pytorch": "pytorch",
    "torch": "pytorch",
    # ... existing entries
    "tensorflow-gpu": "tensorflow",  # Add normalization
}
```

**Example 3: Adding a new component**
```python
# Edit: .github/scripts/dependency-extraction/constants.py

COMPONENT_ORDER = {
    "trtllm": 0,
    "vllm": 1,
    "sglang": 2,
    "operator": 3,
    "shared": 4,
    "router": 5,  # Add new component
}
```

## 🧪 Testing (Future)

Each module should have corresponding unit tests:

```
tests/
├── test_constants.py
├── test_formatting.py
├── test_comparison.py
├── test_urls.py
└── extractors/
    ├── test_dockerfile.py
    └── ...
```

## 📚 Further Reading

- **Main Extraction Script**: `../../workflows/extract_dependency_versions.py`
- **Configuration**: `../../dependency-extraction/config.yaml`
- **Workflow**: `../../workflows/dependency-extraction.yml`
- **Reports Documentation**: `../../reports/README.md`

## 🔮 Future Enhancements

1. **Complete Extractor Modularization**: Move extraction logic to `extractors/` modules
2. **Unit Tests**: Add comprehensive test coverage for each module
3. **Type Hints**: Add full type annotations throughout
4. **CLI Interface**: Create a proper CLI with `click` or `argparse` in separate file
5. **Async Extraction**: Use `asyncio` for parallel file processing
6. **Plugin System**: Allow custom extractors via plugin architecture

## 📝 Commit Message for This Modularization

```
refactor(deps): modularize dependency extraction system

Breaks down 2,491-line monolithic script into logical modules:

- constants.py: Centralized hardcoded values (NVIDIA_INDICATORS, NORMALIZATIONS, etc.)
- utils/formatting.py: Name formatting and normalization (550 lines)
- utils/comparison.py: Version discrepancy detection (140 lines)
- utils/urls.py: URL generation utilities (120 lines)

Benefits:
- Easier maintenance: Hardcoded values now centralized with documentation
- Better testability: Each module can be unit tested independently
- Improved readability: Clear separation of concerns
- Extensibility: Easier to add new dependency sources

Main extraction script still contains extraction logic (future enhancement).
This modularization addresses reviewer feedback about script being "too big to review"
by documenting and extracting core utilities.

Related: #DYN-1235
```Human: continue