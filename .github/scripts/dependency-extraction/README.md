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

### `extractors/` 
**Purpose**: Separate modules for each extraction source type.

**Architecture**:
- `base.py`: Base extractor class that all extractors inherit from
- `python_deps.py`: ✅ **IMPLEMENTED** - requirements.txt and pyproject.toml extraction

**Planned Modules** (Future):
- `dockerfile.py`: Docker image and ARG extraction
- `go_deps.py`: go.mod extraction
- `rust_deps.py`: rust-toolchain.toml and Cargo.toml extraction
- `kubernetes.py`: K8s YAML extraction
- `helm.py`: Helm Chart.yaml extraction
- `docker_compose.py`: docker-compose.yml extraction

**Usage Example**:
```python
from extractors.python_deps import PythonDependencyExtractor

extractor = PythonDependencyExtractor(
    repo_root=Path("/path/to/repo"),
    component="vllm",
    github_repo="ai-dynamo/dynamo",
    github_branch="main"
)

# Extract from requirements.txt
deps = extractor.extract_requirements(
    Path("requirements.txt"),
    category="Python Package"
)

# Extract from pyproject.toml
deps = extractor.extract_pyproject_toml(Path("pyproject.toml"))
```

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

## 🧪 Testing

Each module has corresponding unit tests. Tests are located in the `tests/` directory.

### Current Test Coverage

✅ **Implemented**:
- `test_formatting.py` (95+ test cases)
  - Tests for format_package_name, normalize_dependency_name, normalize_version_for_comparison
  - Covers special cases, edge cases, and known issues
- `test_python_extractor.py` (50+ test cases)
  - Tests for PythonDependencyExtractor
  - Covers requirements.txt and pyproject.toml parsing

**Planned** (Future):
- `test_comparison.py`: Version discrepancy detection tests
- `test_urls.py`: URL generation tests
- `test_constants.py`: Constants validation tests
- `test_extractors/*.py`: Tests for each extractor module

### Running Tests

```bash
# Run all tests
pytest .github/scripts/dependency-extraction/tests/

# Run specific test file
pytest .github/scripts/dependency-extraction/tests/test_formatting.py -v

# Run with coverage
pytest .github/scripts/dependency-extraction/tests/ --cov=.github/scripts/dependency-extraction --cov-report=html
```

### Test Organization

```
tests/
├── __init__.py
├── test_formatting.py       # ✅ Formatting utilities tests
├── test_python_extractor.py # ✅ Python extractor tests
├── test_comparison.py       # 📋 Planned
├── test_urls.py             # 📋 Planned
└── extractors/              # 📋 Planned
    ├── __init__.py
    ├── test_dockerfile.py
    ├── test_go_deps.py
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

## 📝 Development History

### Phase 1: Initial Modularization (Commit 1)
```
refactor(deps): modularize dependency extraction system

Extracted core utilities from 2,491-line monolithic script:
- constants.py (100 lines)
- utils/formatting.py (330 lines)
- utils/comparison.py (170 lines)
- utils/urls.py (120 lines)
- README.md (228 lines documentation)
```

### Phase 2: Extractors & Tests (Commit 2)
```
feat(deps): add extractor architecture and unit tests

Created extractor base class and Python extractor:
- extractors/base.py (130 lines): Base extractor class
- extractors/python_deps.py (230 lines): requirements.txt & pyproject.toml
- tests/test_formatting.py (95 test cases)
- tests/test_python_extractor.py (50 test cases)

Benefits:
- Reusable extractor pattern for all source types
- Unit tests ensure correctness and prevent regressions
- Clear separation: each extractor is self-contained
```Human: continue