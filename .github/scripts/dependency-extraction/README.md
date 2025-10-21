# Dependency Extraction System

## Overview

This system automatically extracts and tracks software dependencies across all Dynamo components (trtllm, vllm, sglang, operator, shared). It parses 10 different source types and generates comprehensive CSV reports with version tracking, critical dependency flagging, and version discrepancy detection.

## Architecture

### Directory Structure

```
.github/scripts/dependency-extraction/
├── README.md                    # This file
├── extract_dependencies.py      # Main CLI entry point
├── extractors/                  # Source-specific extractors
│   ├── __init__.py
│   ├── base.py                 # Base extractor class
│   ├── dockerfile.py           # Docker image & ARG extraction
│   ├── python_deps.py          # requirements.txt, pyproject.toml
│   ├── go_mod.py               # go.mod parsing
│   ├── helm.py                 # Helm Chart.yaml
│   ├── rust.py                 # Cargo.toml, rust-toolchain.toml
│   ├── kubernetes.py           # K8s recipe YAMLs
│   └── shell_scripts.py        # Shell script parsing
├── utils/                       # Utility modules
│   ├── __init__.py
│   ├── config.py               # Config loading and constants
│   ├── formatters.py           # Name/version formatting
│   ├── url_generators.py       # Package source URL generation
│   └── version_comparison.py   # Version normalization & discrepancy detection
└── core/                        # Core functionality
    ├── __init__.py
    └── extractor.py            # Main DependencyExtractor class
```

### Component Responsibilities

#### `extract_dependencies.py`
**Main entry point** - CLI argument parsing and orchestration
- Parses command-line arguments
- Initializes `DependencyExtractor`
- Runs extraction workflow
- Outputs CSV reports

#### `core/extractor.py`
**Central coordinator** - Manages the extraction process
- Coordinates all extractors
- Aggregates dependency data
- Tracks errors and warnings
- Generates output CSV
- Compares versions and detects changes

#### `extractors/`
**Source-specific parsers** - Each module handles one source type
- `dockerfile.py`: Parses Dockerfiles for base images, ARGs, binary downloads
- `python_deps.py`: Extracts from requirements.txt and pyproject.toml
- `go_mod.py`: Parses go.mod for direct/indirect dependencies
- `helm.py`: Reads Helm Chart.yaml for chart dependencies
- `rust.py`: Handles Cargo.toml and rust-toolchain.toml
- `kubernetes.py`: Parses K8s YAML files for container images
- `shell_scripts.py`: Extracts from install scripts (pip, wget, curl)

#### `utils/`
**Shared utilities** - Reusable helper functions
- `config.py`: Loads YAML config and defines constants
- `formatters.py`: Cleans up dependency names and formats notes
- `url_generators.py`: Generates package source URLs (PyPI, NGC, Docker Hub, etc.)
- `version_comparison.py`: Normalizes versions and detects discrepancies

---

## Configuration

### Config File Location
`.github/dependency-extraction/config.yaml`

### Config Structure

```yaml
# GitHub repository information
github:
  repo: "ai-dynamo/dynamo"
  branch: "main"

# Baseline dependency count (for warning on increases)
baseline:
  dependency_count: 251  # Fallback if latest CSV not found

# Critical dependencies (flagged in output)
critical_dependencies:
  - "CUDA"
  - "PyTorch"
  - "TensorRT-LLM"
  - "vLLM"
  - "SGLang"
  # ... (add more as needed)

# Component definitions (where to find dependencies)
components:
  trtllm:
    dockerfiles:
      - "container/Dockerfile.trtllm"
    requirements:
      - "components/backends/trtllm/requirements.txt"
    pyproject:
      - "components/backends/trtllm/pyproject.toml"
    scripts:
      - "container/deps/trtllm/install_nixl.sh"

  vllm:
    dockerfiles:
      - "container/Dockerfile.vllm"
    requirements:
      - "components/backends/vllm/requirements.txt"
    # ... (similar for other components)

  operator:
    go_mod:
      - "deploy/cloud/operator/go.mod"
    helm:
      - "deploy/cloud/helm/platform/Chart.yaml"

# Extraction rules
extraction:
  skip_go_indirect: true  # Skip indirect Go dependencies
  skip_test_deps: false   # Include test dependencies

# Known version discrepancies (intentional differences)
known_version_discrepancies:
  - dependency: "PyTorch"
    reason: "TensorRT-LLM uses NVIDIA container (2.8.0), vLLM uses 2.7.1+cu128 (ARM64 wheel compatibility)"
  - dependency: "torchvision"
    reason: "Matches corresponding PyTorch versions across components"
```

---

## Usage

### Command Line

```bash
# Basic usage (outputs to .github/reports/dependency_versions_<timestamp>.csv)
python3 .github/scripts/dependency-extraction/extract_dependencies.py

# Specify output path
python3 .github/scripts/dependency-extraction/extract_dependencies.py \
  --output /path/to/output.csv

# Compare against previous versions
python3 .github/scripts/dependency-extraction/extract_dependencies.py \
  --output output.csv \
  --previous-latest .github/reports/dependency_versions_latest.csv \
  --previous-release .github/reports/releases/dependency_versions_v0.6.0.csv

# Create release snapshot
python3 .github/scripts/dependency-extraction/extract_dependencies.py \
  --output .github/reports/releases/dependency_versions_v1.0.0.csv \
  --release 1.0.0

# Export removed dependencies
python3 .github/scripts/dependency-extraction/extract_dependencies.py \
  --output output.csv \
  --report-removed removed_deps.json

# Custom config
python3 .github/scripts/dependency-extraction/extract_dependencies.py \
  --config /path/to/custom_config.yaml
```

### Python API

```python
from pathlib import Path
from core.extractor import DependencyExtractor

# Initialize
extractor = DependencyExtractor(
    repo_root=Path("/path/to/dynamo"),
    github_repo="ai-dynamo/dynamo",
    github_branch="main",
    config_path=Path(".github/dependency-extraction/config.yaml"),
    previous_latest_csv=Path(".github/reports/dependency_versions_latest.csv"),
    previous_release_csv=Path(".github/reports/releases/dependency_versions_v0.6.0.csv")
)

# Run extraction
extractor.extract_all()

# Detect version discrepancies
discrepancies = extractor.detect_version_discrepancies()
for disc in discrepancies:
    print(f"{disc['normalized_name']}: {disc['versions']}")

# Write output
extractor.write_csv(Path("output.csv"))
```

---

## Adding New Dependency Sources

### 1. Create New Extractor

```python
# .github/scripts/dependency-extraction/extractors/new_source.py

from pathlib import Path
from .base import BaseExtractor

class NewSourceExtractor(BaseExtractor):
    """Extract dependencies from NewSource files."""

    def extract(self, file_path: Path, component: str) -> None:
        """
        Extract dependencies from a NewSource file.

        Args:
            file_path: Path to the source file
            component: Component name (trtllm, vllm, etc.)
        """
        if not file_path.exists():
            self.log_missing_file(file_path, component)
            return

        try:
            with open(file_path) as f:
                content = f.read()

            # Parse content and extract dependencies
            # ...

            self.add_dependency(
                component=component,
                category="NewSource Dependency",
                name="dependency-name",
                version="1.2.3",
                source_file=str(file_path.relative_to(self.repo_root)),
                line_number="10",
                notes="Extracted from NewSource file"
            )

        except Exception as e:
            self.log_failed_file(file_path, component, str(e))
```

### 2. Register in Config

```yaml
# .github/dependency-extraction/config.yaml

components:
  trtllm:
    new_source:  # Add new source type
      - "path/to/new_source_file"
```

### 3. Integrate in Main Extractor

```python
# .github/scripts/dependency-extraction/core/extractor.py

from extractors.new_source import NewSourceExtractor

class DependencyExtractor:
    def extract_all(self):
        # ... existing code ...

        # Add new source extraction
        for component, paths in self.config.get("components", {}).items():
            for file_path in paths.get("new_source", []):
                path = self.repo_root / file_path
                new_extractor = NewSourceExtractor(self.repo_root, self)
                new_extractor.extract(path, component)
```

---

## Maintenance

### Updating Critical Dependencies

Edit `.github/dependency-extraction/config.yaml`:

```yaml
critical_dependencies:
  - "CUDA"          # GPU compute platform
  - "PyTorch"       # ML framework
  - "TensorRT-LLM"  # Inference engine
  - "NewCriticalDep"  # Add here
```

### Adding Extraction Patterns

**For Dockerfiles** (`.github/scripts/dependency-extraction/extractors/dockerfile.py`):
- Add regex patterns to `extract()` method
- Handle new ARG formats or download patterns

**For Python** (`.github/scripts/dependency-extraction/extractors/python_deps.py`):
- Update `extract_requirements()` for new pip syntax
- Extend `extract_pyproject_toml()` for new pyproject sections

### Hardcoded Values & Constants

**Location:** `.github/scripts/dependency-extraction/utils/config.py`

```python
# NVIDIA product indicators (for auto-detection)
NVIDIA_INDICATORS = [
    "nvcr.io",      # NGC container registry
    "nvidia",       # NVIDIA packages
    "tensorrt",     # TensorRT inference
    "cuda",         # CUDA toolkit
    # Add more as needed
]

# Special cases for dependency name normalization
SPECIAL_CASES = {
    "pytorch": "PyTorch",
    "tensorflow": "TensorFlow",
    "kubernetes": "Kubernetes",
    # Add more as needed
}

# Category priorities for sorting
CATEGORY_PRIORITIES = {
    "Base Image": 1,
    "Runtime Image": 2,
    "Python Package": 3,
    # ... add more
}
```

**To Update:** Edit the constants in `utils/config.py` and document the reason for each entry.

### Known Version Discrepancies

When a version discrepancy is **intentional** (e.g., different PyTorch versions for different backends), document it in config:

```yaml
known_version_discrepancies:
  - dependency: "PyTorch"
    reason: "TensorRT-LLM uses NVIDIA container (2.8.0), vLLM uses 2.7.1+cu128 (ARM64 wheel compatibility)"
```

This will still report the discrepancy but mark it as "known" with the provided reason.

---

## Troubleshooting

### "Config file not found"
**Solution:** Ensure `.github/dependency-extraction/config.yaml` exists. The script uses this path by default.

### "No dependencies extracted"
**Solution:**
1. Check config file has correct component paths
2. Verify files exist at specified paths
3. Check file permissions
4. Run with `--verbose` for detailed logs

### "Version discrepancy false positives"
**Solution:**
1. Check `normalize_dependency_name()` in `utils/version_comparison.py`
2. Add exceptions for specific packages (e.g., "pytorch triton" is not PyTorch)
3. Update normalization rules for your use case

### "Import errors when running script"
**Solution:** Ensure you're in the repo root and using Python 3.10+:
```bash
cd /path/to/dynamo
python3 .github/scripts/dependency-extraction/extract_dependencies.py
```

---

## Testing

### Manual Testing

```bash
# Test full extraction
python3 .github/scripts/dependency-extraction/extract_dependencies.py \
  --output /tmp/test_deps.csv

# Verify output
cat /tmp/test_deps.csv | head -20

# Test specific component (temporarily modify config)
# ... edit config to only include one component ...
python3 .github/scripts/dependency-extraction/extract_dependencies.py --output /tmp/test.csv
```

### Unit Testing (Future)

```bash
# Run unit tests (when implemented)
pytest .github/scripts/dependency-extraction/tests/
```

---

## Workflow Integration

The extraction system is called by `.github/workflows/dependency-extraction.yml`:

- **Nightly:** Runs at 2 AM UTC, updates `dependency_versions_latest.csv`
- **Release:** Triggers on `release/*` branches, creates versioned snapshot

See workflow file for invocation details.

---

## Contributing

When modifying the extraction system:

1. **Update this README** if adding new features or changing architecture
2. **Test thoroughly** with sample files before committing
3. **Document constants** in `utils/config.py` if adding hardcoded values
4. **Follow code style** (black, isort, ruff)
5. **Sign commits** with DCO (`git commit -s`)

---

## Support

- **Documentation:** `.github/reports/README.md` (user-facing CSV documentation)
- **Configuration:** `.github/dependency-extraction/config.yaml`
- **Issues:** Report bugs via GitHub issues with label `dependencies`

---

**Last Updated:** 2025-10-21
**Maintainer:** @ai-dynamo/python-codeowners

