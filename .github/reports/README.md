# Dependency Reports

This directory contains the latest dependency extraction reports for the Dynamo repository.

## Files

### `dependency_versions_latest.csv`
The most recent dependency extraction results. Updated nightly by the automated CI workflow.

### `unversioned_dependencies_latest.csv`
List of dependencies without explicit version constraints. These should be reviewed and pinned for reproducible builds.

### `releases/dependency_versions_vX.X.X.csv`
Permanent snapshots of dependencies for each release version. Created automatically when release branches are cut.

**Examples:**
- `releases/dependency_versions_v1.2.3.csv` - Release 1.2.3 snapshot
- `releases/dependency_versions_v2.0.0.csv` - Release 2.0.0 snapshot

**CSV Columns:**
- **Component** - Component category (trtllm, vllm, sglang, operator, shared)
- **Category** - Dependency type (Base Image, Framework, Go Module, Python Package, Docker Compose Service, Helm Chart, etc.)
- **Dependency Name** - Human-readable name
- **Version** - Version number or constraint
- **Source File** - Relative path to file defining the dependency
- **GitHub URL** - Clickable link to source line
- **Package Source URL** - Direct link to package documentation:
  - PyPI for Python packages
  - Docker Hub or NGC Catalog for containers
  - Artifact Hub for Helm charts
  - pkg.go.dev for Go modules
  - Official download pages for languages/tools
- **Status** - Legacy status field (New, Changed, Unchanged)
- **Diff from Latest** - Comparison to latest nightly:
  - `New` - New dependency not in latest nightly
  - `Unchanged` - Same version as latest nightly
  - `X → Y` - Version changed from X to Y
  - `N/A` - No latest nightly to compare against
- **Diff from Release** - Comparison to latest release:
  - `New` - New dependency not in latest release
  - `Unchanged` - Same version as latest release
  - `X → Y` - Version changed from X to Y
  - `N/A` - No release snapshot to compare against
- **Critical** - Yes/No flag for critical dependencies
- **NVIDIA Product** - Yes/No flag indicating if dependency is an NVIDIA product
- **Notes** - Additional context

**CSV Sorting:**
The CSV is sorted to make critical dependencies easy to identify:
1. By Component (trtllm → vllm → sglang → operator → shared)
2. By Critical status (Yes before No) within each component
3. Alphabetically by dependency name

**Extraction Sources:**
The script extracts dependencies from multiple sources:
- **Dockerfiles** - Base images and ARG/ENV versions
- **requirements.txt** - Python packages (main, test, docs, standard)
- **pyproject.toml** - Project metadata and dependencies
- **go.mod** - Go module dependencies
- **shell scripts** - Version variables from install scripts
- **docker-compose.yml** - Service container versions
- **Chart.yaml** - Helm chart and dependency versions
- **rust-toolchain.toml** - Rust compiler version
- **Cargo.toml** - Rust Git dependencies
- **K8s recipe YAML** - Git-based pip installs from recipe files

### Critical Dependencies

Critical dependencies are flagged in the CSV to highlight components that require special attention for:
- Security updates
- Version compatibility
- Production stability
- Compliance requirements

The list of critical dependencies is maintained in `../workflows/extract_dependency_versions_config.yaml` under the `critical_dependencies` section. Examples include:
- CUDA (compute platform)
- PyTorch (ML framework)
- Python (runtime)
- Kubernetes (orchestration)
- NATS (message broker)
- etcd (key-value store)

## Timestamped Versions

Timestamped CSV files (e.g., `dependency_versions_20251009_1924.csv`) are:
- **Generated** by the nightly workflow
- **Stored** in GitHub Artifacts (90-day retention)
- **Not committed** to the repo to avoid clutter
- **Available** for download from the workflow run page

## Workflows

### Nightly Tracking (`.github/workflows/dependency-extraction-nightly.yml`)
- **Schedule:** Daily at 2 AM UTC
- **Trigger:** Can be manually triggered via Actions UI
- **Output:** Updates `*_latest.csv` files, creates PR when changes detected
- **Artifacts:** Uploads timestamped CSVs for 90-day retention

### Release Snapshots (`.github/workflows/dependency-extraction-release.yml`)
- **Trigger:** Automatically when `release/*.*.*` branches are pushed
- **Output:** Creates permanent `releases/dependency_versions_vX.X.X.csv`
- **Purpose:** Permanent record of dependencies for each release
- **Artifacts:** Stored for 365 days (1 year)

## Manual Extraction

To run manually from repository root:

```bash
# Basic extraction
python3 .github/workflows/extract_dependency_versions.py

# With options
python3 .github/workflows/extract_dependency_versions.py \
  --output .github/reports/dependency_versions_latest.csv \
  --report-unversioned

# Validate configuration
python3 .github/workflows/extract_dependency_versions.py --validate

# See all options
python3 .github/workflows/extract_dependency_versions.py --help
```

## Files

- 🤖 [Extraction Script](../workflows/extract_dependency_versions.py)
- ⚙️ [Configuration](../workflows/extract_dependency_versions_config.yaml)
- 📋 [Nightly Workflow](../workflows/dependency-extraction-nightly.yml)
- 📸 [Release Workflow](../workflows/dependency-extraction-release.yml)

