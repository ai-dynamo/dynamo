<!-- bdcc8fc1-5a9f-4b9b-a057-5e06b697beac 0568a9a9-d98d-469d-ace5-dd2c44b3f5c5 -->
# Robust Dependency Extraction System

## Overview

Transform the dependency extraction script to be resilient against repo structure changes through configuration-based sources, file discovery, validation, and comprehensive maintenance documentation.

## Implementation Steps

### 1. Create Configuration File (`config.yaml`)

Create `scripts_extract_dependencies/config.yaml` with:

- Component definitions (trtllm, vllm, sglang, operator, shared)
- Source file patterns using glob patterns and fallback locations
- Baseline dependency count
- GitHub repository settings

Structure:

```yaml
github:
  repo: "ai-dynamo/dynamo"
  branch: "main"

baseline:
  dependency_count: 251

components:
  trtllm:
    dockerfiles:
      - "container/Dockerfile.trtllm"
      - "containers/Dockerfile.trtllm"  # fallback
    scripts: []
  
  vllm:
    dockerfiles:
      - "container/Dockerfile.vllm"
    scripts:
      - "container/deps/vllm/install_vllm.sh"
  
  sglang:
    dockerfiles:
      - "container/Dockerfile.sglang"
  
  operator:
    dockerfiles:
      - "deploy/cloud/operator/Dockerfile"
    go_modules:
      - "deploy/cloud/operator/go.mod"
  
  shared:
    dockerfiles:
      - "container/Dockerfile"
    requirements:
      - pattern: "container/deps/requirements*.txt"
        exclude: []
    pyproject:
      - "pyproject.toml"
      - "benchmarks/pyproject.toml"
```

### 2. Add Configuration Loader

Modify `extract_dependency_versions.py`:

- Add `load_config()` method to DependencyExtractor class
- Support YAML parsing (add pyyaml to dependencies if not present, or use json as fallback)
- Validate configuration structure
- Merge CLI args with config file settings

### 3. Implement File Discovery

Add new methods to DependencyExtractor:

- `discover_files(patterns: List[str]) -> List[Path]`: Find files matching patterns with fallbacks
- `validate_critical_files() -> Dict[str, bool]`: Check if critical files exist
- `find_file_alternatives(base_pattern: str) -> Optional[Path]`: Try common variations

Update `extract_all()` to:

- Use config-driven file discovery instead of hardcoded paths
- Try multiple location patterns before failing
- Report missing files with suggestions
- Continue processing other components even if one fails

### 4. Enhanced Error Handling

Add comprehensive error tracking:

- Track missing files separately from extraction errors
- Collect warnings for unversioned dependencies
- Generate summary report of extraction success/failures
- Add `--strict` mode that fails on missing files vs. warning mode (default)

Add new summary sections:

```
Extraction Summary:
  Files Processed: 15/18
  Files Missing: 3
    - container/deps/requirements.standard.txt (optional)
    - ...
  Components:
    trtllm: ✓ Complete
    vllm: ⚠ Partial (missing install script)
    ...
```

### 5. Create Maintenance Documentation

Create `scripts_extract_dependencies/MAINTENANCE.md`:

**Sections:**

- How to add new components (step-by-step)
- How to add new file types (requirements, dockerfiles, etc.)
- How to update file paths when repo structure changes
- How to update extraction patterns for new file formats
- Troubleshooting guide for common issues
- Config file reference documentation
- How to update baseline count
- Testing checklist before committing changes

### 6. Add Validation & Testing

Add `--validate` mode:

- Check config file syntax
- Verify all configured paths exist
- Test extraction patterns without writing output
- Report configuration issues

Add `--dry-run` mode:

- Show what files would be processed
- Display discovered files
- Skip actual extraction

### 7. Update README

Update `scripts_extract_dependencies/README.md`:

- Add section on configuration file
- Document file discovery behavior
- Explain how to handle missing files
- Add troubleshooting section
- Link to MAINTENANCE.md
- Add examples for common maintenance tasks

### 8. Add Version Detection Improvements

Enhance extraction methods:

- Better regex patterns for version strings
- Support more version specifier formats (>= , ~=, ^, etc.)
- Extract versions from comments if present
- Add heuristics to guess versions from Git tags/branches when "latest" is used

## Files to Create/Modify

**New Files:**

- `scripts_extract_dependencies/config.yaml` - Configuration
- `scripts_extract_dependencies/MAINTENANCE.md` - Maintenance guide

**Modified Files:**

- `scripts_extract_dependencies/extract_dependency_versions.py` - Add config loading, discovery, validation
- `scripts_extract_dependencies/README.md` - Add config documentation, update examples

## Expected Outcomes

After implementation:

1. Script survives file moves - uses discovery patterns
2. Easy to add new components - edit config.yaml
3. Clear error messages - shows what's missing and where to look
4. Maintainable - documentation guides future updates
5. Validated - catches config errors before extraction
6. Flexible - multiple fallback locations, graceful degradation

### To-dos

- [ ] Create config.yaml with component definitions, file patterns, and settings
- [ ] Add configuration loading and validation to DependencyExtractor class
- [ ] Implement file discovery with glob patterns and fallback locations
- [ ] Add comprehensive error tracking and reporting with strict/warning modes
- [ ] Create MAINTENANCE.md with guides for adding components, updating paths, troubleshooting
- [ ] Add --validate and --dry-run modes for testing configuration
- [ ] Update README.md with configuration documentation and troubleshooting
- [ ] Enhance version extraction with better patterns and heuristics