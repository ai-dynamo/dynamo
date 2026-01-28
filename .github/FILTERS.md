# CI Filters

The `filters.yaml` file controls which CI jobs run based on changed files.

## How It Works

When you open a PR, CI checks which files changed and runs only relevant jobs:

| Filter | Triggers |
|--------|----------|
| `core` | Main test suite (vLLM, SGLang, TRT-LLM containers) |
| `operator` | Kubernetes operator tests |
| `deploy` | Helm chart validation |
| `vllm` / `sglang` / `trtllm` | Backend-specific tests |
| `docs` | Nothing (classification only) |
| `examples` | Nothing (classification only) |
| `ignore` | Nothing (classification only) |
| `rust` | Rust pre merge checks |

> **Note:** `docs`, `examples`, and `ignore` don't trigger any CI jobs. They exist to satisfy coverage requirements - every file must match at least one filter.

## Fixing "Uncovered Files" Errors

If CI fails with:
```
ERROR: The following files are not covered by any CI filter
```

Add patterns to `filters.yaml`:

1. **New source files** → Add to `core` or relevant backend filter
2. **New examples/recipes** → Add to `examples`
3. **Documentation** → Add to `docs`
4. **Config files that don't need CI** → Add to `ignore`

## Testing Locally

```bash
cd .github/scripts
npm install
npm run coverage  # Check if all repo files are covered
```

## Pattern Syntax

- `**` matches any path depth (but not dotfiles by default)
- `*` matches within a directory
- `!pattern` excludes files (used in `core` to skip docs)
- For dotfiles, add explicit pattern like `dir/.*`

Example: `lib/**/*.rs` matches all Rust files under `lib/`.

## Adding a New Filter Group

If you create a new filter in `filters.yaml`, you must also update the workflows:

1. Add the filter to `filters.yaml`
2. Update **both** workflow files to include the new filter in the uncovered files check:
   - `.github/workflows/container-validation-backends.yml`
   - `.github/workflows/container-validation-dynamo.yml`

In each workflow, find the `COVERED_FILES` line and add your new filter:

```bash
COVERED_FILES=$(echo "... ${{ steps.filter.outputs.YOURFILTER_all_modified_files }} ..." | ...)
```

If you skip this step, CI will fail with "uncovered files" even though your filter exists.

