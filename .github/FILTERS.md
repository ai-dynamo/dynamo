# CI Filters

The `filters.yaml` file controls which CI jobs run based on changed files.

## How It Works

When you open a PR, CI checks which files changed and runs only relevant jobs:

| Filter                                                  | Triggers                                                                                                                                                                             |
| ------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `core`                                                  | Main test suite (vLLM, SGLang, TRT-LLM containers)                                                                                                                                   |
| `dev_images`                                            | dev / local-dev image builds only (no runtime or GPU jobs)                                                                                                                           |
| `operator`                                              | Kubernetes operator tests                                                                                                                                                            |
| `snapshot`                                              | All-framework standalone Snapshot deploy tests (github.com/ai-dynamo/snapshot is external; this covers Dynamo's integration surface)                                               |
| `snapshot_vllm` / `snapshot_sglang` / `snapshot_trtllm` | That framework's checkpoint deploy suite                                                                                                                                             |
| `deploy`                                                | Deploy-specific tests                                                                                                                                                                |
| `vllm` / `sglang` / `trtllm` / `triton`                 | Backend-specific tests                                                                                                                                                               |
| `sidecar`                                               | Unified multi-architecture sidecar image build, publish, and compliance checks for changes under `lib/sidecar/**` (docs excluded), its shared workflow, and shared compliance inputs |
| `benchmarks`                                            | Dynamo runtime pipeline (runs `tests/benchmarks/**` pytest suite)                                                                                                                    |
| `sample`                                                | Sample-backend unified test (piggybacks on vllm image)                                                                                                                               |
| `efa`                                                   | EFA runtime image builds for vLLM, SGLang, TRT-LLM (`container/templates/aws.Dockerfile` change)                                                                                     |
| `docs`                                                  | Docs Lint, Fern Configuration, Docs Website Composition, and Fern Broken Links checks; Fern preview or publish workflow                                                              |
| `fern_components`                                       | Parse custom MDX components (a step inside Fern Configuration Check)                                                                                                                 |
| `examples`                                              | Recipe Kustomize generation and docs-artifact unit checks                                                                                                                            |
| `planner_gym` | Planner Gym CPU tests (native adapters excluded), package builds, and changed-file reporting regressions (Python 3.11 and 3.12) |
| `ignore`                                                | Nothing (classification only)                                                                                                                                                        |
| `rust`                                                  | Rust pre merge checks                                                                                                                                                                |

> [!NOTE]
> `ignore` doesn't directly trigger CI jobs.
> It exists to satisfy coverage requirements - every file must match at least one filter.
> Sidecar source and proto files also match `rust`, so the existing workspace Rust checks cover sidecar tests before the image is built and published.
> `docs` gates the Docs Lint, Fern Configuration Check, Docs Website Composition Check, and Fern Broken Links Check jobs in `pre-merge.yml`.
> `examples` gates Recipe Check.

> [!TODO]
> The sidecar image also consumes root Cargo files, shared libraries, and composite actions.
> Expanding the filter to cover every remaining build input is deferred until the additional PR CI fan-out is evaluated and agreed.

## Fixing "Uncovered Files" Errors

If CI fails with:
```
ERROR: The following files are not covered by any CI filter
```

Add patterns to `filters.yaml`:

1. **New source files** → Add to `core` or relevant backend filter
2. **New examples, recipes, and recipe validation helpers** → Add to `examples`
3. **Fern docs-site content** (anything under `docs/fern/`) → Add to `docs`
4. **Markdown elsewhere in the repo** (a `lib/` or `container/` README) → Add to `ignore`.
   It is documentation, but the Fern site does not read it, and `docs` gates four jobs
   including the composition check.
5. **Config files that don't need CI** → Add to `ignore`

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

Add the group to `filters.yaml`. The changed-files reporter automatically includes
its JSON file list in coverage checks, except for the `all` catch-all group.
No parallel list of filter names needs updating.

If a job uses the filter to decide whether to run, expose its `*_any_modified`
value as an output in `.github/actions/changed-files/action.yml`, then connect
that output to the job's condition.

The reporter consumes JSON files written by the pinned changed-files action.
Keep `json`, `escape_json`, and `write_output_files` enabled and `safe_output`
disabled: v42 removes one quote-escape layer when writing each file, while its
shell sanitization would alter filenames. Filenames are never interpolated into
shell source. The reporter preserves spaces, quotes, and newlines and prints
JSON-escaped names so they cannot introduce workflow commands into the log.
