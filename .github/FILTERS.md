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

## Standalone runtime admission

The PR workflow may omit standalone SGLang CPU/GPU jobs for a nonempty set of
ordinary changes contained entirely in one audited class in
`actions/changed-files/report.py`. The exact operator class allows additions and
modifications; the vLLM unit-test class allows modifications only. Classes cannot
be mixed. All eight change statuses must be present, valid and account for the
complete changed-file set without overlapping additions/modifications. Copies, deletions, renames,
type changes, unmerged/unknown files, unlisted siblings and malformed or missing
data retain the existing full selection. Set repository variable
`FORCE_FULL_CI=true` to bypass admission. Main, postmerge and nightly are unchanged.

The initial vLLM processor unit-test class retains vLLM CPU tests and mypy.
The operator class below retains builds/compliance and every existing operator,
Helm, deployment, DGDR and Snapshot gate. Only `sglang-test` and
`sglang-multi-gpu-test` are omitted: this includes standalone `gpu_0` tests on
amd64/arm64 and `gpu_1`/`gpu_2` tests on amd64. Shared CPU selection is unchanged;
execution still follows the original path filters and `RUN_DEPLOY_TESTS`.

The exact operator consumer audit is below. These inputs do not configure the
standalone backend launch scripts or runtime Python/Rust packages. Generated
schemas are consumed by Kustomize, not by standalone model workers. Future
consumer changes must revisit admission before broadening these dependencies.
Paths are relative to the repository root.

| Exact path | Consumer retained when its existing gate selects it |
| --- | --- |
| `deploy/helm/charts/platform/README.md` | Operator/Helm selection; no standalone runtime selection |
| `deploy/helm/charts/platform/components/operator/templates/deployment.yaml` | Helm tests, operator deployment and Snapshot operator setup |
| `deploy/helm/charts/platform/components/operator/values.yaml` | Helm tests, operator deployment and Snapshot operator setup |
| `deploy/helm/charts/platform/tests/namespace_restriction_deployment_test.yaml` | Helm chart tests |
| `deploy/helm/charts/platform/values.yaml` | Helm chart tests and deployment setup |
| `deploy/operator/api/v1beta2/dynamographdeploymentcandidate_types.go` | Operator Go build/tests, CRD generation and DGDR deployment |
| `deploy/operator/api/v1beta2/dynamographdeploymentrequest_types.go` | Operator Go build/tests, CRD generation and DGDR deployment |
| `deploy/operator/api/v1beta2/dynamographdeploymentrun_types.go` | Operator Go build/tests, CRD generation and DGDR deployment |
| `deploy/operator/api/v1beta2/groupversion_info.go` | Operator API registration/build/tests |
| `deploy/operator/api/v1beta2/types_test.go` | Operator Go tests |
| `deploy/operator/api/v1beta2/zz_generated.deepcopy.go` | Operator Go build/tests |
| `deploy/operator/cmd/crd-apply/main.go` | Operator image/build/tests and CRD deployment setup |
| `deploy/operator/cmd/crd-apply/main_test.go` | Operator Go tests |
| `deploy/operator/config/crd/bases/nvidia.com_dynamographdeploymentcandidates.yaml` | Operator CRD installation, DGDR and schema generation |
| `deploy/operator/config/crd/bases/nvidia.com_dynamographdeploymentrequests.yaml` | Operator CRD installation, DGDR and schema generation |
| `deploy/operator/config/crd/bases/nvidia.com_dynamographdeploymentruns.yaml` | Operator CRD installation, DGDR and schema generation |
| `deploy/operator/docs/fix-api-anchors.py` | Operator API documentation generation |
| `docs/fern/pages/kubernetes/installation/install-dynamo.md` | Fern documentation checks |
| `docs/fern/pages/reference/kubernetes-api/additional-resources/api-reference-k8s.md` | Fern/API documentation checks |
| `docs/fern/pages/reference/kubernetes-api/full-api-reference.mdx` | Generated API documentation checks |
| `docs/fern/scripts/tests/test_gen_kubernetes_api.py` | Existing API generator regression checks |
| `recipes/kustomize/components/dynamo-openapi/dynamo-openapi.json` | Recipe/schema generation validation and Kustomize consumers |
| `recipes/templates/kustomize/components/dynamo-openapi/dynamo-openapi.json` | Recipe/schema generation validation and Kustomize consumers |

Historical PRs #15930 and #13603 each contain 15 modified and eight added paths
in this exact list. Both actual status-bearing diffs qualify for operator
admission. Their recorded runtime cost is an opportunity estimate; selection
replay alone does not demonstrate measured admission savings.

Admission saves runtime work only when the original gates selected that work.
For this operator list, only these four paths select standalone backend
runtime jobs through the `deploy` filter:

- `deploy/helm/charts/platform/components/operator/templates/deployment.yaml`
- `deploy/helm/charts/platform/components/operator/values.yaml`
- `deploy/helm/charts/platform/tests/namespace_restriction_deployment_test.yaml`
- `deploy/helm/charts/platform/values.yaml`

README-, Go-, CRD-, documentation- and OpenAPI-only subsets already omit those
runtime jobs and therefore save zero additional runtime work. A mixed set of
allowed additions/modifications can save work when it includes a listed Helm path.
