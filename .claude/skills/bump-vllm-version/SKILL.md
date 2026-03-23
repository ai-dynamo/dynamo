---
name: bump-vllm-version
description: Bump vLLM and related dependencies across Dynamo, validate compatibility in the devcontainer, and work through the vLLM test ladder from single-file unit tests to gpu_1 and broader suite runs.
---

# Bump vLLM Version

Use this skill when upgrading Dynamo to a newer vLLM release, especially when the work includes dependency pin changes, API compatibility fixes, and staged validation in the devcontainer.

## Start Here

1. Work from the Dynamo repo root.
2. Read `/workspace/CLAUDE.md`.
3. Read `/workspace/workflow-local.md`.
4. Run:

```bash
python3 deploy/sanity_check.py
```

Treat the devcontainer as the default execution environment for source edits and local validation. Do not start with Docker rebuilds unless the task explicitly requires container pin changes to be validated.

## Sandbox vs Escalation

Safe inside sandbox:
- Read/search repo files
- Edit repo files
- Compare pins and inspect code
- Run fast local commands that do not need network or real GPU visibility

Requires escalation:
- `nvidia-smi`
- `uv pip install ...`
- `python3 -m pip index versions ...`
- `git pull ...`
- Any Docker command
- Any meaningful vLLM or torch test run that needs real GPU visibility

Practical rule:
- If the command needs package download, PyPI metadata, Docker, or actual CUDA/NVML visibility, run it outside the sandbox.
- If a sandboxed test/import fails with GPU or NVML initialization errors, do not treat that as a real repo regression until it is rerun outside the sandbox.

## Gather Version Targets

Collect or infer:
- Target vLLM version
- LMCache version
- vllm-omni policy: keep current, bump if needed, or exclude from scope
- Any explicit overrides for FlashInfer, DeepGEMM, torch, or related packages

Read the current pins from:
- `/workspace/pyproject.toml`
- `/workspace/container/context.yaml`
- `/workspace/container/templates/args.Dockerfile`
- `/workspace/container/deps/vllm/install_vllm.sh`
- `/workspace/container/deps/requirements.test.txt`

For current repo structure, prefer these files over old `container/Dockerfile.vllm` guidance.

Also check test-only dependencies that may lag behind vLLM's transitive minimums.
In particular, compare the installed or declared `vllm` requirement against
`mistral-common` in `/workspace/container/deps/requirements.test.txt`.
If the test requirements pin or constrain `mistral-common` below the version
required by the target vLLM release, bump it as part of the same change.

## Install and Validate in Devcontainer

Default validation flow:

1. Confirm host GPU visibility outside sandbox:

```bash
nvidia-smi
```

2. Install the candidate runtime into the devcontainer environment outside sandbox.

Example for the `0.17.0` bump:

```bash
uv pip install "vllm[flashinfer,runai]==0.17.0" "lmcache==0.3.15"
```

3. Resolve the installed FlashInfer version and align the runtime wheels:

```bash
python3 -c "import flashinfer; print(flashinfer.__version__)"
uv pip install "flashinfer-cubin==<resolved-version>"
uv pip install "flashinfer-jit-cache==<resolved-version>" --extra-index-url <matching-flashinfer-index>
```

Do not assume the index URL. Match it to the CUDA/torch combo actually in use.

4. If the user asked for extra overrides, install them explicitly in the same environment.

## Known Traps

### vllm-omni

- `vllm-omni` is optional, but an old installed version can break test collection before core vLLM compatibility is known.
- For the `0.17.0` session, the devcontainer had `vllm-omni 0.14.0`, which was too old for the upgraded stack.
- PyPI had `vllm-omni 0.16.0`; upgrading to that version allowed collection to move forward.

Check availability outside sandbox:

```bash
python3 -m pip index versions vllm-omni
```

Check the installed version:

```bash
python3 -c "import vllm_omni; print(vllm_omni.__version__)"
```

Rule:
- Do not let omni block the core bump.
- Exclude `components/src/dynamo/vllm/tests/omni/test_omni_handler.py` until the non-omni path is green.
- Re-enable omni only after core vLLM tests are passing or the omni version is upgraded to a compatible release.

### PyTorch / NVML / CUDA visibility

Common failure during collection:
- `UserWarning: Can't initialize NVML`

Meaning:
- The process can import torch, but it does not have usable GPU/NVML visibility.
- In this repo, that often means the run is happening in the sandbox or an otherwise GPU-invisible environment.

Rule:
- Rerun the same test outside sandbox before concluding the bump broke Dynamo.

Related `0.17.0` note:
- CUDA 12.9+ can hit `CUBLAS_STATUS_INVALID_VALUE` due to library mismatch.
- If that appears, check release notes and verify `LD_LIBRARY_PATH`, CUDA libraries, and install method before changing repo code.

## Test Ladder

Always go from smallest to largest. Do not jump straight to the full suite.

Use this prefix for vLLM-heavy runs:

```bash
FLASHINFER_WORKSPACE_BASE=/tmp
```

Use `python3 -m pytest`, not bare `pytest`.

### Phase 1: Single-file component tests

Run these one by one, in this order:

```bash
FLASHINFER_WORKSPACE_BASE=/tmp python3 -m pytest /workspace/components/src/dynamo/vllm/tests/test_vllm_chat_message_utils.py --tb=short
FLASHINFER_WORKSPACE_BASE=/tmp python3 -m pytest /workspace/components/src/dynamo/vllm/tests/test_vllm_renderer_api.py --tb=short
FLASHINFER_WORKSPACE_BASE=/tmp python3 -m pytest /workspace/components/src/dynamo/vllm/tests/test_vllm_kv_events_api.py --tb=short
FLASHINFER_WORKSPACE_BASE=/tmp python3 -m pytest /workspace/components/src/dynamo/vllm/tests/test_vllm_logging.py --tb=short
FLASHINFER_WORKSPACE_BASE=/tmp python3 -m pytest /workspace/components/src/dynamo/vllm/tests/test_vllm_worker_factory.py --tb=short
FLASHINFER_WORKSPACE_BASE=/tmp python3 -m pytest /workspace/components/src/dynamo/vllm/tests/test_vllm_engine_monitor_stats.py --tb=short
FLASHINFER_WORKSPACE_BASE=/tmp python3 -m pytest /workspace/components/src/dynamo/vllm/tests/test_vllm_prompt_embeds.py --tb=short
FLASHINFER_WORKSPACE_BASE=/tmp python3 -m pytest /workspace/components/src/dynamo/vllm/tests/test_vllm_sleep_wake_handlers.py --tb=short
FLASHINFER_WORKSPACE_BASE=/tmp python3 -m pytest /workspace/components/src/dynamo/vllm/tests/test_vllm_unit.py --tb=short
```

Do not run `components/src/dynamo/vllm/tests/omni/test_omni_handler.py` yet.

### Phase 2: Whole vLLM component test directory, omni handler excluded

```bash
FLASHINFER_WORKSPACE_BASE=/tmp python3 -m pytest /workspace/components/src/dynamo/vllm/tests --ignore=/workspace/components/src/dynamo/vllm/tests/omni/test_omni_handler.py --tb=short
```

### Phase 3: Frontend tests one by one

Run these individually so failures stay attributable:

```bash
FLASHINFER_WORKSPACE_BASE=/tmp python3 -m pytest /workspace/tests/frontend/test_vllm.py --tb=short
FLASHINFER_WORKSPACE_BASE=/tmp python3 -m pytest /workspace/tests/frontend/test_prepost.py --tb=short
FLASHINFER_WORKSPACE_BASE=/tmp python3 -m pytest /workspace/tests/frontend/test_prepost_mistral.py --tb=short
```

### Phase 4: Early `gpu_0` / integration contract checks

Run the cheap vLLM integration/assumption checks before the broader unit sweep and long GPU phases.
This is where `tests/kvbm_integration/test_kvbm_vllm_integration.py` belongs.

```bash
FLASHINFER_WORKSPACE_BASE=/tmp python3 -m pytest /workspace/tests/kvbm_integration/test_kvbm_vllm_integration.py -m "vllm and gpu_0 and integration" --tb=short 2>&1 | tee /tmp/vllm-kvbm-vllm-integration.log
```

### Phase 5: Broader unit sweep across components and tests

Always ignore `target/` to avoid collection noise from vendored or generated content.

```bash
FLASHINFER_WORKSPACE_BASE=/tmp python3 -m pytest /workspace/components /workspace/tests -m "vllm and unit" --ignore=/workspace/target --ignore=/workspace/components/src/dynamo/vllm/tests/omni/test_omni_handler.py --tb=short 2>&1 | tee /tmp/vllm-unit-components-tests.log
```

### Phase 6: Narrow `gpu_1` / `pre_merge` serve path

After the unit sweep is stable, move to the smallest single-GPU serve path first.
This repo has an annoying split between `vllm and gpu_1 and pre_merge` and
`vllm and gpu_1 and e2e`, so do not jump straight to a broad marker run.

Start with the focused serve file and keep it narrow:

```bash
FLASHINFER_WORKSPACE_BASE=/tmp python3 -m pytest /workspace/tests/serve/test_vllm.py -m "vllm and gpu_1 and pre_merge" --tb=short 2>&1 | tee /tmp/vllm-serve-gpu1-premerge.log
```

### Phase 7: Targeted `gpu_1` / `e2e` files, narrow to wider

After the serve path is stable, run the known heavier vLLM e2e/integration files one by one, in this order.
On a machine with only 1 visible GPU, always keep at least `-m "vllm and gpu_1"` on these targeted runs so pytest does not wander into `gpu_2` cases just because the file contains both.

```bash
FLASHINFER_WORKSPACE_BASE=/tmp python3 -m pytest /workspace/tests/router/test_router_e2e_with_vllm.py -m "vllm and gpu_1" --tb=short 2>&1 | tee /tmp/vllm-router-e2e.log
FLASHINFER_WORKSPACE_BASE=/tmp python3 -m pytest /workspace/tests/kvbm_integration/test_kvbm.py -m "vllm and gpu_1" --tb=short 2>&1 | tee /tmp/vllm-kvbm.log
FLASHINFER_WORKSPACE_BASE=/tmp python3 -m pytest /workspace/tests/kvbm_integration/test_consolidator_router_e2e.py -m "vllm and gpu_1" --tb=short 2>&1 | tee /tmp/vllm-consolidator-router-e2e.log
```

Only after those file-level passes are green, expand to a broader marker run.
Always ignore `target/` and skip `tests/fault_tolerance/` for this phase because those runs can take a while and are not part of the default bump ladder.

```bash
FLASHINFER_WORKSPACE_BASE=/tmp python3 -m pytest /workspace/components /workspace/tests -m "vllm and gpu_1" --ignore=/workspace/target --ignore=/workspace/tests/fault_tolerance --ignore=/workspace/components/src/dynamo/vllm/tests/omni/test_omni_handler.py --tb=short 2>&1 | tee /tmp/vllm-gpu1-components-tests.log
```

### Phase 8: Broader `e2e` / full vLLM suite

Only after the earlier phases are green:

```bash
FLASHINFER_WORKSPACE_BASE=/tmp python3 -m pytest -m "vllm and e2e and gpu_1" --ignore=/workspace/target --ignore=/workspace/tests/fault_tolerance --tb=short 2>&1 | tee /tmp/vllm-e2e-gpu1.log
```

If the user wants the broadest pass after that:

```bash
FLASHINFER_WORKSPACE_BASE=/tmp python3 -m pytest /workspace/components /workspace/tests -m "vllm" --ignore=/workspace/target --ignore=/workspace/tests/fault_tolerance --tb=short 2>&1 | tee /tmp/vllm-full-suite.log
```

### Phase 9: Explicit Omni handler pass

Only after the earlier phases are stable, run the Omni-specific handler test that was excluded above:

```bash
FLASHINFER_WORKSPACE_BASE=/tmp python3 -m pytest /workspace/components/src/dynamo/vllm/tests/omni/test_omni_handler.py --tb=short
```

## Code-Level `0.17.0` Watchpoints

When `0.17.0` compatibility issues appear, check these first:

### Removed or renamed Python fields

- `EngineCoreRequest.eos_token_id` was removed in `vLLM 0.17.0`
- Update production code first, then tests
- Do not "fix" only the contract tests if runtime code still depends on the old field

### KV event shape changes

- `extra_keys` was added on the vLLM side
- Dynamo ingest path already handles this in `/workspace/lib/kv-router/src/zmq_wire.rs`
- The consolidator subscriber can ingest it
- The consolidator publisher still republishes an older `BlockStored` shape and drops it

Important scope rule:
- Unless the task explicitly includes end-to-end multimodal metadata preservation through the consolidator, do not widen the consolidator data model just because `extra_keys` exists upstream
- For the `0.17.0` session, the narrower conclusion was: ingest-side handling was present, but consolidator republish still dropped the field

Relevant paths to inspect when KV event compatibility is in scope:
- `/workspace/lib/kv-router/src/zmq_wire.rs`
- `/workspace/lib/llm/src/kv_router/publisher.rs`
- `/workspace/lib/llm/src/block_manager/kv_consolidator`
- `/workspace/lib/llm/src/block_manager/kv_consolidator/publisher.rs`
- `/workspace/lib/llm/src/block_manager/kv_consolidator/subscriber.rs`

## Updating Repo Pins

After validation and compatibility fixes, update the repo pins in:
- `/workspace/pyproject.toml`
- `/workspace/container/context.yaml`
- `/workspace/container/templates/args.Dockerfile`
- `/workspace/container/deps/vllm/install_vllm.sh`

Update docs only where the version is intentionally surfaced or the install instructions changed materially.

Version format rules:
- vLLM uses both `X.Y.Z` and `vX.Y.Z` forms depending on file
- `vllm[flashinfer,runai]==...` in Python deps uses `X.Y.Z`
- `VLLM_REF` and similar git-style refs use `vX.Y.Z`
- LMCache uses `X.Y.Z`
- `FLASHINF_REF` typically uses `vX.Y.Z`

Common file-level update checklist:
- `/workspace/pyproject.toml`: update the Python vLLM dependency
- `/workspace/container/context.yaml`: update `vllm_ref`, `vllm_omni_ref`, and any related defaults
- `/workspace/container/templates/args.Dockerfile`: update version ARG defaults or references
- `/workspace/container/deps/vllm/install_vllm.sh`: update default install versions and fallback logic
- vLLM-facing docs: update only if they intentionally mention the bumped version or changed install commands

## Verify the Changes

After editing:

1. Show a concise summary of what changed.
2. Inspect the diff:

```bash
git diff -- /workspace/pyproject.toml /workspace/container/context.yaml /workspace/container/templates/args.Dockerfile /workspace/container/deps/vllm/install_vllm.sh
```

3. Re-run the relevant part of the test ladder, not just the broadest suite.
4. If the change touched contracts or wire formats, verify both:
- production code paths
- tests that encode those contracts

Rule:
- Do not stop at “tests compile now”.
- Confirm the repo pins, runtime imports, and the staged validation path all agree.

## Commit and PR Workflow

Only do this if the user asked for commit/PR handling.

### Branching

Prefer a fresh branch from the latest `main` so unrelated local changes do not leak in.

Typical flow:

```bash
git stash
git checkout main
git pull --ff-only
git checkout -b <prefix>bump-vllm-X.Y.Z
```

If the user wants to stay on the current branch, follow that instruction instead.

### Staging

Stage only the relevant bump and compatibility files.

At minimum, consider:

```bash
git add /workspace/pyproject.toml /workspace/container/context.yaml /workspace/container/templates/args.Dockerfile /workspace/container/deps/vllm/install_vllm.sh
```

If compatibility fixes were required, also stage those Python/Rust/test files.

### Commit

Use signed commits:

```bash
git commit -s -m "chore(deps): bump vLLM to X.Y.Z"
```

Prefer a short message unless the user explicitly wants a longer body.

### PR

When creating the PR, include:
- bumped versions
- core compatibility fixes
- which stages of the test ladder passed
- whether omni was excluded, bumped, or validated

Suggested test-plan structure:
- single-file component tests
- full component test directory with omni excluded
- frontend tests one by one
- broader `vllm and unit`
- `vllm and gpu_1`
- `vllm and e2e and gpu_1` if run

## Monitor the PR

If a PR was opened, check status with:

```bash
gh pr checks <PR_NUMBER>
```

Or watch continuously:

```bash
gh pr checks <PR_NUMBER> --watch
```

If checks fail:
- identify whether the failure is infra, environment, or real compatibility
- fetch the failed logs
- summarize the failing stage and the likely root cause before making more edits

## Release Cherry-Pick Flow

If the bump also needs a release branch:

1. Wait for the main PR to merge.
2. Fetch the merged commit from `origin/main`.
3. Create a release branch worktree from the correct release base.
4. Cherry-pick the single merged commit.
5. Open a release-targeted PR.

Rule:
- Cherry-pick the squash-merged commit, not a pile of pre-merge commits.

## Working Rules

- Prefer the devcontainer path first
- Prefer incremental test execution over marker-wide sweeps
- Pipe long runs to `tee`
- Keep omni isolated until core is stable
- Treat sandbox NVML/CUDA failures as environment issues first
- Do not assume old container-file paths still apply; follow the current templated container layout
