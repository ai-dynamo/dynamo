---
name: bump-vllm-version
description: Bump vLLM and related dependency versions across the Dynamo codebase
---

# Bump vLLM Version

Update vLLM and its related dependencies (FlashInfer, LMCache, DeepGemm) across the Dynamo codebase. This skill includes pre-bump validation testing and automated PR creation.

## Requirements

- Must be run from the Dynamo repository root
- User must provide target vLLM version
- GPU must be available for running tests (at minimum 1 GPU)
- `gh` CLI must be authenticated for PR creation

## Process

### Step 1: Gather Version Information

Ask the user for the target versions:

1. **vLLM version** (required): e.g., "0.16.0"
2. **FlashInfer version** (optional override): By default, resolve from `vllm[flashinfer]==X.Y.Z`; only ask for an explicit override if the user wants a non-default pairing
3. **LMCache version** (optional): If not provided, keep current
4. **DeepGemm ref** (optional): Usually empty string unless specified
5. **vLLM-Omni version** (usually same as vLLM but may be rc): e.g., "v0.16.0rc1"
6. **Additional dependency overrides** (optional): The user may specify pinned versions for transitive deps like `torch`, `numba`, `ray`, `torchaudio`, `torchvision`, etc. Accept these as-is and install alongside vLLM.
7. **Branch prefix** (optional): e.g., "username/" - will be prepended to branch name

Format clarification:
- vLLM uses `vX.Y.Z` format in some places and `X.Y.Z` in others
- FlashInfer uses `X.Y.Z` for `uv pip install` package versions, but Dynamo refs (`FLASHINF_REF`) use `vX.Y.Z`
- LMCache uses `X.Y.Z` format (no leading 'v')

### Step 2: Read Current Versions

Before making changes, read the current versions from:
1. `container/Dockerfile.vllm` - Look for `VLLM_REF`, `FLASHINF_REF`, `DEEPGEMM_REF`, `LMCACHE_REF` ARGs
2. `container/deps/vllm/install_vllm.sh` - Look for `VLLM_VER`, `FLASHINF_REF`, `LMCACHE_REF`, `VLLM_OMNI_REF`
3. `container/context.yaml` - Look for `vllm_omni_ref`
4. `pyproject.toml` - Look for `vllm[flashinfer,runai]==X.Y.Z` in the vllm extras

Show the user the current versions and confirm the changes to be made.

### Step 3: Pre-Bump Validation Testing

**Critical:** Before modifying any files, validate that the new versions work with the existing test suite.

#### Environment Detection

Check the environment to determine the validation approach:

**If in devcontainer environment** (check `$AGENT_ENVIRONMENT` or if vLLM is already installed):
- Skip Docker setup - devcontainer has GPU access, vLLM, and all dependencies pre-configured
- Proceed directly to Step 3b

**If in linux environment without devcontainer**:
- Use the Docker-based approach in Step 3a

#### 3a: Start a Dynamo vLLM Container (Docker approach - skip if in devcontainer)

Run the latest Dynamo vLLM container with GPU access and the repo mounted:

```bash
docker run --rm -it --gpus all \
  -v $(pwd):/workspace \
  -w /workspace \
  nvcr.io/nvidia/dynamo/dynamo-vllm:latest \
  bash
```

Or if running interactively isn't possible, use a detached container and exec into it.

#### 3b: Install New Versions

Preferred install path (keeps vLLM and FlashInfer aligned automatically):

```bash
uv pip install "vllm[flashinfer]==X.Y.Z"
```

After install, capture the resolved FlashInfer version for Dockerfile/script updates:

```bash
python -c "import flashinfer; print(flashinfer.__version__)"
```

Then install the FlashInfer runtime packages Dynamo expects (match the resolved FlashInfer version):

```bash
FLASHINFER_VER=$(python -c "import flashinfer; print(flashinfer.__version__)")
uv pip install "flashinfer-cubin==${FLASHINFER_VER}"
uv pip install "flashinfer-jit-cache==${FLASHINFER_VER}" --extra-index-url https://flashinfer.ai/whl/cu129
```

Use an explicit `flashinfer` install only when the user requests an override pairing:

```bash
uv pip install flashinfer==X.Y.Z
```

If LMCache is also being updated:
```bash
uv pip install lmcache==X.Y.Z
```

**Common Issue - FlashInfer package mismatch/index gotcha:**
- Symptom: `flashinfer-cubin version (...) does not match flashinfer version (...)`
- Cause: `flashinfer` was upgraded but `flashinfer-cubin` and/or `flashinfer-jit-cache` were not updated to the same version
- Fix: reinstall `flashinfer-cubin` and `flashinfer-jit-cache` to the exact resolved `flashinfer` version
- `flashinfer-jit-cache` is not available on default PyPI index for this setup; always use `--extra-index-url https://flashinfer.ai/whl/cu129`
- For Dockerfile/install script updates, set `FLASHINF_REF` to `v${FLASHINFER_VER}` after resolving `FLASHINFER_VER`

**Common Issue - CuPy Conflicts:**
After installation, you may see warnings about duplicate CuPy packages:
```
UserWarning: CuPy may not function correctly because multiple CuPy packages are installed: cupy-cuda12x, cupy-cuda13x
```

To fix, uninstall the conflicting package (usually the CUDA 13 version if using CUDA 12):
```bash
uv pip uninstall cupy-cuda13x
```

#### 3c: Run Validation Tests

**Run unit tests first** (faster, catches most API issues):

```bash
pytest -m "vllm and unit" --tb=short
```

If unit tests pass, run the e2e tests. **Always pipe output to a log file** so results can be monitored:

```bash
pytest -m "vllm and e2e and gpu_1" --tb=short 2>&1 | tee /tmp/dynamo_tests/pytest_e2e_output.log
```

**Note:** This aligns with the CI's `e2e-single-gpu-tests` job. Always use `tee` for long-running test suites so you can tail the log to check progress.

**Known Infrastructure-Related Test Failures:**
The following test categories may fail due to infrastructure issues (not vLLM API changes) and can be ignored for version bump validation:
- `etcd_ha` tests - Require specific etcd cluster setup
- `kvbm_integration` tests - Require KVBM infrastructure

Focus on failures in these categories as potential API issues:
- `cancellation` tests
- `migration` tests
- `health_check` tests
- `frontend` tests
- `router` tests
- `serve` tests (including multimodal: `multimodal_agg_frontend_decoding`, `multimodal_agg_qwen`)

**Multimodal serve tests** (`tests/serve/test_vllm.py`) exercise the vLLM-Omni integration and are particularly sensitive to VllmConfig changes. Run these individually (they need ~47GB GPU memory each):
```bash
pytest tests/serve/test_vllm.py::test_serve_deployment[multimodal_agg_frontend_decoding] --tb=short
pytest tests/serve/test_vllm.py::test_serve_deployment[multimodal_agg_qwen] --tb=short
```

**Decision point:**
- **Tests pass** (ignoring infrastructure failures): Proceed to Step 4
- **Tests fail** with API errors: Proceed to Step 3d to attempt fixes

#### 3d: Debug and Fix Test Failures

If tests fail due to vLLM API changes, Claude should attempt to fix the issues before reporting to the user.

**Step 3d-i: Clone vLLM for Reference**

Clone the vLLM repository to investigate API changes:

```bash
VLLM_TMP=$(mktemp -d)
git clone git@github.com:vllm-project/vllm.git $VLLM_TMP/vllm
cd $VLLM_TMP/vllm
git checkout vX.Y.Z  # The target version being bumped to
```

**Step 3d-ii: Analyze Failures**

For each test failure:
1. Identify the failing test and the error message
2. Look for `ImportError`, `AttributeError`, or `TypeError` - these indicate API changes
3. Search the vLLM repo for the changed API:
   - Check if functions/classes were renamed
   - Check if function signatures changed
   - Check if modules were reorganized
4. Compare with the previous vLLM version if needed:
   ```bash
   git diff vOLD_VERSION..vNEW_VERSION -- path/to/changed/file.py
   ```

**Step 3d-iii: Fix Dynamo Code**

Based on the analysis:
1. Update Dynamo's vLLM integration code to match the new API
2. Common fixes include:
   - Updating import paths
   - Adjusting function arguments
   - Handling renamed classes/functions
   - Adding compatibility shims if needed

**Step 3d-iii-b: Verify Production Code Impact (REQUIRED)**

When updating contract/field tests (e.g., `test_vllm_renderer_api.py`), you MUST also verify
that the **production code** using those fields still works correctly. Do NOT just update the
test expectations and move on.

For each field change:
1. **Field removed** (e.g., `reasoning_content` removed from `DeltaMessage`):
   - `grep -r "field_name" components/src/dynamo/ --include="*.py"` to find all usages
   - Verify Dynamo never directly accesses the removed field
   - If Dynamo accesses it, update the production code

2. **Field added** (e.g., `reasoning_ended` added to `EngineCoreRequest`):
   - Check if the new field has a default value: `python3 -c "from vllm.X import Y; print(Y.__struct_defaults__)"`
   - Check if Dynamo constructs the struct — if so, verify construction still works without the new field
   - If no default exists, Dynamo code must be updated to provide the field

3. **Field renamed**:
   - Find all usages of old name and update to new name

Always confirm with a quick Python smoke test:
```python
python3 -c "
from vllm.v1.engine import EngineCoreOutput, EngineCoreRequest
# Test Dynamo's construction pattern still works
out = EngineCoreOutput(request_id='test', new_token_ids=[1], finish_reason=None, stop_reason=None)
print('EngineCoreOutput construction OK, new_field_default =', out.new_field_name)
"
```

Report the verification results to the user before proceeding.

**Step 3d-iv: Re-run Tests**

After making fixes:
```bash
pytest -m "vllm and unit" --tb=short
pytest -m "vllm and e2e and gpu_1" --tb=short
```

**Loop:** Repeat 3d-ii through 3d-iv until tests pass or Claude determines the issues require user intervention.

**Step 3d-v: Cleanup**

Remove the temporary vLLM clone:
```bash
rm -rf $VLLM_TMP
```

**Final decision point:**
- **Tests now pass**: Proceed to Step 4 (include the Dynamo fixes in the version bump)
- **Cannot fix automatically**: Stop and report to user with:
  - Summary of failures
  - Analysis of what changed in vLLM
  - Attempted fixes and why they didn't work
  - Recommendations for manual intervention

### Step 4: Update Files

**Note:** If fixes were made in Step 3d, those changes are already in the working directory and should be included in this commit alongside the version bump files.

Update the following files in order:

#### 4a: container/Dockerfile.vllm

Update the ARG declarations (around lines 76-83):

```dockerfile
ARG VLLM_REF="vX.Y.Z"
ARG FLASHINF_REF="vX.Y.Z"
ARG DEEPGEMM_REF=""
ARG LMCACHE_REF="X.Y.Z"
```

#### 4b: container/deps/vllm/install_vllm.sh

Update the version defaults (around lines 14-27):

```bash
VLLM_VER="X.Y.Z"
VLLM_REF="v${VLLM_VER}"
...
FLASHINF_REF="vX.Y.Z"
LMCACHE_REF="X.Y.Z"
```

#### 4c: pyproject.toml

Update the vLLM version in the `vllm` optional dependencies:

```toml
vllm = [
    "uvloop",
    "nixl[cu12]<=0.9.0",
    "vllm[flashinfer,runai]==X.Y.Z",
]
```

#### 4d: vLLM-Omni (lockstep update)

vLLM-Omni **must** be updated in lockstep with vLLM. It may lag behind (e.g., `v0.16.0rc1` when vLLM is at `0.16.0`), and is often **not on PyPI** — requiring source install.

Update the following files:

1. **`container/context.yaml`**: Update `vllm_omni_ref` (e.g., `"v0.16.0rc1"`)
2. **`container/deps/vllm/install_vllm.sh`**: Update `VLLM_OMNI_REF` and ensure the install block tries PyPI first then falls back to building from source:
   ```bash
   if uv pip install vllm-omni==${VLLM_OMNI_REF#v} 2>&1; then
       echo "✓ vLLM-Omni installed from PyPI"
   else
       echo "⚠ PyPI install failed, building from source..."
       git clone --depth 1 --branch ${VLLM_OMNI_REF} https://github.com/vllm-project/vllm-omni.git ...
       uv pip install ...
   fi
   ```
3. **`pyproject.toml`**: If vllm-omni is NOT on PyPI, comment out the dependency with an explanatory note:
   ```toml
   # vllm-omni X.Y.Z is not on PyPI; installed from source in container builds.
   # pip install ai-dynamo[vllm] will not include vllm-omni.
   # "vllm-omni==X.Y.Z",
   ```
4. **`docs/pages/backends/vllm/vllm-omni.md`**: Update the source install command in the Prerequisites section.

**Important**: vllm-omni may overwrite the `vllm` CLI entrypoint. The install script preserves the original by backing up and restoring the entrypoint binary.

#### 4e: docs/reference/support-matrix.md

Update the Build Dependency table to reflect the new vLLM version in the **main (ToT)** column.

### Step 5: Verify Changes

After making changes:
1. Show a summary of all modifications
2. Run `git diff` to display the changes
3. Ask the user to verify the changes look correct

### Step 6: Create Commit and PR

#### 6a: Create a Feature Branch

**Important:** Create a fresh branch from the latest main to avoid mixing unrelated changes:

```bash
git stash  # Stash any unrelated changes
git checkout main
git pull --ff-only
git checkout -b <prefix>bump-vllm-X.Y.Z
```

Where `<prefix>` is the user's branch prefix (e.g., `username/`).

#### 6b: Apply and Commit Changes

Re-apply the version bump edits on the fresh branch, then stage only the relevant files:

```bash
git add container/Dockerfile.vllm container/deps/vllm/install_vllm.sh pyproject.toml docs/reference/support-matrix.md
```

If API fixes were made, also add those files.

#### 6c: Create Signed Commit

Use signed commits (no Claude co-author):

```bash
git commit -s -m "$(cat <<'EOF'
chore(deps): bump vLLM to X.Y.Z

- Update VLLM_REF to vX.Y.Z in Dockerfile.vllm
- Update VLLM_VER to X.Y.Z in install_vllm.sh
- Update vllm dependency to X.Y.Z in pyproject.toml
- Update support-matrix.md for main (ToT)

Pre-bump validation tests passed locally (unit + e2e gpu_1).
EOF
)"
```

#### 6d: Push and Create PR

```bash
git push -u origin <prefix>bump-vllm-X.Y.Z
```

Create PR with:
```bash
gh pr create --title "chore(deps): bump vLLM to X.Y.Z" --body "$(cat <<'EOF'
## Summary
- Bump vLLM from OLD_VERSION to X.Y.Z
- Update version references in Dockerfile.vllm, install_vllm.sh, and pyproject.toml
- Update support-matrix.md for main (ToT)

## Test plan
- [x] Pre-bump validation: `pytest -m "vllm and unit"` - passed
- [x] Pre-bump validation: `pytest -m "vllm and e2e and gpu_1"` - passed
- [ ] CI pipeline validation

## Changes
| File | Change |
|------|--------|
| `container/Dockerfile.vllm` | `VLLM_REF="vOLD"` → `"vNEW"` |
| `container/deps/vllm/install_vllm.sh` | `VLLM_VER="OLD"` → `"NEW"` |
| `pyproject.toml` | `vllm[flashinfer,runai]==OLD` → `==NEW` |
| `docs/reference/support-matrix.md` | main (ToT): `OLD` → `NEW` |
EOF
)"
```

Save the PR URL/number for monitoring.

### Step 7: Monitor PR Pipeline

#### 7a: Check PR Status

Use the GitHub API to monitor the CI pipeline:

```bash
gh pr checks <PR_NUMBER> --watch
```

Or poll periodically:

```bash
gh pr checks <PR_NUMBER>
```

#### 7b: Handle Pipeline Results

- **All checks pass**: Notify user that PR is ready for review
- **Checks fail**:
  1. Fetch the failed check details: `gh pr checks <PR_NUMBER>`
  2. Get workflow run logs if needed: `gh run view <RUN_ID> --log-failed`
  3. Report failures to user with relevant log snippets
  4. Offer to help debug or fix issues

#### 7c: Ongoing Monitoring

If checks are still running, inform the user and offer to:
- Continue monitoring in background
- Check back later with `gh pr checks <PR_NUMBER>`

### Step 8: Completion Summary

Present a completion checklist:

**File Updates:**
- [ ] `container/Dockerfile.vllm` - Updated `VLLM_REF`
- [ ] `container/Dockerfile.vllm` - Updated `FLASHINF_REF` (if changed)
- [ ] `container/Dockerfile.vllm` - Updated `LMCACHE_REF` (if changed)
- [ ] `container/Dockerfile.vllm` - Updated `DEEPGEMM_REF` (if changed)
- [ ] `container/deps/vllm/install_vllm.sh` - Updated `VLLM_VER`
- [ ] `container/deps/vllm/install_vllm.sh` - Updated `FLASHINF_REF` (if changed)
- [ ] `container/deps/vllm/install_vllm.sh` - Updated `LMCACHE_REF` (if changed)
- [ ] `container/deps/vllm/install_vllm.sh` - Updated `VLLM_OMNI_REF`
- [ ] `container/context.yaml` - Updated `vllm_omni_ref`
- [ ] `pyproject.toml` - Updated vLLM version in `vllm` extras
- [ ] `pyproject.toml` - Updated/commented vllm-omni dependency
- [ ] `docs/reference/support-matrix.md` - Updated Build Dependency table
- [ ] `docs/pages/backends/vllm/vllm-omni.md` - Updated source install command

**Validation & PR:**
- [ ] Pre-bump unit tests passed (`pytest -m "vllm and unit"`)
- [ ] Pre-bump e2e tests passed (`pytest -m "vllm and e2e and gpu_1"`)
- [ ] API compatibility fixes applied (if needed)
- [ ] Signed commit created with proper message
- [ ] PR created and pushed
- [ ] CI pipeline status: [PASS/FAIL/PENDING]

**PR Link:** `<PR_URL>`

### Step 9: Cherry-pick to Release Branch (if needed)

If the bump also needs to go to a release branch (e.g., `release/1.0.0`):

1. Wait for the main PR to be squash-merged
2. Fetch main to get the squash-merged commit: `git fetch origin main`
3. Find the single squash-merged commit ID: `git log --oneline origin/main -5`
4. Create a new branch off the release branch:
   ```bash
   git worktree add .worktree/cherry-pick-vllm-release -b <prefix>bump-vllm-X.Y.Z-release origin/release/Y.Z.0
   cd .worktree/cherry-pick-vllm-release
   ```
5. Cherry-pick the **single squash-merged commit** (NOT the individual pre-merge commits):
   ```bash
   git cherry-pick <squash-commit-id>
   ```
6. Push and create PR targeting the release branch:
   ```bash
   git push -u origin <prefix>bump-vllm-X.Y.Z-release
   gh pr create --base release/Y.Z.0 --title "chore(deps): bump vLLM to X.Y.Z (release/Y.Z.0)" ...
   ```

## Guidelines

- **Version format consistency**: Ensure `v` prefix is used correctly per file
- **Dependency compatibility**: FlashInfer and LMCache versions should be compatible with the vLLM version
- **Test before bump**: ALWAYS run validation tests before modifying files
- **Check release notes**: vLLM release notes often specify compatible FlashInfer versions
- **Don't force push**: If CI fails, create fixup commits rather than amending
- **Use signed commits**: Always use `git commit -s -m` for proper attribution
- **Fresh branch from main**: Always create the feature branch from latest main to avoid mixing changes
- **Branch naming**: Use user's preferred prefix (e.g., `username/bump-vllm-X.Y.Z`)
- **Never monkey-patch VllmConfig**: Use `vllm_config.additional_config` dict for custom data. Grep for `vllm_config.<attr> = ` assignments that target non-standard fields.
- **vLLM-Omni lockstep**: Always check if vllm-omni needs a matching version bump. It may not be on PyPI.
- **Git worktree paths in devcontainer**: Worktrees created on the host use host absolute paths (e.g., `/home/user/repo`). In the container (mounted at `/workspace`), fix the `.git` file and `.git/worktrees/*/gitdir` reverse pointer by replacing the host path with `/workspace`.

## Example

User request: "Bump vLLM to 0.14.0"

**Conversation flow (happy path in devcontainer):**
1. Claude asks for FlashInfer, LMCache versions, and branch prefix
2. User provides versions (or says keep current) and prefix (e.g., "aflowers/")
3. Claude reads current versions, shows planned changes
4. Claude installs new vLLM version with `uv pip install vllm==0.14.0`
5. Claude fixes any CuPy conflicts if needed
6. Claude runs `pytest -m "vllm and unit"` - all pass
7. Claude runs `pytest -m "vllm and e2e and gpu_1"` - core tests pass
8. Claude creates fresh branch from main: `aflowers/bump-vllm-0.14.0`
9. Claude applies edits, shows `git diff`, user confirms
10. Claude creates signed commit, pushes branch, creates PR
11. Claude reports PR link: `https://github.com/ai-dynamo/dynamo/pull/XXXX`

**Conversation flow (with API fixes):**
1. Claude asks for FlashInfer, LMCache versions, and branch prefix
2. User provides versions and prefix
3. Claude reads current versions, shows planned changes
4. Claude installs new vLLM version
5. Claude runs `pytest -m "vllm and unit"` - **3 tests fail**
6. Claude clones vLLM repo, checks out target version
7. Claude analyzes failures: `ImportError: cannot import 'OldClass' from 'vllm.foo'`
8. Claude searches vLLM repo, finds `OldClass` renamed to `NewClass` in `vllm.bar`
9. Claude updates Dynamo's import and usage
10. Claude re-runs tests - **all pass**
11. Claude cleans up temp vLLM clone
12. Claude creates fresh branch from main
13. Claude applies version bump edits AND API fixes
14. Claude shows `git diff` with both version bumps and API fixes
15. User confirms, Claude creates signed commit and PR
16. PR description includes note about API fixes

## KVBM Debugging Guide

KVBM (KV Block Manager) is a critical component that integrates with vLLM's scheduler and KV connector APIs. It frequently breaks during vLLM upgrades due to internal API changes.

### KVBM-Specific Test Commands

After installing the new vLLM version, run KVBM tests:

```bash
# KVBM integration tests (requires GPU)
pytest tests/kvbm_integration/test_kvbm.py --tb=short

# KVBM vLLM integration unit tests
pytest lib/bindings/kvbm/tests/test_kvbm_vllm_integration.py --tb=short

# KVBM import tests
pytest tests/dependencies/test_kvbm_imports.py --tb=short
```

### Common KVBM Breaking Points

KVBM integrates with these vLLM internal APIs that frequently change:

1. **`vllm.v1.core.sched.output.SchedulerOutput`** - Scheduler output structure
2. **`vllm.v1.core.sched.output.CachedRequestData`** - Cached request metadata
3. **`vllm.distributed.kv_transfer.kv_connector.v1.base`** - KV connector base classes
4. **`vllm.v1.core.kv_cache_manager`** - KV cache management APIs
5. **`vllm.v1.request.Request`** - Request object structure

### Key KVBM Files to Check

When KVBM tests fail, examine these files:

```
lib/bindings/kvbm/python/kvbm/vllm_integration/
├── connector_leader.py      # Scheduler-side connector (MOST COMMON BREAKAGE)
├── connector_worker.py      # Worker-side connector
├── kv_cache_manager.py      # KV cache management
├── kv_cache_utils.py        # Cache utilities
└── connector/
    ├── dynamo_connector.py  # Main Dynamo connector
    └── pd_connector.py      # Prefill-decode connector
```

### Debugging KVBM Failures

**Step 1: Identify the error type**

Look for these common error patterns:
- `AttributeError: 'X' object has no attribute 'Y'` - Field renamed or removed
- `ImportError: cannot import name 'X'` - Module reorganized
- `TypeError: X() got an unexpected keyword argument` - Signature changed

**Step 2: Compare vLLM versions**

Clone vLLM and diff the relevant files:
```bash
VLLM_TMP=$(mktemp -d)
git clone --depth 1 --branch vX.Y.Z https://github.com/vllm-project/vllm.git $VLLM_TMP/vllm

# Check scheduler output changes
cat $VLLM_TMP/vllm/vllm/v1/core/sched/output.py

# Check KV connector base changes
cat $VLLM_TMP/vllm/vllm/distributed/kv_transfer/kv_connector/v1/base.py
```

**Step 3: Apply fixes and rebuild**

After fixing KVBM code, rebuild the package:
```bash
cd lib/bindings/kvbm && maturin develop --uv
```

Then reinstall the main package:
```bash
cd /workspace && uv pip install -e .
```

### Historical KVBM API Changes

**vLLM 0.14.0:**
- `CachedRequestData.resumed_from_preemption` (list[bool]) → `CachedRequestData.resumed_req_ids` (set[str])
  - Fix: Change iteration to compute `resumed_from_preemption = req_id in resumed_req_ids`
  - File: `lib/bindings/kvbm/python/kvbm/vllm_integration/connector_leader.py`

**vLLM 0.16.0:**
- `VllmConfig` became strict dataclass — serialization via `is_init_field()` → `get_field()` rejects undeclared attributes
  - Dynamo was monkey-patching `vllm_config.consolidator_endpoints = ...` which broke serialization
  - Fix: Use `vllm_config.additional_config["consolidator_endpoints"]` instead (dict field for custom data)
  - Files: `components/src/dynamo/vllm/main.py`, `lib/bindings/kvbm/python/kvbm/vllm_integration/connector_leader.py`
  - **Rule**: NEVER monkey-patch attributes onto `VllmConfig`. Always use `additional_config` dict for custom data.

**Example fix (0.14.0):**
```python
# Before (0.13.0):
for (req_id, resumed_from_preemption, ...) in zip(
    scheduler_output.scheduled_cached_reqs.req_ids,
    scheduler_output.scheduled_cached_reqs.resumed_from_preemption,
    ...
):

# After (0.14.0):
resumed_req_ids = scheduler_output.scheduled_cached_reqs.resumed_req_ids
for (req_id, ...) in zip(
    scheduler_output.scheduled_cached_reqs.req_ids,
    ...
):
    resumed_from_preemption = req_id in resumed_req_ids
```

### KVBM Validation Checklist

Before finalizing the PR, ensure:
- [ ] `pytest tests/kvbm_integration/test_kvbm.py` passes
- [ ] `pytest lib/bindings/kvbm/tests/test_kvbm_vllm_integration.py` passes
- [ ] All vLLM imports in KVBM files are valid (no ImportError)
- [ ] KVBM package rebuilt with `maturin develop --uv`
- [ ] Main package reinstalled with `uv pip install -e .`

### Searching for KVBM vLLM Dependencies

To find all vLLM imports in KVBM:
```bash
grep -r "from vllm" lib/bindings/kvbm/python --include="*.py"
```

To find usages of specific vLLM classes:
```bash
grep -r "SchedulerOutput\|CachedRequestData\|KVConnector" lib/bindings/kvbm/python --include="*.py"
```

## CI Container Dependencies

vLLM upgrades may introduce new Python package dependencies that require additional system libraries in the CI container. Tests may pass locally (in devcontainer) but fail in CI due to missing libraries.

### Symptoms

CI failures with errors like:
```
ImportError: libXXX.so.1: cannot open shared object file: No such file or directory
```

These typically occur during multiprocessing spawn when Python imports a native module.

### Debugging Process

**Step 1: Identify the missing library**

The error message shows the missing `.so` file. Common examples:
- `libxcb.so.1` - X11 library (required by opencv)
- `libGL.so.1` - OpenGL library
- `libglib-2.0.so.0` - GLib library

**Step 2: Find the package that provides it**

```bash
# On Ubuntu/Debian
apt-file search libxcb.so.1
# Result: libxcb1: /usr/lib/x86_64-linux-gnu/libxcb.so.1
```

**Step 3: Check if installed locally**

```bash
find /usr -name "libxcb.so*" 2>/dev/null
```

**Step 4: Add to Dockerfile.vllm**

Edit `container/Dockerfile.vllm` and add the package to the apt install list in the final stage (around line 646):

```dockerfile
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    ...
    # opencv-python-headless requires libxcb
    libxcb1 \
    ...
```

### Historical Container Dependency Issues

**vLLM 0.14.0:**
- Added `opencv-python-headless` dependency
- Requires `libxcb1` even in "headless" mode for multiprocessing spawn
- Fix: Add `libxcb1` to `container/Dockerfile.vllm` apt install

**vLLM 0.16.0 - CUDA 13 wheel problem:**
- `pip install vllm` on PyPI pulls the `cu12` wheel even in CUDA 13 containers
- The cu12 wheel's `_C.abi3.so` links `libcudart.so.12` which doesn't exist in CUDA 13 containers
- Symptom: `ImportError: libcudart.so.12: cannot open shared object file` on ALL vLLM tests in CUDA 13 CI
- Fix: `install_vllm.sh` must detect CUDA version and use `--extra-index-url` for the matching wheel:
  ```bash
  TORCH_BACKEND="cu$(echo $CUDA_VERSION | tr -d '.')"  # e.g., cu129 or cu130
  uv pip install "vllm==${VLLM_VER}" --extra-index-url "https://download.pytorch.org/whl/${TORCH_BACKEND}"
  ```
- **Important**: When debugging CUDA wheel issues, always run `ldd` checks from within the matching CUDA environment. Running `ldd` on a cu130 `.so` from a CUDA 12.9 env gives misleading results.

### Checking for Missing Native Dependencies

After installing the new vLLM version, check for missing shared libraries:

```bash
# Check vLLM's native dependencies
ldd $(python -c "import vllm; print(vllm.__file__.replace('__init__.py', ''))") 2>/dev/null | grep "not found"

# Check opencv dependencies (common culprit)
ldd /opt/dynamo/venv/lib/python3.12/site-packages/cv2/cv2.abi3.so 2>/dev/null | grep "not found"

# Check all site-packages .so files for missing deps
find /opt/dynamo/venv/lib/python3.12/site-packages -name "*.so" -exec ldd {} \; 2>/dev/null | grep "not found" | sort -u
```

### CI vs Devcontainer Differences

| Aspect | Devcontainer | CI Container |
|--------|--------------|--------------|
| Base image | Full development image | Minimal runtime image |
| X11 libraries | Usually installed | Often missing |
| Build tools | Full toolchain | Minimal |
| Purpose | Development | Testing/Production |

When tests pass locally but fail in CI, always check for missing system libraries first.
