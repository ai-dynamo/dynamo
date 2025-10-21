# PR #3547: Automated Dependency Version Tracking and Extraction

## 📋 Quick Summary

**Title:** `feat: Add automated dependency version tracking and extraction`  
**Status:** Open, Mergeable, Review Required  
**Linear Issue:** DYN-1235  
**Commits:** 27 (will be squashed to 1 on merge)  
**Created:** 2025-10-10  
**Last Updated:** 2025-10-20

---

## 🎯 Overview

This PR implements comprehensive automated dependency tracking across all Dynamo components (trtllm, vllm, sglang, operator, shared). The system extracts dependencies from 10 source types and runs nightly with automated PRs when versions change.

### Key Capabilities
- 🔍 **10 Source Types**: Dockerfiles, requirements.txt, pyproject.toml, go.mod, Helm charts, docker-compose, rust-toolchain, Cargo.toml, K8s recipes, shell scripts
- 📊 **Smart CSV Output**: 13 columns with critical dependencies first, NVIDIA product detection, package source URLs
- 🔄 **Nightly Automation**: Runs at 2 AM UTC, creates PRs only when changes detected
- 📸 **Release Snapshots**: Auto-triggers on `release/*` branches for permanent versioning
- ⚠️ **Version Discrepancy Detection**: Flags dependencies pinned at different versions across the repo
- 🔔 **Failure Monitoring**: Auto-creates GitHub issues when workflows fail

---

## 📁 Files Changed

### New Files (6)
1. **`.github/workflows/extract_dependency_versions.py`** (1,800+ lines)
   - Main extraction script with 10 parsers
   - Version discrepancy detection
   - Critical dependency and NVIDIA product identification
   - Dual diff tracking (nightly + release)

2. **`.github/workflows/extract_dependency_versions_config.yaml`** (173 lines)
   - Component path configuration
   - Critical dependencies list
   - Known discrepancy documentation
   - Extraction rules and patterns

3. **`.github/workflows/dependency-extraction-nightly.yml`** (236 lines)
   - Nightly workflow (2 AM UTC schedule)
   - Change detection and PR creation
   - Removed dependency tracking
   - Failure monitoring with GitHub issue creation

4. **`.github/workflows/dependency-extraction-release.yml`** (207 lines)
   - Release snapshot workflow
   - Version-based snapshot naming
   - Failure monitoring

5. **`.github/actions/dependency-extraction-setup/action.yml`** (42 lines)
   - Composite action for shared setup steps
   - Eliminates duplication in workflows

6. **`.github/reports/README.md`** (154 lines)
   - Documentation for CSV structure
   - Sorting methodology
   - Critical dependencies explanation
   - Workflow documentation

### Modified Files (1)
- **`.gitignore`** - Added patterns to ignore timestamped CSVs while keeping `*_latest.csv` and `releases/*` files

---

## ✅ .cursorrules Compliance Review

### 1. Commit Message Convention ✅
- **PR Title**: `feat: Add automated dependency version tracking and extraction`
  - ✅ Uses `feat:` prefix (new feature)
  - ✅ Clear, concise description
  - ✅ Follows conventional commits format

- **Individual Commits**: Mixed compliance
  - ✅ Recent commits (monitoring, discrepancy detection) follow conventions
  - ⚠️ Early commits use generic messages ("fix: address CodeRabbit feedback")
  - ℹ️ **Not a blocker**: Repo uses squash-merge strategy (verified from recent PRs #3243, #3536, #3531), so individual commit messages won't appear in main branch

### 2. DCO Sign-off ⚠️
- **Status**: ACTION_REQUIRED on older merge commits
- **Recent commits**: All properly signed with `-s`
- **Issue**: Some early merge commits from October 10 lack DCO
- **Impact**: Minor - recent development commits are signed

### 3. GPG Signing ℹ️
- **Not enforced** for this PR (no indication of missing GPG signatures)
- **DCO sign-off present** on all development commits

### 4. Pre-commit Hooks ❌
- **Status**: FAILING
- **Issue**: Latest commit `086124eda` has trailing whitespace or formatting issues
- **Action Required**: Run `pre-commit run --all-files` and fix

### 5. Code Ownership ✅
- **Python files** (`.github/workflows/extract_dependency_versions.py`): Requires @ai-dynamo/python-codeowners @ai-dynamo/Devops
- **`.github/` directory**: Requires @ai-dynamo/Devops
- **Action**: Ensure these teams are requested for review

### 6. Python Development Standards ✅
- **Package manager**: Uses standard Python (no package manager required for workflow scripts)
- **Formatting**: Black and isort applied (some issues remaining to fix)
- **Linting**: Ruff issues present (unused variables, f-string placeholders) - need cleanup
- **Testing**: N/A for workflow scripts (no pytest needed)

### 7. Documentation ✅
- **`.github/reports/README.md`** provides comprehensive documentation
- **Clear structure**: Explains CSV columns, sorting, workflows
- **User-facing**: Appropriate for team members to understand the system

### 8. Code Quality 📊

**Strengths:**
- ✅ Well-structured with clear separation of concerns
- ✅ Comprehensive parsing for 10 source types
- ✅ Smart formatting and user-friendly output
- ✅ Robust error handling in most areas
- ✅ Version discrepancy detection with filtering for false positives

**Areas for Improvement:**
- ⚠️ Some broad exception catches (`except Exception`) - CodeRabbit flagged this
- ⚠️ Unused variables (`var_name`, `pkg`, `service`, `critical_reason`, `original_line`) - need cleanup
- ⚠️ F-strings without placeholders - clean up extraneous `f` prefixes
- ⚠️ YAMLlint warnings (trailing blank lines)

### 9. CI/CD Integration ✅
- **Workflows properly configured** with schedules, manual triggers, and event-based triggers
- **Artifact uploads** with appropriate retention (90 days nightly, 365 days release)
- **PR automation** with detailed summaries
- **Failure monitoring** with GitHub issue creation

---

## 🔍 Key Features

### 1. Multi-Source Dependency Extraction
Extracts from 10 source types:
- Dockerfiles (base images, ARGs, binary downloads)
- requirements.txt
- pyproject.toml (dependencies + optional)
- go.mod (direct + indirect)
- Shell scripts (pip installs, binary downloads)
- docker-compose.yml
- Helm Chart.yaml
- rust-toolchain.toml
- Cargo.toml
- K8s recipe YAMLs

### 2. Smart CSV Output (13 Columns)
```
Component | Category | Dependency Name | Version | Source File | GitHub URL | 
Package Source URL | Status | Diff from Latest | Diff from Release | Critical | 
NVIDIA Product | Notes
```

**Sorting:**
- Critical dependencies first within each component
- Alphabetical by dependency name

### 3. Version Discrepancy Detection 🆕
- Detects dependencies pinned at different versions across the repo
- Outputs GitHub Actions `::warning` annotations for CI visibility
- Highlights critical dependencies (PyTorch, CUDA, TensorRT-LLM, etc.)
- Filters false positives:
  - Base/runtime Docker images (intentionally different)
  - Go indirect dependencies
  - Pinning style differences (`0.6.0` vs `<=0.6.0`)
  - Sub-dependencies (e.g., `NIXL_UCX_REF`)

**Current Discrepancies Detected:** 4
1. 🔴 **PyTorch** (CRITICAL): 2.8.0 (trtllm) vs 2.7.1+cu128 (vllm) - documented as intentional
2. 🔴 **torchvision** (CRITICAL): 0.22.0a0 (trtllm) vs 0.22.1 (vllm) - matches PyTorch versions
3. ⚪ **fastapi**: `==0.115.12` vs `>=0.115.0`
4. ⚪ **pydantic**: `>=2` vs `>=2.10.6`

### 4. Automated Workflows

#### Nightly Extraction
- **Schedule**: 2 AM UTC daily
- **Trigger**: `workflow_dispatch` for manual runs
- **Output**: Timestamped CSV + `dependency_versions_latest.csv`
- **PR Creation**: Only when changes detected
  - Includes counts (new, changed, removed, unchanged)
  - Lists removed dependencies explicitly
  - Dynamic baseline from previous extraction
- **Failure Monitoring**: Creates/updates GitHub issue on failure

#### Release Snapshots
- **Trigger**: Push to `release/*` branches or manual `workflow_dispatch`
- **Output**: `dependency_versions_v{VERSION}.csv` in `.github/reports/releases/`
- **PR Creation**: Only if snapshot doesn't exist
- **Failure Monitoring**: Creates version-specific GitHub issue on failure

### 5. Composite Action 🆕
Created `.github/actions/dependency-extraction-setup/` to eliminate duplication:
- Checkout repository
- Set up Python 3.12
- Install pyyaml
- Used by both nightly and release workflows

### 6. Failure Monitoring 🆕
**Nightly Workflow:**
- Creates GitHub issue on workflow failure
- Updates existing issue if already open (avoids spam)
- Includes direct link to failed run and troubleshooting steps

**Release Workflow:**
- Creates version-specific issues for each release failure
- Includes version info and actionable troubleshooting

---

## 📊 CSV Output Structure

### Columns (13)
1. **Component**: trtllm, vllm, sglang, operator, shared
2. **Category**: Python Package, Go Module, Base Image, Runtime Image, etc.
3. **Dependency Name**: User-friendly (removes Ver/Version/Ref/Tag suffixes)
4. **Version**: As declared in source
5. **Source File**: Relative path from repo root
6. **GitHub URL**: Direct link to file on GitHub
7. **Package Source URL**: PyPI, NGC Catalog, Docker Hub, Artifact Hub, pkg.go.dev
8. **Status**: tracked, unversioned, indirect
9. **Diff from Latest**: NEW, CHANGED (old → new), UNCHANGED, REMOVED
10. **Diff from Release**: Same as above
11. **Critical**: Yes/No (based on config list)
12. **NVIDIA Product**: Yes/No (auto-detected from keywords/sources)
13. **Notes**: Formatted description (e.g., "From Docker ARG", "Python optional dependency")

---

## 🧪 Testing & Validation

### Manual Testing
- ✅ Extraction script runs successfully
- ✅ Generates valid CSV output
- ✅ Handles all 10 source types
- ✅ Version discrepancy detection works correctly
- ✅ Filters false positives (base images, Go indirect deps, etc.)

### CI Checks
- ✅ **Build and Test - dynamo**: PASSING (31m 32s)
- ✅ **Copyright checks**: PASSING
- ✅ **Link checks**: PASSING
- ✅ **PR title validation**: PASSING
- ❌ **Pre-commit**: FAILING (formatting issues)
- ⚠️ **DCO**: ACTION_REQUIRED (older merge commits)

---

## 🐛 Known Issues & Action Items

### 1. Pre-commit Failures ❌
**Issue:** Latest commit has formatting issues  
**Action:** Run `pre-commit run --all-files` and commit fixes

### 2. Ruff Linting Issues ⚠️
**Issues:**
- Unused variables (`var_name`, `pkg`, `service`, `critical_reason`, `original_line`)
- F-strings without placeholders (remove extraneous `f` prefix)
- Broad exception catches

**Action:** Clean up Python code per Ruff/CodeRabbit suggestions

### 3. DCO Sign-off ⚠️
**Issue:** Some early merge commits lack DCO  
**Action:** Consider rebasing to fix, or leave as-is (recent commits are signed)

### 4. CodeRabbit Suggestions 📝
**Main feedback:**
- Move config out of `.github/workflows/` to avoid actionlint noise
- Narrow exception handling
- Clean up unused variables
- Fix duplicate markdown headings

**Status:** Most addressed; minor cleanup remaining

---

## 📈 Impact & Value

### Benefits
1. 🎯 **Comprehensive visibility** into all dependencies across Dynamo
2. 🔄 **Automated tracking** reduces manual effort and errors
3. 📊 **Historical record** via release snapshots for debugging/audits
4. ⚠️ **Proactive alerts** via version discrepancy detection
5. 🔔 **Failure monitoring** prevents silent breakage
6. 📦 **Package source URLs** for quick access to documentation

### Use Cases
- Security audits and vulnerability tracking
- License compliance verification
- Debugging version conflicts
- Release planning and impact analysis
- Dependency upgrade coordination
- Historical version tracking

---

## 🔄 Merge Strategy

**Recommendation:** Squash and merge (repo standard)
- ✅ Verified from recent PRs (#3243, #3536, #3531)
- ✅ PR title follows conventional commits
- ✅ Description is comprehensive

**Final merge commit will be:**
```
feat: Add automated dependency version tracking and extraction

DYN-1235

[Full PR description will be included in merge commit body]
```

---

## 👥 Reviewers & Approvals

### Required Reviewers (per CODEOWNERS)
- @ai-dynamo/python-codeowners (for `.py` files)
- @ai-dynamo/Devops (for `.github/` directory)

### Current Reviews
- **CodeRabbit**: Provided detailed feedback (mostly addressed)
- **rmccorm4**: Commented
- **nv-tusharma**: Reviewed and provided feedback (addressed)
- **dagil-nvidia**: Author, responded to feedback

### Approval Status
- ❌ Awaiting final approval (REVIEW_REQUIRED)

---

## 🚀 Next Steps

1. ✅ **Fix pre-commit issues** - Run `pre-commit run --all-files`
2. ✅ **Clean up Python linting** - Address Ruff warnings (unused vars, f-strings)
3. ✅ **Request reviews** - Ensure @ai-dynamo/python-codeowners and @ai-dynamo/Devops are requested
4. ❓ **DCO decision** - Keep as-is or rebase to fix early commits?
5. ✅ **Final approval** - Get LGTM from required reviewers
6. ✅ **Merge** - Squash and merge when approved

---

## 📝 Notes

### Intentional Design Decisions
- **CSV over JSON**: Easier to review diffs in PRs
- **Critical dependencies in config**: Explicitly maintained list for clarity
- **Dual diff columns**: Compare against both nightly and release baselines
- **Removed dependencies tracked**: Explicitly list in PR summary
- **GitHub Actions warnings**: Visible in CI and Files Changed tab
- **Composite action**: Reduce duplication across workflows
- **Known discrepancies documented**: Reduces noise for intentional differences

### Future Enhancements (Post-Merge)
- Add more extraction sources as needed
- Enhance NVIDIA product detection
- Add dependency vulnerability scanning
- Integrate with Dependabot or Renovate
- Add historical trending/visualization

---

## 🔗 Links

- **PR**: https://github.com/ai-dynamo/dynamo/pull/3547
- **Linear Issue**: DYN-1235
- **Documentation**: `.github/reports/README.md`
- **Config**: `.github/workflows/extract_dependency_versions_config.yaml`

---

**Last Updated:** 2025-10-20  
**Author:** @dagil-nvidia  
**Status:** Ready for final review and merge (pending pre-commit fixes)

