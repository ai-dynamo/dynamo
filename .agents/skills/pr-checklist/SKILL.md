---
name: pr-checklist
description: Checks a Dynamo pull request for design fit, testing and validation, code quality, and the pull request description. Use when making a pull request to Dynamo or reviewing one.
license: Apache-2.0
metadata:
  author: NVIDIA
  tags:
    - dynamo
    - github
    - code-review
    - pull-request
---

# PR Checklist

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

## Usage

Use this skill when making a pull request to Dynamo or reviewing one. Run the review with multiple agents that investigate each section below, plus an adversarial agent that presses on the weakest areas. Report what needs to change. Keep the PR description up to date as new commits change the PR or address review comments.

## 1. Design Fit

**Core components.** Core components are shared by every backend and deployment mode, so changes to them carry the most risk. Keep those changes minimal, and keep logic for one backend, one model family, or a non-default feature out of shared code paths. Core components include:

- Distributed runtime: `lib/runtime/`
- LLM pipeline: `lib/llm/`
- KV router: `lib/kv-router/`
- Python bindings: `lib/bindings/python/`
- Sidecars: `lib/sidecar/`
- Shared Python components: `components/src/dynamo/common/`, `components/src/dynamo/frontend/`
- Kubernetes operator API: `deploy/operator/api/`

**Replication.** Search the codebase for existing functionality before adding new code, and reuse it instead of duplicating it. Check test code too, such as shared fixtures and helpers. Justify any duplication.

**Existing work.** Check `main`, open pull requests, and open issues for concurrent or existing work on the same problem. Build on it or coordinate with its author instead of opening a duplicate, and record what you find under **Issues and Ongoing Work** in the PR description.

**Complexity.** As a rule of thumb, the complexity of a change can be estimated by the number of new lines of code added to the main codebase:

- Low complexity: < 50 lines. Generally easy to review and unlikely to introduce significant complexity.
- Medium complexity: 50-250 lines. Requires careful review and consideration of potential impact on maintainability.
- High complexity: > 250 lines. May introduce significant complexity and requires thorough review and justification.

Exclude tests, documentation, comments, and whitespace from this count, as they typically do not contribute significantly to the complexity of the main change.

**Correctness and compatibility.** The change must stay correct for every path that shares the changed code: each backend (vLLM, SGLang, TensorRT-LLM), aggregated and disaggregated serving, and Python callers through the bindings. State what the change makes worse, such as performance on another configuration or a feature it no longer supports, and justify any breaking change to a user-facing contract (request fields, CLI flags, environment variables, metrics, CRD fields) in the PR description.

**Area review guidance.** Apply the review guidance for each area the change touches:

- Frontend: [frontend review prompt](../../../.github/review-prompts/frontend.md)
- Runtime: [runtime review prompt](../../../.github/review-prompts/runtime.md)
- Rust: `graham-code-review`
- Python: [Python guidelines](../../../.ai/python-guidelines.md)
- Tests: [pytest guidelines](../../../.ai/pytest-guidelines.md) and [test model size guardrails](../../../.ai/test-model-size-guardrails.md)
- Docs, examples, and recipes: `dynamo-docs`
- Any directory: its nearest `AGENTS.md`

## 2. Performance

Analyze any change to the hot path (per-request or per-token code) statically for locks, copies, and allocations (`dynamo-clone-hotpath-audit` covers Rust clones), and measure it empirically, ideally with mockers (`dynamo-frontend-benchmark`).

## 3. Testing and Validation

Every change must be validated: bug fixes on a reproducer, and features and performance changes end to end.

**Coverage and effectiveness.** Add tests only where the change is prone to errors or regressions, and recommend deleting redundant or trivial tests. A test must exercise the new path, not a fallback, and a bug-fix test must fail without the fix. Avoid excessive mock objects, especially ones coupled to internal logic.

**Reliability and speed.** Tests must not be flaky: avoid bit-exact floating-point assertions, dependence on test order or shared state, and hardcoded ports or paths. Tests must also be fast: no fixed or long sleeps or waits, and a proper timeout on every wait. Dynamo has three test levels. E2E tests generally live under `tests/` and use pytest; other Rust and Python tests are unit or integration tests. As a rule of thumb:

- Unit: milliseconds.
- Integration: single-digit seconds.
- E2E: ideally single-digit seconds, up to tens of seconds.

Run and time every added test, and check it against these budgets.

**CI coverage.** New tests must run in CI. If CI cannot run the changed path, for lack of hardware or a model, say so in the PR and show the validation done outside CI.

## 4. Code Quality and Style

Keep comments few and short, and use them to explain why, not what. Bug history, experiments run, and investigation details belong in the PR description, not in code comments. User-visible changes need docs and example updates in the same PR.

## 5. Pull Request Contents

Fill in every section of `.github/pull_request_template.md`, and title the PR per Dynamo's [commit message rules](../../../docs/fern/pages/community/contributing/code-quality.md#commit-messages), which the `Lint PR` check enforces.

**Overview.** Write 1-3 sentences on why the change exists and what users get from it: the error they no longer hit or the feature they can now use.

**Key Changes.** State the core-logic line count (see Complexity under Design Fit), then up to three changes that most need review, each with a shortened file link and one line on what changed.

**Validation.** Back each claim with evidence, ideally a table with one row per behavior: the case, what happened before, and what happens now. For bug fixes, explain what caused the problem and why the fix addresses it. Include performance data when the change claims or risks a performance impact. Name each key test added and the regression it catches. Do not list routine test runs, clippy or linter checks, or unnecessary detail.

**Issues and Ongoing Work.** Reference the GitHub or Linear issue the PR implements, for example `Closes #8480` or `Closes DYN-1234`. List any similar PRs, issues, or code on `main` that the Existing work check found, each with one line on how this PR differs; if the search found nothing, say so. As the final step, run `dynamo-issue-first` to validate the issue and its reference.
