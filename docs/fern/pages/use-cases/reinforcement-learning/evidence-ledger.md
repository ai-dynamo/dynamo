---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: RL Evidence Ledger
subtitle: Maintain pins, validation state, ownership, and expiration triggers for RL documentation claims
---

This hidden maintenance page records the load-bearing evidence behind the RL documentation. It is not a replacement for run artifacts. Update it before changing a maturity label, version claim, command, API shape, or benchmark statement.

## Status Vocabulary

| Status | Meaning |
|---|---|
| Released | Present in a tagged/released dependency and validated on the recorded combination. |
| Main | Present on the cited main-branch commit; release compatibility is not implied. |
| Experimental artifact | Runnable public pin with limitations and validation evidence, but no general compatibility promise. |
| Open integration | Public PR/branch or design work that is not an accepted release contract. |
| Design research | Candidate contract or gap analysis without a runnable Dynamo integration. |
| Unsupported | Explicitly absent, unimplemented, or outside the documented contract. |

“Evidence checked” means the source and state were reviewed on the listed date. “Validated” requires an executed run with preserved results. Do not substitute one for the other.

## Baseline Pins

| Component | Pin reviewed | Date | Owner | Expiration trigger |
|---|---|---|---|---|
| Dynamo | [`0718004e5d075c6eed352ae8a53fb48df4554067`](https://github.com/ai-dynamo/dynamo/tree/0718004e5d075c6eed352ae8a53fb48df4554067) | 2026-08-27 | Dynamo RL documentation maintainers | Dynamo minor release or changes under RL tests, frontend request extensions, backend RL handlers, router, tracing, or simulation |
| vLLM package | `vllm==0.27.1` in [Dynamo `pyproject.toml`](https://github.com/ai-dynamo/dynamo/blob/0718004e5d075c6eed352ae8a53fb48df4554067/pyproject.toml) | 2026-08-27 | Dynamo vLLM maintainers | Package pin, runtime image, or backend contract change |
| SGLang package | `sglang[diffusion]==0.5.18` in [Dynamo `pyproject.toml`](https://github.com/ai-dynamo/dynamo/blob/0718004e5d075c6eed352ae8a53fb48df4554067/pyproject.toml) | 2026-08-27 | Dynamo SGLang maintainers | Package pin, runtime image, or backend contract change |
| TensorRT-LLM package | `tensorrt-llm==1.3.0rc24` in [Dynamo `pyproject.toml`](https://github.com/ai-dynamo/dynamo/blob/0718004e5d075c6eed352ae8a53fb48df4554067/pyproject.toml) | 2026-08-27 | Dynamo TensorRT-LLM maintainers | Package pin, runtime image, or backend contract change |
| verl-recipe | [`461b830cfee4f5a67c21edc300c24373230babc7`](https://github.com/verl-project/verl-recipe/tree/461b830cfee4f5a67c21edc300c24373230babc7) | 2026-08-27 | verl recipe and Dynamo RL maintainers | Recipe change under `dynamo/`, `REQUIRED_VERL.txt` change, or Dynamo/verl dependency change |
| verl core required by recipe | [`d82d2777b5dc3e96a8a45168d02660312707ab98`](https://github.com/verl-project/verl/tree/d82d2777b5dc3e96a8a45168d02660312707ab98) from the pinned recipe's `dynamo/REQUIRED_VERL.txt` | 2026-08-27 | verl recipe and Dynamo RL maintainers | `REQUIRED_VERL.txt`, installer behavior, or core verl API/configuration change |
| Prime-RL | PR pins recorded in [framework evidence](#framework-evidence) | 2026-08-27 | Prime-RL integration and Dynamo RL maintainers | PR merge, close, supersession, rebase, release, or schema change |
| SLIME | PR pins recorded in [framework evidence](#framework-evidence) | 2026-08-27 | SLIME integration and Dynamo RL maintainers | PR merge, close, supersession, rebase, release, or schema change |
| NeMo RL | [`6ae035784fe40fd9c9e31d27fffa4a403243a0bd`](https://github.com/NVIDIA-NeMo/RL/tree/6ae035784fe40fd9c9e31d27fffa4a403243a0bd) | 2026-08-27 | NeMo RL and Dynamo RL maintainers after an adapter exists | Public adapter/recipe, generation contract change, or backend lifecycle change |

## Claim Ledger

| ID | Page and claim | Status | Source | Validation | Last checked | Owner | Expiration trigger |
|---|---|---|---|---|---|---|---|
| RL-CONTRACT-001 | [Integration reference](integration-reference.md#contract-at-a-glance): generation and worker administration are separate planes | Main | [RL discovery crate](https://github.com/ai-dynamo/dynamo/blob/0718004e5d075c6eed352ae8a53fb48df4554067/lib/rl/src/lib.rs) and [worker discovery E2E](https://github.com/ai-dynamo/dynamo/blob/0718004e5d075c6eed352ae8a53fb48df4554067/tests/rl/test_worker_discovery.py) | Code and test structure reviewed; GPU E2E not rerun by this docs change | 2026-08-27 | Dynamo RL maintainers | Listener route or system-server architecture change |
| RL-CONTRACT-002 | SGLang native `/generate` accepts token input and emits streaming token/logprob objects under current limits | Main | [TITO E2E](https://github.com/ai-dynamo/dynamo/blob/0718004e5d075c6eed352ae8a53fb48df4554067/tests/rl/test_token_in_token_out.py) | Code-backed E2E exists; not rerun by this docs change | 2026-08-27 | Dynamo SGLang and RL maintainers | Test/request schema, SGLang version, or frontend route change |
| RL-CONTRACT-003 | vLLM OpenAI-compatible TITO returns exact completion IDs and aligned selected log probabilities | Main | [TITO E2E](https://github.com/ai-dynamo/dynamo/blob/0718004e5d075c6eed352ae8a53fb48df4554067/tests/rl/test_token_in_token_out.py) | Code-backed E2E exists; not rerun by this docs change | 2026-08-27 | Dynamo vLLM and RL maintainers | Test, response builder, vLLM version, or request extension change |
| RL-CONTRACT-004 | SGLang metadata upload uses a trusted fsspec destination and fails the request on upload failure | Main | [NVIDIA request extensions](../../developer-guide/additional-resources/nvidia-request-extensions-nvext.md) and [metadata upload implementation](https://github.com/ai-dynamo/dynamo/blob/0718004e5d075c6eed352ae8a53fb48df4554067/components/src/dynamo/common/metadata_upload.py) | Implementation/reference reviewed; remote backend matrix not rerun | 2026-08-27 | Dynamo SGLang and RL maintainers | Upload schema, serializer, fsspec, or credential behavior change |
| RL-CONTRACT-005 | `/v1/rl/workers` is read-only, uses a dedicated listener, and currently discovers RL-enabled vLLM workers | Main | [Worker discovery E2E](https://github.com/ai-dynamo/dynamo/blob/0718004e5d075c6eed352ae8a53fb48df4554067/tests/rl/test_worker_discovery.py) | Code/test reviewed; E2E not rerun | 2026-08-27 | Dynamo RL and vLLM maintainers | Discovery component/endpoint/model or backend registration change |
| RL-CONTRACT-006 | Dynamo generation does not promise sampling idempotency or framework duplicate suppression | Contract boundary | Framework-owned semantics; no Dynamo idempotency key is documented in the cited generation contract | Review statement; framework guides must test retry behavior | 2026-08-27 | Dynamo RL documentation maintainers | A released idempotency/deduplication contract is added |
| RL-CONTRACT-007 | Experimental `/inference/v1/generate` is an opt-in vLLM-compatible frontend route and does not currently implement streaming | Main experimental boundary | [Generate service](https://github.com/ai-dynamo/dynamo/blob/0718004e5d075c6eed352ae8a53fb48df4554067/lib/llm/src/http/service/generate.rs) and [HTTP service registration](https://github.com/ai-dynamo/dynamo/blob/0718004e5d075c6eed352ae8a53fb48df4554067/lib/llm/src/http/service/service_v2.rs) | Code and unit-test surface reviewed; not used as the cross-backend framework path | 2026-08-27 | Dynamo protocol, vLLM, and RL maintainers | Route path, activation, streaming support, or backend coverage change |
| RL-WEIGHT-001 | vLLM shared-disk update requires pause, resets prefix cache, and records caller-supplied version under a per-worker lock | Main | [vLLM handlers](https://github.com/ai-dynamo/dynamo/blob/0718004e5d075c6eed352ae8a53fb48df4554067/components/src/dynamo/vllm/handlers.py) and [discovery/admin E2E](https://github.com/ai-dynamo/dynamo/blob/0718004e5d075c6eed352ae8a53fb48df4554067/tests/rl/test_worker_discovery.py) | Code-backed E2E exists; not rerun by this docs change | 2026-08-27 | Dynamo vLLM and RL maintainers | Handler lifecycle, engine RPC, cache-reset, or route result change |
| RL-WEIGHT-002 | vLLM group initialization defaults to a 30-second watchdog and can terminate a blocked worker | Main | [vLLM handlers](https://github.com/ai-dynamo/dynamo/blob/0718004e5d075c6eed352ae8a53fb48df4554067/components/src/dynamo/vllm/handlers.py) | Unit/release evidence reviewed; integration rendezvous not rerun | 2026-08-27 | Dynamo vLLM maintainers | Timeout default or failure action change |
| RL-WEIGHT-003 | vLLM tensor update route is not implemented | Unsupported | [vLLM handlers](https://github.com/ai-dynamo/dynamo/blob/0718004e5d075c6eed352ae8a53fb48df4554067/components/src/dynamo/vllm/handlers.py) | Direct implementation review | 2026-08-27 | Dynamo vLLM maintainers | Route implementation lands |
| RL-WEIGHT-004 | SGLang fixed weight routes and result schemas are backend/version-specific | Main | [SGLang handler](https://github.com/ai-dynamo/dynamo/blob/0718004e5d075c6eed352ae8a53fb48df4554067/components/src/dynamo/sglang/request_handlers/handler_base.py) and [backend reference](../../developer-guide/knowledge-base/modular-components/backends/sglang/reference-guide.md#engine-routes) | Code/reference reviewed; distributed path not rerun | 2026-08-27 | Dynamo SGLang maintainers | SGLang request classes, route registration, or result schema change |
| RL-WEIGHT-005 | ModelExpress is documented for model loading and fleet distribution, not as the current framework hot-refit path | Released documentation boundary | [ModelExpress guide](../../developer-guide/knowledge-base/kubernetes/model-loading/modelexpress.md) and [upstream deployment guide](https://github.com/ai-dynamo/modelexpress/blob/main/docs/DEPLOYMENT.md) | Documentation/implementation boundary reviewed | 2026-08-27 | ModelExpress and Dynamo RL maintainers | A validated RL hot-refit integration lands |
| RL-ROUTE-001 | KV routing can use prefix overlap, load, predicted placements, queueing, and session affinity | Main | [Router configuration and tuning](../../developer-guide/knowledge-base/modular-components/router/configuration-and-tuning.md) | Current reference reviewed; RL-shaped benchmark not rerun | 2026-08-27 | Dynamo router maintainers | Router cost model, flags, metrics, or defaults change |
| RL-ROUTE-002 | Supplying a session ID does not enable affinity by itself | Main | [Router session affinity reference](../../developer-guide/knowledge-base/modular-components/router/configuration-and-tuning.md#session-affinity) | Current reference reviewed | 2026-08-27 | Dynamo router maintainers | Session-affinity activation semantics change |
| RL-ROUTE-003 | Current typed request context does not contain stable RL policy-version or maximum-lag fields | Main gap | [Request extensions schema](https://github.com/ai-dynamo/dynamo/blob/0718004e5d075c6eed352ae8a53fb48df4554067/lib/llm/src/protocols/common/extensions.rs) | Schema reviewed | 2026-08-27 | Dynamo protocol and RL maintainers | Typed RL context fields land |
| RL-OBS-001 | Request trace v1 provides request IDs, timing/replay shape, optional payloads/headers, and session-aware data but not a full RL event model | Main | [Request trace reference](../../reference/observability/request-traces.mdx) and [tested synthetic join](https://github.com/ai-dynamo/dynamo/blob/main/docs/fern/scripts/rl_trace_join.py) | Schema/reference reviewed and synthetic join executed; live framework join not independently executed | 2026-08-27 | Dynamo observability and RL maintainers | Trace schema/version, application-header capture, join script, or fixture change |
| RL-OBS-002 | The checked request-plane report summarizes terminal, policy-header, token, latency, queue, prefill/decode, KV, finish-reason, and worker fields with explicit coverage while preserving the weight-update and aggregate-metrics boundaries | Main query artifact | [Operations report](https://github.com/ai-dynamo/dynamo/blob/main/docs/fern/scripts/rl_ops_report.py), [request trace schema](https://github.com/ai-dynamo/dynamo/blob/0718004e5d075c6eed352ae8a53fb48df4554067/lib/llm/src/request_trace/types.rs), and [checked synthetic result](https://github.com/ai-dynamo/dynamo/blob/main/docs/fern/scripts/fixtures/rl_trace_join/expected-operations-report.json) | Dependency-free synthetic report and negative tests execute in docs CI; no live framework or Prometheus join is claimed | 2026-08-27 | Dynamo observability and RL maintainers | Request-trace metric field, report schema/statistic/strictness, synthetic fixture, test, or CI workflow change |
| RL-SIM-001 | DynoSim replays serving request shape and does not simulate the trainer, rewards, or policy transitions | Main boundary | [DynoSim overview](../../cli/operations/simulation-with-dynosim/overview.md) and [Agent Trace Replay](../agents/agent-simulation.mdx) | Tool/docs boundary reviewed; RL calibration run not executed | 2026-08-27 | DynoSim and Dynamo RL maintainers | Closed-loop events/tooling or replay schema lands |
| RL-PROCESS-001 | Every non-diagram fenced example and extracted inline contract token is mapped to evidence, a freshness owner, and an expiration trigger; runtime records remain separate from tooling | Main maintenance contract | [Evidence audit](https://github.com/ai-dynamo/dynamo/blob/main/docs/fern/scripts/check_rl_evidence.py), [framework validation checker](https://github.com/ai-dynamo/dynamo/blob/main/docs/fern/scripts/check_rl_validation_record.py), and [program evidence checker](https://github.com/ai-dynamo/dynamo/blob/main/docs/fern/scripts/check_rl_program_record.py) | Offline audit and unit tests execute in docs CI; snippet/token/source checks do not substitute for GPU validation | 2026-08-27 | Dynamo RL documentation maintainers | Fenced example, inline environment/option/route/header/field/port token, evidence mapping, checker schema, test, or CI workflow change |
| RL-PROCESS-002 | Publication requires an independent, artifact-backed clean-room review that links passed framework and program records, accepted owners, the documented user journey, findings disposition, and a broken-link decision | Main publication contract | [Clean-room review checker](https://github.com/ai-dynamo/dynamo/blob/main/docs/fern/scripts/check_rl_clean_room_record.py) and [checked template](https://github.com/ai-dynamo/dynamo/blob/main/docs/fern/scripts/rl_clean_room_record.template.json) | Structure and rejection tests execute in docs CI; no real review record exists and the checker cannot authenticate external artifacts or identities | 2026-08-27 | Dynamo RL documentation maintainers and accepted integration owners | Clean-room schema, owner roles, journey gates, publication criteria, test, or CI workflow change |
| RL-PROCESS-003 | A locally closed publication bundle can prove that all artifact-bearing record fields resolve to immutable regular files, all three records pass their publication gates, clean-room record links match the actual framework/program bytes, and the index matches an externally anchored digest | Main artifact-closure contract | [Artifact bundle index and verifier](https://github.com/ai-dynamo/dynamo/blob/main/docs/fern/scripts/rl_artifact_bundle.py) | Build/verify, index-anchor, mutation, missing/external artifact, traversal, symlink, digest-link, planned-record, and output-containment tests execute in docs CI; semantic artifact truth and reviewer identity remain human checks | 2026-08-27 | Dynamo RL documentation maintainers and clean-room reviewer | Bundle URI/index schema, artifact-bearing record field, publication checker, digest/link, external index anchor, path-safety, test, or CI workflow change |
| RL-PROCESS-004 | Five source-checked product gaps are prioritized as three P0, one P1, and one P2 proposal with issue/DEP vehicles, dependencies, owner teams, acceptance evidence, and a request-plane-now/follow-on-DEP closed-loop decision | Main gap/decision contract | [Product-gap register](https://github.com/ai-dynamo/dynamo/blob/main/docs/fern/scripts/rl_product_gaps.json) and [register checker](https://github.com/ai-dynamo/dynamo/blob/main/docs/fern/scripts/check_rl_product_gaps.py) | Registry and 14 negative/source/docs/dependency/decision tests execute in docs CI; proposals are not filed issues, accepted DEPs, roadmap commitments, or owner assignments | 2026-08-27 | Dynamo RL program DRI plus protocol, observability, routing, weight, planner, and simulation owner teams | Pinned source or docs boundary changes, a gap closes, priority/dependency/vehicle changes, a proposal is filed/accepted, or closed-loop ownership is decided |
| RL-PROCESS-005 | Framework publication requires artifact-backed environment and hardware claims, and a privacy-bounded preflight compares the record with the actual GPU host before execution | Main runtime-preflight contract | [Environment preflight](https://github.com/ai-dynamo/dynamo/blob/main/docs/fern/scripts/rl_environment_preflight.py), [validation template](https://github.com/ai-dynamo/dynamo/blob/main/docs/fern/scripts/rl_validation_record.template.json), and [publication checker](https://github.com/ai-dynamo/dynamo/blob/main/docs/fern/scripts/check_rl_validation_record.py) | Seventeen preflight tests plus expanded validation-record negative tests execute in docs CI; image provenance, fabric meaning, runtime results, and human identity remain separate evidence | 2026-08-27 | Framework integration, Dynamo RL, infrastructure, and clean-room maintainers | Preflight schema/check/CLI/privacy boundary, validation environment/hardware artifact field, GPU/package probe, image provenance, test, or CI change |

## Product Gap and Closed-Loop Decision Register

The checked register converts the former unprioritized limitations list into five issue-ready proposals. It validates the pinned source evidence and documented boundary for every gap, enforces unique IDs, ordered priorities, an acyclic dependency graph, at least four acceptance artifacts, owner teams, and explicit expiration triggers. Run `python3 docs/fern/scripts/check_rl_product_gaps.py` whenever the baseline, trace/context schema, weight lifecycle, replay lowering, or simulation scope changes.

P0 contains typed RL context, served-policy content identity, and standard update lifecycle events because they block correctness, freshness, or release-grade diagnosis. P1 contains the broader lifecycle replay event contract. P2 contains the closed-loop simulator packaging/ownership decision and depends on all four earlier gaps. The current decision is request-plane capture/replay/simulation in scope, closed-loop RL simulation as a follow-on DEP, and no package owner assignment before that DEP is approved.

`issue_ready` means the proposal has a problem statement, current boundary, desired contract, source evidence, dependencies, acceptance evidence, and candidate owner teams. It does not mean an issue was filed, a DEP was accepted, engineering committed to a date, or a named person accepted ownership. When a proposal is filed, replace its status with the real issue/DEP reference only after extending the schema and checker to verify that external state.

## GPU Host Preflight Required for Framework Publication

The validation record now requires artifact URIs for both its environment and hardware assertions. On the allocated GPU host, the checked preflight binds the record to three clean Git heads, Linux, required binaries, the installed backend and PyTorch CUDA versions, visible GPU count/model and driver, an operator-supplied image digest, and a captured GPU-topology table. Strict mode fails on every machine-checkable mismatch and preserves the failed report. It does not read process environment variables or capture the hostname.

The preflight deliberately leaves two conclusions to evidence review. The supplied image digest must be corroborated by scheduler, runtime, or registry metadata, and a reviewer must interpret the topology plus allocation/fabric inventory for the recorded interconnect and network. Those artifacts must be closed into the final bundle through `environment.artifacts` and `hardware.artifacts`. A passing `dynamo.rl.environment-preflight.v1` artifact establishes host/pin agreement only; it cannot satisfy any generation, token/logprob, optimizer, update, recovery, trace, performance, owner-acceptance, or clean-room gate.

## Framework Evidence

| ID | Framework | Artifact | State | Validation record | Maturity | Owner | Expiration trigger |
|---|---|---|---|---|---|---|---|
| RL-FW-VERL-001 | verl | [Dynamo recipe at `461b830c`](https://github.com/verl-project/verl-recipe/tree/461b830cfee4f5a67c21edc300c24373230babc7/dynamo) with recipe content last changed at [`52cdedf7`](https://github.com/verl-project/verl-recipe/commit/52cdedf7e0cfbc3b7d518faefcb2035b12f689f4) | Public main-branch recipe | Upstream recipe includes validation-only smoke, training command, routing comparison, and ThunderAgent evidence; independent Dynamo docs clean-room record absent | Experimental | verl recipe and Dynamo RL maintainers | Recipe/core pin, Dynamo requirement, backend, config, or validation evidence changes |
| RL-FW-SLIME-001 | SLIME | [PR #1](https://github.com/Aphoh/slime/pull/1) | Closed, superseded | Not a current validation artifact | Open integration history | SLIME integration and Dynamo RL maintainers | Replacement accepted or project direction changes |
| RL-FW-SLIME-002 | SLIME | [PR #2 at `4d39b5a`](https://github.com/Aphoh/slime/pull/2) | Closed without merge | PR records prototype streaming and training validation; no maintained merged recipe | Open integration history | SLIME integration and Dynamo RL maintainers | Replacement accepted or artifact removed |
| RL-FW-SLIME-003 | SLIME | [PR #3 at `06d397f`](https://github.com/Aphoh/slime/pull/3) | Open | No released end-to-end record | Integration in progress | SLIME integration and Dynamo RL maintainers | Merge, close, rebase, or supersession |
| RL-FW-PRIME-001 | Prime-RL | [PR #3176 at `828ddc7`](https://github.com/PrimeIntellect-ai/prime-rl/pull/3176) | Open | Discovery path not released | Integration in progress | Prime-RL integration and Dynamo RL maintainers | Merge, close, rebase, or supersession |
| RL-FW-PRIME-002 | Prime-RL | [PR #3180 at `2f67c72`](https://github.com/PrimeIntellect-ai/prime-rl/pull/3180) | Open | Recipe examples not released or clean-room validated | Integration in progress | Prime-RL integration and Dynamo RL maintainers | Merge, close, rebase, or supersession |
| RL-FW-PRIME-003 | Prime-RL | [PR #3181 at `b17ceea`](https://github.com/PrimeIntellect-ai/prime-rl/pull/3181) | Open draft | Combined sidecar path not released | Integration in progress | Prime-RL integration and Dynamo RL maintainers | Merge, close, rebase, or supersession |
| RL-FW-NEMO-001 | NeMo RL | [Generation design at `6ae03578`](https://github.com/NVIDIA-NeMo/RL/blob/6ae035784fe40fd9c9e31d27fffa4a403243a0bd/docs/design-docs/generation.md) | Public framework contract; no validated Dynamo adapter established | Research review only; the latest one-commit advance changed Megatron FP8/skip-load preparation behavior and disabled affected functional tests, while the generation design remained unchanged | Design research | NeMo RL and Dynamo RL maintainers after adapter ownership is established | Public adapter/recipe, generation contract, or backend lifecycle change |

## Validation Records Required for Graduation

Every framework record must attach or link to evidence for:

1. Exact framework, recipe, Dynamo, backend, image/CUDA, model/tokenizer, hardware, and topology pins.
2. Generation-only smoke from a clean environment.
3. Exact prompt/completion token IDs, logprob alignment, masks where applicable, and terminal reasons.
4. One minimal complete training iteration.
5. Policy update, per-worker target verification, cache invalidation, and post-update generation.
6. Request retry/deduplication and one canceled/incomplete sample.
7. One failed worker and one failed update recovery.
8. Framework-to-Dynamo trace correlation.
9. Named framework and Dynamo maintainers and a last-validated date.

Do not set a framework to supported when any item is represented only by design intent or an open PR.

Start from the checked [validation-record template](https://github.com/ai-dynamo/dynamo/blob/main/docs/fern/scripts/rl_validation_record.template.json) and run the [publication-gate checker](https://github.com/ai-dynamo/dynamo/blob/main/docs/fern/scripts/check_rl_validation_record.py). The template's `planned` state and `not_run` gates preserve missing evidence honestly; only a record backed by artifacts for every graduation requirement can pass `--publication-gate`.

## Program Evidence Required for Cross-Cutting Claims

Framework validation proves one adapter/run contract; program evidence proves the cross-cutting routing, weight-transfer, observability, replay, and simulation claims. Start from the checked [program-record template](https://github.com/ai-dynamo/dynamo/blob/main/docs/fern/scripts/rl_program_record.template.json) and use the [program publication-gate checker](https://github.com/ai-dynamo/dynamo/blob/main/docs/fern/scripts/check_rl_program_record.py):

```bash
python3 docs/fern/scripts/check_rl_program_record.py \
  /approved/artifacts/rl-program-record.json

python3 docs/fern/scripts/check_rl_program_record.py \
  /approved/artifacts/rl-program-record.json \
  --publication-gate
```

The structure check accepts a truthful `planned` record. Publication requires immutable pins and named owners; an independent clean-room reviewer; a matched, repeated live routing comparison; colocated and disaggregated weight-update evidence; all three controlled operational diagnoses; measured trace overhead; reconciled capture counts; repeated live replay and DynoSim runs; and numerically consistent calibration/error disclosure. The checker validates the record's contract but cannot verify external artifacts, so the reviewer must inspect every linked command, log, trace, configuration, and result.

## Independent Clean-Room Review Required for Release

Passing framework and program records proves that the required runtime evidence exists; it does not prove that a new user can find and execute the documented journey. Before publishing an integration or raising its maturity, copy the checked [clean-room review template](https://github.com/ai-dynamo/dynamo/blob/main/docs/fern/scripts/rl_clean_room_record.template.json) into the approved artifact store and validate it with the [clean-room publication-gate checker](https://github.com/ai-dynamo/dynamo/blob/main/docs/fern/scripts/check_rl_clean_room_record.py). Run `python3 docs/fern/scripts/check_rl_clean_room_record.py <record>` while drafting and add `--publication-gate` only for the signed release decision.

The publication record must pin the reviewed guide, integration artifact, framework recipe/core, Dynamo commit, image digest, model revision, hardware, and artifact root. It must link digest-pinned framework and program records together with their publication-gate commands, outputs, and passing results. Named owners must explicitly accept the program, framework, Dynamo contract, routing, weight-update, observability, and replay/simulation roles. The reviewer must be independent of every named owner and disclose no conflicts.

The reviewer starts from a fresh workspace, reaches the framework guide in no more than two navigation clicks, executes only commands present in the documentation, and records no undocumented recovery or setup step. Artifact-backed conclusions are required for navigation and pin selection, clean installation and launch, generation and training, weight update and recovery, observability and diagnosis, replay and simulation, and troubleshooting and security. No finding may remain open; blocking and major findings must be resolved. RL broken-link errors must be zero, while unrelated baseline errors must either be resolved or have a named, expiring waiver. A signed approved decision and validation timestamp close the record.

The template intentionally passes only the structure check in its `planned` state. The checker verifies record completeness and internal consistency, not the contents of linked logs, traces, images, commands, or reviewer identity. Release approval therefore remains a human evidence review, and a passing JSON record must not be treated as self-attestation.

## Close and Verify the Publication Artifact Bundle

After the framework, program, and clean-room records are complete, copy the three records and every referenced run artifact under one approved root. Use canonical root-relative `artifact://bundle/` URIs in every `artifact`, `artifacts`, `checker_output_artifact`, and linked-record `uri` field. Finalize the framework and program record bytes first, calculate their digests, and place those exact URIs and digests in the clean-room record. Then build the content-addressed index:

```bash
python3 docs/fern/scripts/rl_artifact_bundle.py build \
  --artifact-root /approved/artifacts/verl-run-001 \
  --record records/framework-validation.json \
  --record records/program-evidence.json \
  --record records/clean-room-review.json \
  --index-json bundle-index.json \
  --strict
```

Strict build mode requires exactly one framework, program, and clean-room schema; runs every record's publication gate; requires at least one artifact reference; rejects missing or externally hosted artifact-bearing fields; resolves every local URI to a regular non-symlinked file below the root; and verifies that the clean-room linked-record URIs and SHA-256 values match the bundled framework and program records. It writes an incomplete index before returning failure so missing evidence remains inspectable. A successful build prints the index SHA-256. Record that digest in the signed review decision or another immutable change-control system outside the bundle; otherwise an attacker could replace both the index and the files it authenticates. The tool does not copy artifacts, follow remote links, or decide whether a log proves the claim attached to it.

Move or copy the complete root as one unit. At the review destination, rehash the records and artifacts and rerun all three publication checkers against the indexed bytes:

```bash
python3 docs/fern/scripts/rl_artifact_bundle.py verify \
  --artifact-root /approved/artifacts/verl-run-001 \
  --index-json bundle-index.json \
  --expected-index-sha256 <digest-from-signed-review-decision>
```

Verification first checks the index against the externally anchored digest, then fails when a record or artifact was changed, removed, replaced by a symlink, moved outside the root, or no longer passes its publication checker. A passing `dynamo.rl.artifact-bundle.v1` index proves local existence, externally anchored byte integrity, record-gate closure, and linked-record digest consistency only. The independent reviewer must still inspect artifact meaning, command provenance, data handling, owner acceptance, and reviewer independence. If policy requires remote durable storage, download an immutable review copy into the bundle; an external URL alone is deliberately not claimed as locally closed.

## Release Audit

The machine-readable manifest at [`docs/fern/scripts/rl_evidence.json`](https://github.com/ai-dynamo/dynamo/blob/main/docs/fern/scripts/rl_evidence.json) maps every ledger ID to its source and documentation assertions. It inventories every non-diagram fenced example by page and ordinal, pins a digest over its language and complete content, maps it to one or more evidence IDs, and assigns a page-level freshness owner and expiration trigger. It separately extracts the complete inline environment-variable, CLI-option, route, header, API/config-field, and documented URL-port surface and requires every token to have evidence, ownership, and an expiration trigger. Adding, removing, reordering, or editing an accounted contract surface without reviewing that evidence fails the network-free docs CI audit:

```bash
python3 docs/fern/scripts/check_rl_evidence.py
```

Before a release or a framework maturity change, run the stricter audit from a full Dynamo checkout:

```bash
python3 docs/fern/scripts/check_rl_evidence.py \
  --release \
  --online \
  --max-age-days 30
```

`--release` compares the watched RL, backend, routing, tracing, and simulation paths with the reviewed Dynamo commit. `--online` compares the recorded framework branch heads, PR heads, open/closed state, draft state, and merge state with the GitHub API. It also reads the pinned verl installer, core-version requirement, trainer configuration, smoke, server, rollout, and metrics files and asserts the exact commands and configuration keys used by the guide. Set `GITHUB_TOKEN` when unauthenticated API limits are insufficient. A failure is an expiration trigger: inspect the change, update the documentation and validation boundary if necessary, then advance the manifest and ledger together. Do not update a pin only to make the audit green.

For every Dynamo minor release:

- [ ] Run the offline, release-drift, online, and review-age audit above; resolve every expiration trigger.
- [ ] Recheck framework configuration keys and canonical dependency-pin files not yet represented by a source assertion.
- [ ] Recheck frontend ports, environment variables, route names, request bodies, and result schemas.
- [ ] Recheck TITO tests and named response fields for vLLM, SGLang, and TensorRT-LLM.
- [ ] Recheck SGLang native `/generate` limits and cancellation behavior.
- [ ] Recheck discovery backend coverage and advertised route behavior.
- [ ] Recheck vLLM and SGLang weight lifecycle, timeouts, cache invalidation, and version semantics.
- [ ] Recheck router modes, flags, defaults, metrics, and session-affinity behavior.
- [ ] Recheck request trace schema, header capture, replay eligibility, and DynoSim/AIPerf entry points.
- [ ] Rerun every framework record whose expiration trigger fired or lower its maturity.
- [ ] Run Fern validation and broken-link checks.
- [ ] Confirm every visible page still links to this one authoritative compatibility source instead of adding a duplicate matrix.

## Benchmark Claim Template

Before adding any percentage or comparative claim, record:

| Field | Required content |
|---|---|
| Claim | Exact metric and direction |
| Workload | Request count, prompt/output distribution, samples per prompt, schedule, sessions, prefix-sharing shape |
| Baseline and variant | Complete configs with only the intended difference where possible |
| Software and model | All pins and image/runtime details |
| Hardware/topology | GPUs, nodes, network, parallel layout, cache tiers |
| Repetitions | Warm-up, run count, variance/spread |
| Serving result | Latency, tokens, queue, cache, utilization, errors |
| RL result | Fresh completed trajectories or accepted samples and full-step timing |
| Mechanism | Causal evidence explaining the change |
| Claim boundary | Live measurement, calibrated prediction, or directional simulation |
| Owner and expiration | Maintainer and revalidation trigger |
