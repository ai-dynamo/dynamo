<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# User Workload

Canonical path: `runs/<EXP_ID>/user_workload.yaml`

The user workload is the single YAML file synthesized from the user's initial request and focused follow-up interview.
It is the durable contract that says what must be served, measured, where it may be deployed, and which exact
user-provided DGD starts the run. `recipe-deployer` and every subsequent optimization role receive its exact path and
SHA256 as supporting context.

The DGD body remains in `runs/<EXP_ID>/inputs/user_provided_dgd.yaml`; this file records its immutable path and SHA256.
The workload file is not a place for manifest bodies, deployment results, tuning history, inferred preferences, or
secret values.

## Ownership And Handoff

- `user-interviewer` is the first specialized role for a new optimization run.
- It invokes `synthesize-user-workload`, creates `EXP_ROOT` when one was not supplied, captures the user-provided DGD,
  and writes this file.
- It asks only for blocking facts missing from the initial request or supplied artifacts.
- Downstream work must not begin until both canonical files are valid and both exact paths and SHA256 values are
  available.
- Freeze both files once handoff to `recipe-deployer` occurs. A material workload or user-provided DGD change belongs
  in a new experiment unless the parent explicitly starts a new benchmark series and preserves the prior contract.

## Minimal Schema

Write a standalone YAML file:

```yaml
profile:
  name: kimi-k26-chat              # custom workload identifier
  type: chat                       # chat | agentic | custom
  description: ""                  # required concrete serving-traffic description

model:
  source: moonshotai/Kimi-K2.6     # HF id or local model path requested by the user
  revision: ""                     # optional exact revision when fixed by the user

hardware:                          # one entry per SKU (e.g. H100, H200, B200, GB300)
  - gpu_type: B200
    gpu_count: 8

kubernetes:
  kube_context: ""                 # required kubectl context for the target cluster
  namespace: ""                    # required existing namespace for all run resources
  storage_class: ""                # optional; needed only when the run must create a PVC

deployment:
  dgd_path: runs/<EXP_ID>/inputs/user_provided_dgd.yaml
  dgd_sha256: ""                    # SHA256 of the immutable canonical DGD copy

traffic:
  input_tokens: null               # optional rough/median input sequence length when known
  output_tokens: null              # optional rough/median output sequence length when known
  request_rate: null               # optional requests/second target when known
  concurrency: []                  # optional exact benchmark concurrency values when known

preferences:
  precision: ""                    # e.g. fp4, fp8, bf16
  framework: ""                    # vLLM, SGLang, or TRT-LLM
  mode: ""                         # agg, disagg

objectives:
  ttft_ms_p95_max: null            # optional time to first token
  itl_ms_p95_max: null             # optional inter-token-latency
  request_latency_ms_p95_max: null # optional end-to-end request latency
  output_tput_per_gpu_min: null    # optional output tokens/second/GPU
  error_rate_max: null             # optional request error rate
  custom_slos: ""                  # optional custom user-provided SLOs

artifacts:
  supporting_traces: []            # optional paths for JSONL trace files (e.g. production traces)

notes: ""
created_at: ""
```

## Rules

- Do not store token values, kubeconfig contents, registry credentials, or deployment secret names here.
- Treat manifest-referenced Hugging Face and image-pull secrets as pre-existing cluster prerequisites. Verify their
  existence by name; do not ask the user for secret values or create secrets.
- Do not infer required values from a convenient recipe, current cluster state, model name, or benchmark default.
- Before finalizing, require a valid user-provided DGD path and SHA256, a non-empty profile name, type, concrete
  traffic description, model source, at least one allowed hardware type and count, and either an exact trace or enough
  traffic-shape information for a defensible benchmark.
- Require `deployment.dgd_path` to resolve to `<EXP_ROOT>/inputs/user_provided_dgd.yaml`, and require
  `deployment.dgd_sha256` to match that file.
- `kube_context` and `namespace` are required. The namespace must already exist.
- `storage_class` is optional when the required PVC already exists or a suitable cluster default is known. If the run
  must create a PVC and no suitable class can be determined safely, return a focused question to the user.
- Token lengths are optional customer-provided traffic hints, not required benchmark keys. Prefer real traces or
  profile descriptions when exact input/output token counts are unknown.
- Preserve exact user-provided traffic and SLO values. Do not "round" or reinterpret them.
- Preserve heterogeneous hardware as separate `hardware` entries, such as 4 H100 plus 4 H200. Do not collapse mixed
  GPUs into a single string.
- Treat `profile.type` as the traffic shape, not as a complete API contract. The deployer should use the selected
  DGD's frontend endpoint, defaulting to `/v1/chat/completions` for current text-generation deployments.
- Keep workload intent separate from candidate configuration. Engine and Dynamo knob changes belong in candidate
  configs, deployment ledgers, or benchmark artifacts.
- Leave unknown values as `null` or `""`. Do not invent a GPU count, DGD, model path, or SLO.
- Keep the DGD body separate. Record only its canonical path and SHA256 in this file.
