<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# DGD comparison test design

## Purpose

The comparison suite generates DynamoGraphDeployments for the same deployment intent through:

- the existing DGDR v1beta1 profiler;
- AI Simulate Sweeper followed by the AI Configurator renderer; and
- AI Simulate Sweeper followed by the standalone direct renderer.

An existing recipe may be deployed as a fourth, independently maintained variant. The suite learns
whether each result is runnable; it does not require the generators to choose identical topologies.
The issue #8469 suite retains all 29 recipe-matrix rows. Eight currently have both native profiler
inputs and can run the comparison; the other 21 remain visible as coverage gaps.

## Attribution

This implementation builds on Ashna Mehrotra's work in
[PR #14031](https://github.com/ai-dynamo/dynamo/pull/14031). In particular, it reuses:

- the eight runnable issue #8469 DGDR and Sweeper input pairs;
- the idea of exercising the existing profiler, Sweeper-AIC, and Sweeper-direct paths together;
- cluster GPU-inventory validation and isolated preparation of multi-document manifests; and
- model discovery when a DGD passes `--model` through a container environment variable.

This design replaces that PR's case metadata, hardware profile, resolved-input, and report schemas
with the smaller case/hardware/suite layout below. It also treats final DGDs as the comparison
artifacts instead of introducing a custom report contract.

## Dimensions

The suite keeps three independent concepts explicit:

```text
case       deployment intent and native profiler inputs
hardware   provider-independent accelerator capabilities and GPU budget
suite      selected case and hardware combinations
```

Cloud-provider binding is not part of the hardware SKU. Provider-specific EFA, RoCE, InfiniBand,
image, and Kubernetes resource settings belong to the existing Recipe Kustomize Components and
Kustomize matrices. They can be composed with a generated Kustomize base for live deployment.

## Layout

```text
components/src/dynamo/profiler/tests/sweeper/
├── DESIGN.md
├── testsuite-issue-8469.yaml
├── cases/
│   └── qwen3-32b-vllm-disagg/
│       ├── dgdr-v1beta1.yaml
│       ├── sweeper.yaml
│       ├── recipe.yaml                  # optional
│       └── recipe.new.yaml              # ignored discovery result
├── hardware/
│   └── h200-sxm-16gpu/
│       ├── dgdr-v1beta1.patch.yaml
│       └── sweeper.patch.yaml
└── generated/
    ├── testsuite-issue-8469/
    │   └── h200-sxm-16gpu/
    │       └── qwen3-32b-vllm-disagg/
    └── manual/                         # ignored individual runs
```

There is one case hierarchy. There is no second recipe catalog or global requirements registry.

## Cases

A case contains the native inputs shared by every hardware run. Hardware values are omitted:

```yaml
# cases/qwen3-32b-vllm-disagg/dgdr-v1beta1.yaml
model: Qwen/Qwen3-32B
backend: vllm
image: nvcr.io/nvidia/ai-dynamo/dynamo-planner:1.5.0
workload:
  isl: 1024
  osl: 1024
  requestRate: 10
sla:
  ttft: 2000
  itl: 25
```

```yaml
# cases/qwen3-32b-vllm-disagg/sweeper.yaml
search_space:
  model_name: Qwen/Qwen3-32B
  deployment_mode: [disagg]
  backend: [vllm]
workload:
  isl: 1024
  osl: 1024
  request_rate: 10
goal:
  target: goodput_per_gpu
```

A matrix row without one or both native inputs is a coverage gap, not a fabricated runnable case.
It contains only `recipe.yaml`; suite execution prints the missing filenames and continues with the
other rows.

## Hardware configurations

Each hardware configuration contains one YAML merge patch per native input schema:

```yaml
# hardware/h200-sxm-16gpu/dgdr-v1beta1.patch.yaml
hardware:
  gpuSku: h200_sxm
  vramMb: 141120
  totalGpus: 16
  numGpusPerNode: 8
  interconnect: nvlink
  rdma: true
```

```yaml
# hardware/h200-sxm-16gpu/sweeper.patch.yaml
search_space:
  hardware_sku: h200_sxm
  gpu_budget: 16
```

Mappings merge recursively, scalar values replace existing values, lists replace existing lists,
and `null` removes a field. The runner validates that the composed inputs agree on their normalized
GPU SKU and GPU budget.

Hardware patches apply only to profiler inputs. They never repair a generated DGD. Provider
specialization, when needed, is a subsequent Kustomize composition rather than part of profiling.

## Suites and individual runs

A suite records concrete case and hardware combinations:

```yaml
source: https://github.com/ai-dynamo/dynamo/issues/8469

tests:
  - case: qwen3-32b-vllm-disagg
    hardware: h200-sxm-16gpu
```

The same combination can be generated without a suite by naming its case and hardware directly.
Suites are batch selection, not another source of case metadata.

## Optional recipes

An optional case-local `recipe.yaml` combines provenance and known hardware requirements:

```yaml
source: recipes/qwen3-32b/vllm/disagg-kv-router/deploy.yaml

requirements:
  h200-sxm-16gpu:
    gpus: 16
```

Recipe requirements only control whether that recipe is eligible for live deployment. A missing
recipe or requirement never prevents either generator from running. Normal execution tries recipes
only on hardware already listed in `requirements`.

With `--sweeper-discover-recipe-hardware`, the live test may try a recipe on the suite entry's
otherwise unknown hardware. Only a successful recipe deployment and inference request prove the
combination. The test leaves `recipe.yaml` unchanged and writes the proposed merged requirements to
the ignored adjacent `recipe.new.yaml`; a failed generated DGD does not establish recipe support.

## Composition

```text
case/dgdr-v1beta1.yaml
  + hardware/<name>/dgdr-v1beta1.patch.yaml
  → dgdr-v1beta1-composed.yaml
  → v1 profiler
  → dgd-profiler-v1beta1.yaml

case/sweeper.yaml
  + hardware/<name>/sweeper.patch.yaml
  → sweeper-composed.yaml
  → one Sweeper search and selected Candidate
  ├──→ AI Configurator renderer → dgd-sweeper-aic.yaml
  └──→ direct renderer          → dgd-sweeper-direct.yaml
```

Both Sweeper renderers receive the same selected Candidate.

## Generated files and goldens

Final DGDs are checked-in generated goldens:

```text
generated/<suite-name>/<hardware>/<case>/
├── dgd-profiler-v1beta1.yaml
├── dgd-sweeper-aic.yaml
└── dgd-sweeper-direct.yaml
```

These are the possible variants, not three mandatory files per case. A failed search or renderer
does not produce a placeholder golden; the absent variant and the runner failure make the coverage
gap explicit.

The suite name is the suite filename without `.yaml`, so every checked-in golden has one owning
testsuite. Individual `--hardware` runs default to the ignored `generated/manual/` tree. All final
DGDs are complete manifests that can be inspected, diffed, or applied individually. A golden CI
check regenerates the same suite in a temporary directory and compares the final `dgd-*.yaml` files
with the suite's goldens. Updating a golden is an explicit contributor action.

The same directory may contain ignored diagnostics after a local update run:

```text
dgdr-v1beta1-composed.yaml
sweeper-composed.yaml
candidate-sweeper.yaml
error-sweeper-aic.txt
error-sweeper-direct.txt
.cache/
```

Composed inputs and Candidates are useful for explaining a diff, but are not stable API outputs or
goldens. No custom report schema is defined; process output and the test framework carry execution
status.

Exact Sweeper goldens require reproducible search inputs and search ordering. Until that is proven,
CI must distinguish deterministic renderer checks from end-to-end search evidence rather than make
a stochastic search an exact-diff gate.

## Live validation

Live validation independently deploys each available DGD and, when eligible, the recipe source.
Each deployment must become ready and serve a valid inference request. Failure of one variant does
not erase evidence from another.

The initial harness validates the provider-neutral DGD on a matching cluster. Provider-specific
composition should reuse existing Recipe Components, for example AWS EFA or GKE RoCE. Hardware
remains the profiler input; Kustomize supplies the concrete Kubernetes platform binding without a
second provider schema in this suite.

## Non-goals

The comparison does not require:

- byte-for-byte equality between different generators;
- identical topology or replica choices;
- performance equivalence or benchmarking;
- case-local hardware declarations;
- a separate recipe hierarchy;
- a custom report archive; or
- post-generation patches that hide an invalid generator result.
