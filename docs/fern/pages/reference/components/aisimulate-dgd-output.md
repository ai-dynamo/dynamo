---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: AISim DGD Output Adapter Configuration
subtitle: Fields, selection rules, and artifacts for recommend --output dgd
---

NVIDIA Dynamo registers `dgd` in the `aisim.output_adapters` entry-point group. Use it with
`aisimulate recommend --output dgd` to render selected recommendations as
`DynamoGraphDeployment` (DGD) resources. For a complete workflow, see
[Generate a DGD with AISim](../../kubernetes/auto-deployment/aisimulate-dgd.mdx).

> [!WARNING]
> **Experimental.** The CLI and output-adapter contracts may change without a standard deprecation
> period.

## Configuration Fields

The adapter consumes the top-level `dgd` mapping after AISim applies `--set` overrides.
Requesting `--output dgd` requires that mapping. Unknown keys are rejected.

| Field | Type and default | Contract |
|---|---|---|
| `name` | Required nonempty string | Resource name. No path separators. Only `{index}` may be templated; required for Pareto output and rejected for scalar output. |
| `namespace` | Optional string, default `null` | Sets `metadata.namespace`. Omitted, `null`, or empty removes the namespace from the generated manifest. |
| `generator` | `aic` (default) or `direct` | `aic` uses AISim's generator bridge. `direct` uses Dynamo's direct renderer. Both apply Dynamo runtime finalization. |
| `format` | `manifest` (default) or `kustomize` | Selects a YAML file or an independent Kustomize bundle per selected candidate. |
| `output_file` | Optional nonempty string | Required for `manifest`, forbidden for `kustomize`. A filename ending in `.yaml` or `.yml`, without path separators; only `{index}` may be templated. |
| `output_dir` | Optional nonempty string | Required for `kustomize`, forbidden for `manifest`. A directory name without path separators; `.` and `..` are rejected. Only `{index}` may be templated. |
| `runtime_image` | Required nonempty string | Runtime image for generated containers. Without a version override, its tag must be a canonical `MAJOR.MINOR.PATCH` version. |
| `runtime_version_override` | Optional string, default `null` | Explicit Dynamo version in canonical `MAJOR.MINOR.PATCH` form for custom, nightly, or digest image references. Also sets `runtimeVersionOverride` on each generated component. |
| `num_gpus_per_node` | Required positive integer | GPU capacity of a target node, used during generation. Does not set the search GPU budget. |

Output destinations are relative to the CLI `--output-dir`, not the input file's directory.
For Pareto results, the selected destination must contain `{index}`. Scalar results reject it.
The adapter substitutes zero-based indices formatted as `000`, `001`, and so on.

## Valid and Invalid Forms

A scalar manifest configuration:

```yaml
dgd:
  name: qwen
  output_file: deployment.yaml
  runtime_image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.7.0
  num_gpus_per_node: 8
```

An independent Kustomize bundle for each Pareto result:

```yaml
dgd:
  name: qwen-{index}
  format: kustomize
  output_dir: candidate-{index}
  runtime_image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.7.0
  num_gpus_per_node: 8
```

The following is invalid because `output_file` is forbidden for Kustomize output:

```yaml
dgd:
  name: qwen
  format: kustomize
  output_file: deployment.yaml
  output_dir: deployment
  runtime_image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.7.0
  num_gpus_per_node: 8
```

For a custom image, declare its actual Dynamo version separately:

```yaml
dgd:
  name: qwen
  output_file: deployment.yaml
  runtime_image: registry.example.com/dynamo-vllm:custom
  runtime_version_override: 1.7.0
  num_gpus_per_node: 8
```

The override controls version-dependent generation; it does not change the image's contents or
select a different backend version for simulation.

## Selection and Artifacts

AISim owns simulation, search, ranking, and final candidate selection. Dynamo owns the
conversion of those selected candidates into DGDs. `--stack engine|dynamo` selects simulation;
`--output dgd` independently enables generation. Neither option substitutes for the other.

For scalar results, the adapter renders one final selected winner. For Pareto results, it renders
all final selected candidates. If no candidate is selected, generation fails without producing a
deployment.
The adapter does not publish intermediate incumbents or apply resources to a cluster.

Manifest output writes one YAML file per selected candidate. Kustomize output writes one directory
per candidate, containing `deploy.yaml` and a `kustomization.yaml` that references it.
The adapter also writes `index.json` in the CLI output directory:

```json
{
  "renderer": "aic",
  "output": "kustomize",
  "artifacts": [{"path": "candidate-000"}, {"path": "candidate-001"}]
}
```

The index uses `output: dgd` for manifest output and `output: kustomize` for bundles. Artifact paths
are relative to the CLI output directory. AISim's `recommendation.json`,
`recommendation.csv`, and selected prediction configurations under `recommendations/` remain
separate from this index.

## Compatibility and Limits

- The adapter supports candidates for vLLM, SGLang, and TensorRT-LLM. Every candidate must carry a
  nonempty backend version. Match the runtime image to the selected backend and its evaluated
  configuration.
- Omitting `--output dgd` keeps the recommendation-only workflow. Remove the adapter-owned `dgd`
  mapping when running without that output adapter.
- Only the public recommendation input uses `engine.model`. Adapter fields such as `name` and
  `output_file` do not select a model.
- DGD generation currently supports a Router `prefill_load_model.type` of `none`; other types fail
  materialization.
- The output configuration has no fields for arbitrary DGD overrides, model-cache PVCs,
  tolerations, or trust-remote-code settings. Inspect and customize generated resources for those
  deployment requirements before applying them.

The implementation sources are the
[adapter configuration and selection logic](https://github.com/ai-dynamo/dynamo/blob/main/components/src/dynamo/aisimulate/output/dgd/adapter.py)
and [shared rendering rules](https://github.com/ai-dynamo/dynamo/blob/main/components/src/dynamo/aisimulate/output/dgd/renderers/base.py).
