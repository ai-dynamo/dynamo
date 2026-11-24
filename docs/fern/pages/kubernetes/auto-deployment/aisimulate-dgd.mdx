---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Generate a DGD with AISim
subtitle: Run an offline recommendation and export the selected configuration as a manifest or Kustomize bundle
---

Use AISim's `recommend` command with NVIDIA Dynamo's `dgd` output adapter to generate a
`DynamoGraphDeployment` (DGD) before deploying it to Kubernetes. The search runs offline; it does
not create cluster resources. You inspect the generated manifest and apply it yourself.

<Warning>
**Experimental.** The AISim CLI and output-adapter contracts may change without a standard
deprecation period. Simulation results are estimates; validate the selected deployment on your
hardware before relying on its performance.
</Warning>

For an operator-managed search with persistent Requests, Runs, and Candidates, use
[Search with DGDR v1beta2](dgdr-v1beta2.md).

## Prerequisites

- A Linux environment with Python 3.11 through 3.13, Dynamo, and the matching `aisim` package.
  Follow [Dynamo Sweeper Integration](../../developer-guide/knowledge-base/modular-components/ai-simulate-experimental/sweeper-experimental/dynamo-integration.md#install)
  for a planner image or source installation. Install `ai-dynamo` in the same Python environment
  as the CLI: AISim alone does not register the `dgd` adapter.
- Access to the model's Hugging Face configuration and AISim performance data for your model,
  hardware, and backend. The search does not require serving GPUs.
- For deployment, a cluster with the [Dynamo operator](../installation/install-dynamo.md), matching
  DGD CRDs, and GPUs that match the configuration. The example targets H200 GPUs and vLLM.

## Generate and Deploy

<Steps>
<Step title="Verify the output adapter installation">

List the registered DGD adapter in the active environment:

```bash
python3 -c 'from importlib.metadata import entry_points; print([ep.value for ep in entry_points(group="aisim.output_adapters") if ep.name == "dgd"])'
```

Expected output:

```text
['dynamo.aisimulate.output.dgd:create_adapter']
```

If the list is empty, install Dynamo in this environment. From a source checkout, run
`python3 -m pip install --no-deps -e .` after completing the source installation above.

</Step>
<Step title="Create the recommendation input">

Save the following as `recommendation.yaml`. It searches one or two aggregated vLLM workers and
two scheduler token budgets for Qwen3-8B, with at most two GPUs per candidate:

```yaml
traffic:
  source: {type: synthetic, input_tokens: 1024, output_tokens: 128}
  load: {type: concurrency, concurrency: 8}
  stop: {requests: 50}

engine:
  mode: aggregated
  model: Qwen/Qwen3-8B
  hardware: h200_sxm
  backend: vllm
  context_length: 8192
  workers:
    aggregated:
      parallelism:
        preset:
          - {replicas: 1, tensor: 1, pipeline: 1, attention_data: 1, moe_tensor: 1, moe_expert: 1}
          - {replicas: 2, tensor: 1, pipeline: 1, attention_data: 1, moe_tensor: 1, moe_expert: 1}
      scheduler:
        max_batched_tokens: {choices: [4096, 8192]}
        max_sequences: 256
      kv_cache:
        block_size: 64
        prefix_caching: true
        capacity: {type: default, memory_fraction: 0.9}

optimization:
  target: throughput
  constraints: {max_candidate_gpus: 2}

optimizer:
  algorithm: random
  max_trials: 4
  parallelism: 1
  candidate_timeout_seconds: 120
  seed: 42

dgd:
  name: qwen-search
  namespace: inference
  generator: aic
  format: manifest
  output_file: deployment.yaml
  runtime_image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.7.0
  num_gpus_per_node: 8
```

Set `engine.hardware`, the worker search space, and `dgd.num_gpus_per_node` to match your target
cluster. Use a `dgd.runtime_image` compatible with your backend and operator. The GPU budget limits
each candidate; `num_gpus_per_node` describes node capacity rather than allocating eight GPUs.

`engine.model` selects the model evaluated and served. `dgd.name` names the Kubernetes resource.
The top-level `dgd` section configures output only and does not change the search objective.

</Step>
<Step title="Run the search">

```bash
aisimulate recommend \
  --stack dynamo \
  --config recommendation.yaml \
  --output dgd \
  --output-dir generated
```

`--stack dynamo` selects Dynamo replay and its configured Router or Planner adapters. `--output dgd`
independently selects DGD generation. The CLI applies `--set` overrides, extracts the `dgd`
section, and validates the recommendation input before searching.

For a scalar objective, the output adapter renders the final selected winner. It does not publish
intermediate trial manifests. Inspect these files after a successful search:

```text
generated/
  recommendation.json
  recommendation.csv
  recommendations/
  deployment.yaml
  index.json
```

`recommendation.json` contains the search results, and `recommendations/` contains selected
prediction configurations. `index.json` lists the DGD artifacts. For this manifest output:

```json
{
  "artifacts": [{"path": "deployment.yaml"}],
  "output": "dgd",
  "renderer": "aic"
}
```

Use a fresh output directory for each run. To override an output setting without editing the file:

```bash
aisimulate recommend \
  --stack dynamo \
  --config recommendation.yaml \
  --set dgd.name=qwen-experiment \
  --output dgd \
  --output-dir generated-experiment
```

</Step>
<Step title="Inspect and apply the manifest">

Review the selected worker replicas, GPU requests, model arguments, runtime image, and namespace
in `generated/deployment.yaml`. Add cluster-specific configuration such as model-cache volumes,
credentials, or scheduling constraints where required; see [Deploy with DGD](../model-deployment/deploy-with-dgd.md).

Create the example namespace if it does not exist, then ask the API server to validate the manifest:

```bash
kubectl create namespace inference --dry-run=client -o yaml | kubectl apply -f -
kubectl apply --dry-run=server -f generated/deployment.yaml
```

Apply the reviewed manifest and wait for readiness:

```bash
kubectl apply -f generated/deployment.yaml
kubectl wait --for=condition=Ready dynamographdeployment/qwen-search \
  -n inference --timeout=600s
```

Expose the service and verify an inference request as described in
[Expose the Frontend](../model-deployment/expose-the-frontend.md).

</Step>
</Steps>

## Export a Kustomize Bundle

To export a bundle instead of a single manifest, replace the complete `dgd` section with:

```yaml
dgd:
  name: qwen-search
  namespace: inference
  format: kustomize
  output_dir: deployment
  runtime_image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.7.0
  num_gpus_per_node: 8
```

Remove `output_file`; the two destination fields are mutually exclusive. Run the recommendation
command with a fresh CLI output directory, for example `--output-dir generated-kustomize`.
The bundle contains `generated-kustomize/deployment/deploy.yaml` and `kustomization.yaml`.
Inspect the rendered bundle before applying it:

```bash
kubectl kustomize generated-kustomize/deployment
kubectl apply --dry-run=server -k generated-kustomize/deployment
kubectl apply -k generated-kustomize/deployment
```

## Export Pareto Alternatives

For multiple final tradeoffs, set `optimization.target: pareto` and include a literal `{index}` in
both the DGD name and its output destination. For manifest output, replace `optimization` and `dgd`
in the original input with:

```yaml
optimization:
  target: pareto
  constraints: {max_candidate_gpus: 2}
dgd:
  name: qwen-candidate-{index}
  namespace: inference
  format: manifest
  output_file: candidate-{index}.yaml
  runtime_image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.7.0
  num_gpus_per_node: 8
```

Run with a fresh CLI output directory. The adapter writes every final selected Pareto candidate
using zero-based, three-digit indices, such as `candidate-000.yaml` and `candidate-001.yaml`.
The number of alternatives depends on the search results. Inspect `index.json` to find them.

For Kustomize output, use `format: kustomize` and `output_dir: candidate-{index}` instead of
`output_file`. Each candidate receives an independent bundle. Choose one alternative to deploy;
applying every candidate creates multiple deployments and consumes their combined GPU capacity.

## Troubleshooting

| Symptom | Action |
|---|---|
| The CLI cannot find `dgd` | Verify the `aisim.output_adapters` entry point in the same environment as `aisimulate`. Install Dynamo as well as `aisim`. |
| Missing `dgd` configuration or an invalid destination | Supply a top-level `dgd` mapping and exactly the destination field required by its format. |
| A nightly or digest image fails version validation | Set `dgd.runtime_version_override` to the image's Dynamo version in `MAJOR.MINOR.PATCH` form. |
| No feasible candidate | Inspect `recommendation.json`; revise the workload, constraints, or search space. No DGD is rendered without a selected candidate. |
| Model or performance-data lookup fails | Verify model access and AISim data coverage for the selected model, backend, and hardware. Do not substitute fixed timings for a capacity-planning result. |

See [DGD Output Adapter Configuration](../../reference/components/aisimulate-dgd-output.md) for the
full output contract, validation rules, and current limitations.
