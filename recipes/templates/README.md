<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Representative recipe examples

This catalog contains concrete, copyable `DynamoGraphDeployment` examples. Select the closest example, copy it into a recipe directory as `deploy.yaml`, and edit the portable serving fields. The `template` filename means "copyable example"; no renderer processes these files.

Each YAML file contains one DGD and any supporting ConfigMaps, ComputeDomains, or DRA resources that the DGD requires. Supporting resources appear before the DGD. The directories intentionally contain no `kustomization.yaml` files or copied OpenAPI data.

## Catalog

| Framework | Topology | `v1alpha1` | `v1beta1` |
| --- | --- | --- | --- |
| vLLM | Aggregate | [example](vllm/agg/deploy-v1alpha1.template.yaml) | [example](vllm/agg/deploy-v1beta1.template.yaml) |
| vLLM | Disaggregated | [example](vllm/disagg/deploy-v1alpha1.template.yaml) | [example](vllm/disagg/deploy-v1beta1.template.yaml) |
| SGLang | Aggregate | [example](sglang/agg/deploy-v1alpha1.template.yaml) | [common example](sglang/agg/deploy-v1beta1.template.yaml); [advanced ComputeDomain and DRA example](sglang/agg/deploy-v1beta1-compute-domain.template.yaml) |
| SGLang | Disaggregated | [example](sglang/disagg/deploy-v1alpha1.template.yaml) | [example](sglang/disagg/deploy-v1beta1.template.yaml) |
| TensorRT-LLM | Aggregate | [example](trtllm/agg/deploy-v1alpha1.template.yaml) | [example](trtllm/agg/deploy-v1beta1.template.yaml) |
| TensorRT-LLM | Disaggregated | [example](trtllm/disagg/deploy-v1alpha1.template.yaml) | [synthesized example](trtllm/disagg/deploy-v1beta1.template.yaml); pending live-cluster qualification |

## Select and copy an example

1. Select the same framework.
2. Select aggregate or disaggregated topology.
3. Select `v1beta1` for new recipe work.
4. Select the advanced SGLang example only when the recipe uses a ComputeDomain or DRA.
5. Select `v1alpha1` only to maintain an alpha recipe.

From the Dynamo repository root, copy the selected file to the new recipe path. Replace the example destination with the path for the recipe you are adding.

```bash
mkdir -p recipes/qwen3-0.6b/vllm/agg-example
cp recipes/templates/vllm/agg/deploy-v1beta1.template.yaml \
  recipes/qwen3-0.6b/vllm/agg-example/deploy.yaml
```

Edit `deploy.yaml`: change the DGD name, model and served-model values, image, runtime arguments, replicas, GPU intent, and framework configuration as needed. Keep any logical external references, such as `model-cache` and `hf-token-secret`, internally consistent. An internal cluster Kustomization can reference the copied multi-document file and apply cluster-specific changes.

## Field ownership and Kustomize boundary

| Concern | Recipe example owns | Cluster Kustomization owns |
| --- | --- | --- |
| API and graph | DGD API, framework, topology, component names | Source-native API patches; no API conversion |
| Model runtime | Image, command, arguments, model identity, framework configuration | Optional organization image overrides |
| Scale intent | Replicas, multinode intent, CPU, memory, and GPU counts | Cluster-approved resource adjustments |
| Scheduling | None | Scheduler, node selection, affinity, tolerations, runtime class, priority, topology references, and label keys |
| Artifacts and application Secrets | Logical names and container-visible paths | Physical PVC and Secret bindings, plus coordinated path overrides |
| Registry credentials and networking | Framework transport intent, such as NIXL roles | Pull-Secret references, provider annotations, resources, interfaces, and endpoints |
| ComputeDomain and DRA | Logical ComputeDomain and claim relationships | Site placement and physical driver or device-class binding |
| Namespace | No `metadata.namespace` | Apply or orchestration selects the namespace |

The current shared provider Components target alpha `spec.services` and offer limited provider networking support. They do not patch beta `spec.components[].podTemplate`; beta bases need beta-targeted patches. A Kustomize Component does not convert a DGD API.

Recipe examples must omit cluster scheduling, physical artifact and Secret bindings, image-pull Secrets, provider-network bindings, host bindings, and `metadata.namespace`. They retain portable serving behavior, logical references, and the fields needed by framework runtime commands.

## Shared memory behavior

For `v1beta1` workers, the operator owns `/dev/shm` unless injection is explicitly disabled:

- When `sharedMemorySize` is omitted, the operator injects an 8Gi `/dev/shm` volume.
- A positive `sharedMemorySize` makes the operator inject a volume of that size and drop any manual mount at `/dev/shm`.
- `sharedMemorySize: "0"` disables operator injection. This is the only mode in which a manual `/dev/shm` volume applies, and the catalog does not use it.

## Limits

These examples are copy starts, not synchronized source mirrors or cluster-qualified deployments. The checks prove YAML syntax and catalog conformance; they do not prove admission, scheduling, networking, model access, readiness, or benchmark performance. Live-cluster qualification belongs to the Kustomize integration and cluster-policy workstream.
