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
| vLLM | Disaggregated | [example](vllm/disagg/deploy-v1alpha1.template.yaml) | [common example](vllm/disagg/deploy-v1beta1.template.yaml); [advanced ComputeDomain and DRA example](vllm/disagg/deploy-v1beta1-compute-domain.template.yaml) |
| SGLang | Aggregate | [example](sglang/agg/deploy-v1alpha1.template.yaml) | [common example](sglang/agg/deploy-v1beta1.template.yaml); [advanced ComputeDomain and DRA example](sglang/agg/deploy-v1beta1-compute-domain.template.yaml) |
| SGLang | Disaggregated | [example](sglang/disagg/deploy-v1alpha1.template.yaml) | [example](sglang/disagg/deploy-v1beta1.template.yaml) |
| TensorRT-LLM | Aggregate | [example](trtllm/agg/deploy-v1alpha1.template.yaml) | [example](trtllm/agg/deploy-v1beta1.template.yaml) |
| TensorRT-LLM | Disaggregated | [example](trtllm/disagg/deploy-v1alpha1.template.yaml) | [beta-shape example](trtllm/disagg/deploy-v1beta1.template.yaml); target-cluster qualification required |

## Select and copy an example

1. Select the same framework.
2. Select aggregate or disaggregated topology.
3. Select `v1beta1` for new recipe work.
4. Select an advanced ComputeDomain example only when the recipe uses ComputeDomain or DRA.
5. Select `v1alpha1` only to maintain an alpha recipe.

The catalog does not yet include a TensorRT-LLM ComputeDomain example. Start from a reviewed TensorRT-LLM deployment with the required mechanism instead of combining unrelated examples without framework-owner review.

From the Dynamo repository root, copy the selected file to the new recipe path. Replace the example destination with the path for the recipe you are adding.

```bash
mkdir -p recipes/qwen3-0.6b/vllm/agg-example
cp recipes/templates/vllm/agg/deploy-v1beta1.template.yaml \
  recipes/qwen3-0.6b/vllm/agg-example/deploy.yaml
```

Edit `deploy.yaml`: change the DGD name, model and served-model values, image, runtime arguments, replicas, GPU intent, and framework configuration as needed. Keep external reference names, such as `model-cache` and `hf-token-secret`, internally consistent. These are concrete Kubernetes object references: direct application requires same-named objects in the target namespace, while a cluster Kustomization can replace the references with site-specific names.

## Field ownership and Kustomize boundary

| Concern | Recipe example owns | Cluster Kustomization owns |
| --- | --- | --- |
| API and graph | DGD API, framework, topology, component names | Source-native API patches; no API conversion |
| Model runtime | Image, command, arguments, model identity, framework configuration | Optional organization image overrides |
| Scale intent | Replicas, multinode intent, CPU, memory, and GPU counts | Cluster-approved resource adjustments |
| Scheduling | None | Scheduler, node selection, affinity, tolerations, runtime class, priority, topology references, and label keys |
| Artifacts and application Secrets | Canonical default object references and container-visible paths | Object provisioning or coordinated site-specific reference and path overrides |
| Registry credentials and networking | Framework transport intent, such as NIXL roles | Pull-Secret references, provider annotations, resources, interfaces, and endpoints |
| ComputeDomain and DRA | Logical ComputeDomain and claim relationships | Site placement and physical driver or device-class binding |
| Namespace | No `metadata.namespace` | Apply or orchestration selects the namespace |

The current shared provider Components target alpha `spec.services` and offer limited provider networking support. They do not patch beta `spec.components[].podTemplate`; beta bases need beta-targeted patches. A Kustomize Component does not convert a DGD API.

Recipe examples must omit cluster scheduling, cluster-specific artifact and Secret identities, image-pull Secrets, provider-network bindings, host bindings, and `metadata.namespace`. They retain portable serving behavior, canonical default object references, and the fields needed by framework runtime commands.

## Shared memory behavior

For `v1beta1` workers, the operator owns `/dev/shm` unless injection is explicitly disabled:

- When `sharedMemorySize` is omitted, the operator injects an 8Gi `/dev/shm` volume.
- A positive `sharedMemorySize` makes the operator inject a volume of that size and drop any manual mount at `/dev/shm`.
- `sharedMemorySize: "0"` disables operator injection. This is the only mode in which a manual `/dev/shm` volume applies, and the catalog does not use it.

## Limits

These examples are copy starts, not synchronized source mirrors or cluster-qualified deployments. Catalog checks cover YAML and static structure; they do not prove admission, scheduling, networking, model access, readiness, or benchmark performance. Qualify a copied recipe with its cluster Kustomization and target-cluster policy.

## Source recipes and validation

Each example starts from the linked repository recipe and is then normalized to the catalog contract. The link records the starting point; it does not make the example a synchronized mirror. Unless a row says otherwise, the model, image, runtime arguments, scale, GPU intent, and framework configuration are retained from that source. All examples use canonical component names, canonical default object references, standard probe shapes, and omit cluster-supplied settings.

"Source-derived" means the workload values come from the linked recipe. "API translation" means the runtime bundle was also projected into a different DGD API shape. Unless a table row records additional evidence, these statuses receive YAML and static catalog checks only; neither claims target-cluster readiness or runtime qualification.

| Example | Source recipe | Preserved source fields | Deliberate template adjustments | Validation |
| --- | --- | --- | --- | --- |
| [vLLM aggregate alpha](vllm/agg/deploy-v1alpha1.template.yaml) | [Llama 3 70B aggregate](../llama-3-70b/vllm/agg/deploy.yaml) | Runtime bundle, scale, GPU intent, 20Gi shared memory | Canonical names and a 60-minute worker startup budget; the source had no worker probe | Source-derived; static checks |
| [vLLM disaggregated alpha](vllm/disagg/deploy-v1alpha1.template.yaml) | [Llama 3 70B multi-node disaggregated](../llama-3-70b/vllm/disagg-multi-node/deploy.yaml) | Runtime and transfer bundle, scale, GPU intent, 80Gi shared memory | Canonical names, transfer hook, and a 60-minute worker startup budget; the source had no worker probe | Source-derived; static checks |
| [vLLM aggregate beta](vllm/agg/deploy-v1beta1.template.yaml) | [DeepSeek V4 Flash aggregate](../deepseek-v4/deepseek-v4-flash/vllm/agg-b200-agentic/deploy.yaml) | Runtime bundle, scale, GPU intent | Online model access, 64Gi shared memory, and a 60-minute worker startup budget; the source did not define the latter two values | Source-derived; static checks |
| [vLLM disaggregated beta](vllm/disagg/deploy-v1beta1.template.yaml) | [GPT-OSS 120B disaggregated](../gpt-oss-120b/vllm/disagg-b200-agentic/deploy.yaml) | Runtime and transfer bundle, scale, GPU intent, 64Gi shared-memory amount, 60-minute worker startup budget | Operator-owned shared memory and the anchored transfer hook | Source-derived; static checks |
| [vLLM disaggregated ComputeDomain beta](vllm/disagg/deploy-v1beta1-compute-domain.template.yaml) | [DeepSeek V4 Pro ComputeDomain deployment](../deepseek-v4/deepseek-v4-pro/vllm/disagg/gb200/deploy.yaml) | Runtime and transfer bundle, two-node workers, 4 GPUs per pod, 40Gi/200Gi shared memory, 5,400-second worker startup budget | Native beta components, canonical names, standard probes, and the anchored transfer hook | Alpha configuration: render and server-side dry run; beta translation: static checks |
| [SGLang aggregate alpha](sglang/agg/deploy-v1alpha1.template.yaml) | [Nemotron 3 Super FP8 aggregate](../nemotron-3-super-fp8/sglang/agg/deploy.yaml) | Runtime bundle, scale, GPU intent, 16Gi shared memory | Canonical names and a 60-minute worker startup budget; the source had no worker probe | Source-derived; static checks |
| [SGLang disaggregated alpha](sglang/disagg/deploy-v1alpha1.template.yaml) | [Nemotron 3 Super FP8 disaggregated](../nemotron-3-super-fp8/sglang/disagg/deploy.yaml) | Runtime and transfer bundle, scale, GPU intent, 16Gi shared memory | Canonical names, anchored transfer setting, and a 120-minute worker startup budget; the source had no worker probe | Source-derived; static checks |
| [SGLang aggregate beta](sglang/agg/deploy-v1beta1.template.yaml) | [Inkling aggregate](../inkling/sglang/agg-b200/deploy.yaml) | Runtime bundle, scale, GPU intent, 512Gi shared-memory amount and scratch volumes | Operator-owned shared memory, explicit offline operation, and a 60-minute worker startup budget; the source had no worker probe | Source-derived; static checks |
| [SGLang disaggregated beta](sglang/disagg/deploy-v1beta1.template.yaml) | [GLM-5.2 disaggregated](../glm-5.2/sglang/disagg-b200-agentic/deploy.yaml) | ConfigMaps, runtime and transfer bundle, scale, 64Gi shared memory, 120-minute worker startup budget | Canonical names and anchored transfer setting | Source-derived; static checks |
| [SGLang ComputeDomain beta](sglang/agg/deploy-v1beta1-compute-domain.template.yaml) | [Qwen 3.8 ComputeDomain aggregate](../qwen3.8-2.4t-a95b-fp8/sglang/agg-gb300-chat/deploy.yaml) | ConfigMap, DRA chain, runtime bundle, node/GPU shape, offline operation, 7,200-second worker startup budget | 200Gi shared memory because the source did not size `/dev/shm`; reduced capability set | Source-derived; static checks |
| [TensorRT-LLM aggregate alpha](trtllm/agg/deploy-v1alpha1.template.yaml) | [GPT-OSS 120B aggregate](../gpt-oss-120b/trtllm/agg/deploy.yaml) | ConfigMap, runtime bundle, scale, GPU intent, 80Gi shared memory | Canonical names and a 60-minute worker startup budget; the source had no worker probe | Source-derived; static checks |
| [TensorRT-LLM disaggregated alpha](trtllm/disagg/deploy-v1alpha1.template.yaml) | [Nemotron 3 Super FP8 disaggregated](../nemotron-3-super-fp8/trtllm/disagg/deploy.yaml) | ConfigMaps, runtime and transfer bundle, scale, 16Gi shared memory, 6,000-second worker startup budget | Canonical names and standard probe shape | Source-derived; static checks |
| [TensorRT-LLM aggregate beta](trtllm/agg/deploy-v1beta1.template.yaml) | [Nemotron 3.5 Lightning aggregate](../nemotron-3.5-lightning/trtllm/agg-b200-bf16/deploy.yaml) | ConfigMap, runtime bundle, scale, GPU intent, 40Gi shared memory, 3,600-second worker startup budget | Canonical names, online model access, and reduced cluster policy | Source-derived; static checks |
| [TensorRT-LLM disaggregated beta](trtllm/disagg/deploy-v1beta1.template.yaml) | [Nemotron 3 Super FP8 disaggregated](../nemotron-3-super-fp8/trtllm/disagg/deploy.yaml) | ConfigMaps, runtime and transfer bundle, scale, 16Gi shared memory, 6,000-second worker startup budget | Native beta components and canonical names | API translation; static checks |

## Field contract

Apply the strongest requirement that affects a field. The DGD API may permit a value that a selected Kustomize Component addresses by an exact name or position; in that case, the Component contract is stricter than admission.

### Fixed requirements

- Keep `kind: DynamoGraphDeployment` and its source-native API shape. `nvidia.com/v1alpha1` uses the `spec.services` map; `nvidia.com/v1beta1` uses the `spec.components` list. Kustomize patches do not convert between them.
- In beta Pod templates, the runtime container is named `main`. Alpha services use `extraPodSpec.mainContainer`.
- Keep `spec.backendFramework` aligned with the runtime image, command, arguments, environment, and any framework ConfigMaps.
- Keep coordinated references synchronized as complete bundles:
  - ComputeDomain/DRA: the ComputeDomain channel template name matches each Pod or alpha service `resourceClaimTemplateName`, and each claim name matches the corresponding container or service `resources.claims` name.
  - ConfigMap runtime configuration: ConfigMap name and key, volume, mount, path, and `--config` or `--extra-engine-args` consumer all agree.
  - Model identity: `MODEL_NAME`, `SERVED_MODEL_NAME`, and their command substitutions agree with the intended served model.
  - Disaggregated transfer: prefill and decode roles, modes, connector/backend, and compatible transfer settings change together.

### Catalog conventions

- Beta aggregate examples list `Frontend`/`frontend`, then `Worker`/`worker`. Beta disaggregated examples list `Frontend`/`frontend`, `PrefillWorker`/`prefill`, then `DecodeWorker`/`decode`. Alpha examples use the same canonical service keys and roles, but map order is not patch-semantic. Put optional `planner` or `epp` components after the canonical components.
- `model-cache` is the canonical default PVC reference and patch anchor when present. Direct application requires a same-named PVC in the target namespace; a cluster Kustomization may replace the reference with a site-specific name. For a cacheless recipe, remove the PVC or volume, every mount, and cache-path couplings such as `HF_HOME` and local-filesystem model paths as one change; also remove any cache-binding Component from the cluster Kustomization. Retain the runtime's required model selector, such as `--model` or `--model-path`, when it names an online model ID.
- `hf-token-secret` is the canonical default Secret reference and patch anchor for online model access. Direct application requires a same-named Secret in the target namespace; a cluster Kustomization may replace the reference with a site-specific name. For offline operation, remove every alpha `envFromSecret` or beta `envFrom.secretRef` reference, add `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` where the runtime needs them, and retain the pre-populated cache unless the recipe is also cacheless. Update every reference together when renaming either default object.
- Networking hooks are stable patch anchors. `KV_TRANSFER_CONFIG` and `SGLANG_DISAGGREGATION_NIXL_BACKEND`, when present, are the first worker environment entries and carry portable defaults. Override their values with guarded `test` plus `replace` operations; do not remove the hook while its argument or runtime consumer remains.
- Review canonical names, roles, ordering, and hook positions before contribution. If no guarded Component is selected, a Kustomize build has no patch precondition that can check them. A selected JSON Patch Component tests the names, roles, positions, and replaced values it depends on and fails the build when they differ. This prevents a rename from silently creating an extra component through merge behavior.

### Editable values

Edit DGD and ConfigMap names, model identities, image tags, replicas, `multinode.nodeCount`, CPU, memory, GPU counts, engine flags, ConfigMap contents, probe budgets such as `failureThreshold`, and `sharedMemorySize` for the target workload.

These fields are examples, not independent knobs. Keep the framework, model, image, parallelism, GPU shape, memory, transfer configuration, and engine settings compatible. YAML parsing and API admission do not prove readiness, correct responses, or performance.

### Cluster-supplied settings

Portable recipe examples omit the following fields. Supply them through cluster Components, guarded site patches, or the deployment environment:

- `metadata.namespace`;
- `nodeSelector`, affinity, tolerations, `schedulerName`, `runtimeClassName`, and `priorityClassName`;
- `imagePullSecrets`, site-specific PVC names, storage classes, and provisioning details;
- provider network annotations and resources, physical interface or device names, endpoints, host bindings, and physical ComputeDomain/DRA realization; and
- cluster-owned environment families such as `NCCL_SOCKET_IFNAME`, `GLOO_SOCKET_IFNAME`, and site interface or device selections.

Cluster Components may add whole policy fields or append environment entries. Defining those values in the base can overwrite policy or create duplicate environment variables. A narrowly scoped runtime capability may remain only when the selected image or transport requires it and the template documents that requirement; provider-wide privileges and host access remain cluster policy.
