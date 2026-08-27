<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Representative recipe examples

This catalog contains concrete, copyable `DynamoGraphDeployment` examples. Select the closest example, copy it into a recipe directory as `deploy.yaml`, and edit the portable serving fields. The `template` filename means "copyable example"; no renderer processes these files.

Each YAML file contains one DGD and any supporting ConfigMaps, ComputeDomains, or DRA resources that the DGD requires. Supporting resources appear before the DGD. The framework and topology example directories intentionally contain no `kustomization.yaml` files or copied OpenAPI data.

For the end-to-end authoring, validation, and shipping workflow, see the [recipe contribution guide](../CONTRIBUTING.md).

## Catalog

| Framework | Topology | `v1alpha1` | `v1beta1` |
| --- | --- | --- | --- |
| vLLM | Aggregate | [example](vllm/agg/deploy-v1alpha1.template.yaml) | [example](vllm/agg/deploy-v1beta1.template.yaml) |
| vLLM | Disaggregated | [example](vllm/disagg/deploy-v1alpha1.template.yaml) | [common example](vllm/disagg/deploy-v1beta1.template.yaml); [advanced ComputeDomain and DRA example](vllm/disagg/deploy-v1beta1-compute-domain.template.yaml) |
| SGLang | Aggregate | [example](sglang/agg/deploy-v1alpha1.template.yaml) | [common example](sglang/agg/deploy-v1beta1.template.yaml); [advanced ComputeDomain and DRA example](sglang/agg/deploy-v1beta1-compute-domain.template.yaml) |
| SGLang | Disaggregated | [example](sglang/disagg/deploy-v1alpha1.template.yaml) | [example](sglang/disagg/deploy-v1beta1.template.yaml) |
| TensorRT-LLM | Aggregate | [example](trtllm/agg/deploy-v1alpha1.template.yaml) | [example](trtllm/agg/deploy-v1beta1.template.yaml) |
| TensorRT-LLM | Disaggregated | [example](trtllm/disagg/deploy-v1alpha1.template.yaml) | [beta-shape example](trtllm/disagg/deploy-v1beta1.template.yaml); target-cluster qualification required |

## Choose a template

Follow these branches in order:

1. **API:** Select `v1beta1` for a new recipe. Select `v1alpha1` only when maintaining an existing alpha recipe. Kustomize cannot convert one source API shape into the other.
2. **Framework:** Select vLLM, SGLang, or TensorRT-LLM to match the runtime image, command, arguments, environment, and framework configuration.
3. **Topology:** Select aggregate for a frontend and one worker role. Select disaggregated for separate prefill and decode roles.
4. **ComputeDomain and Dynamic Resource Allocation (DRA):** Select an advanced example only when the source deployment requires that mechanism and the catalog has an exact example for the selected framework and topology.

The first three choices select one of the six alpha or six common beta examples in the catalog. The fourth choice selects one of the two advanced beta examples, for 14 files in total.

The catalog has exact ComputeDomain examples for beta vLLM disaggregated serving and beta SGLang aggregate serving. It does not have exact ComputeDomain examples for the other framework and topology combinations. When no file matches all four choices, start from the common example with the correct API, framework, and topology. Retain the required mechanism from a reviewed source deployment; do not combine unrelated examples mechanically.

### Gaps that require owner review

- The TensorRT-LLM beta disaggregated example is an alpha-to-beta API translation with static checks only. It requires target-cluster admission, readiness, and runtime qualification.
- The catalog has no TensorRT-LLM ComputeDomain example. Start from a reviewed TensorRT-LLM deployment that already uses the required mechanism instead of combining unrelated examples.

When no exact template fits, involve the recipe/performance owners and the selected framework's backend owners. Use the [owner lookup helper](../../.github/codeowners/README.md) with the proposed recipe path and concrete files for every affected subsystem. Do not pass only a directory. For example:

```bash
python3 .github/codeowners/who_owns.py --codeowners CODEOWNERS \
  recipes/my-model/trtllm/disagg/deploy.yaml \
  components/src/dynamo/trtllm/backend_args.py \
  deploy/operator/internal/dynamo/backend_trtllm.go
```

For frontend behavior, include `components/src/dynamo/frontend/frontend_args.py`. For standalone-router behavior, include `components/src/dynamo/router/args.py`. Include both paths when both subsystems are affected.

## Copy and edit a template

From the Dynamo repository root, copy the selected file to the new recipe path. Replace the example destination with the path for the recipe you are adding.

```bash
mkdir -p recipes/qwen3-0.6b/vllm/agg-example
cp recipes/templates/vllm/agg/deploy-v1beta1.template.yaml \
  recipes/qwen3-0.6b/vllm/agg-example/deploy.yaml
```

Edit `deploy.yaml`: change the DGD name, model and served-model values, image, runtime arguments, replicas, GPU intent, and framework configuration as needed. Keep external reference names, such as `model-cache` and `hf-token-secret`, internally consistent. These are concrete Kubernetes object references: direct application requires same-named objects in the target namespace, while a cluster Kustomization can replace the references with site-specific names.

## Adapt without adding a template

Change workload profiles in the copied `deploy.yaml`. Do not add catalog files for scalar or tuning differences that preserve the same API, framework, graph, and supporting-resource mechanisms.

### Long-context and QoS profiles

Treat a vLLM context-length or quality-of-service profile as a coordinated runtime change. Change `--max-model-len` on every affected worker, then review these coupled fields instead of applying a fixed conversion:

- retune `--max-num-batched-tokens` and other concurrency or memory controls;
- add `VLLM_ALLOW_LONG_MAX_MODEL_LEN: "1"` only when the selected vLLM and model combination requires an override of the model-derived maximum;
- remove `--spec-method` and `--spec-tokens` only when the profile intentionally disables that speculative-decoding configuration;
- adjust decode `replicas` only when capacity and measured service objectives require it; and
- review shared memory, resources, backend or kernel flags, and startup-probe budgets.

Different models and hardware profiles require different subsets of these changes. Treat the field names as a review checklist, not a universal context-length delta. Requalify readiness, memory use, latency, and throughput after the change.

### Plain multi-node workers

Install and enable a supported multi-node orchestration path before adding `multinode`; the operator rejects multi-node workloads when neither Grove nor LeaderWorkerSet (LWS) is available. Add `multinode.nodeCount: N` to each worker component that must span ordinary cluster nodes without ComputeDomain/DRA. `nodeCount` must be at least 2, and total allocated GPUs are `N ×` the main container's GPU request.

- **vLLM:** Keep node count, GPUs per Pod, model configuration, and the selected TP, PP, DP, or expert-parallel strategy coherent. With `multinode` set, the operator uses distributed TP/PP when TP × PP exceeds GPUs per Pod; that path selects multiprocessing or Ray. Otherwise, it handles Elastic EP through its Ray path or injects data-parallel coordination when TP × PP × DP exceeds GPUs per Pod. Only multiprocessing follower Pods receive the wait-for-leader init container.
- **SGLang:** Configure the TP, DP, and EP dimensions that apply to the selected model and strategy. The operator injects `--dist-init-addr`, `--nnodes`, and `--node-rank`; it does not size or validate the parallelism dimensions.
- **TensorRT-LLM:** Keep engine parallelism in the ConfigMap or engine overrides coherent with each role's multi-node and GPU shape. Prefill and decode may use different node counts. The operator computes MPI ranks as `nodeCount × GPUs per Pod`, manages SSH key material, wraps the leader with `mpirun`, and runs `sshd` on followers.

Do not hand-author operator-owned coordination flags, SSH/MPI wrappers, or wait-for-leader init containers; continue to author the framework runtime and parallelism arguments described above.

### Router mode

Router configuration belongs to the frontend and uses the same CLI and environment forms for every backend. `--router-mode kv` and `DYN_ROUTER_MODE=kv` are equivalent; use one form consistently.

- **Event-driven KV routing:** Event consumption is enabled by default. Every worker whose cache the router indexes must publish events. vLLM workers use `--enable-prefix-caching` with an enabled `--kv-events-config`; SGLang workers use `kv-events-config` with a non-null publisher; TensorRT-LLM workers use `--publish-kv-events`.
- **Prediction-based KV routing:** When the copied example does not configure worker event publication, add `--no-router-kv-events` until publishing is configured.
- **Round-robin routing:** This mode requires no KV-event configuration.

Set `--kv-cache-block-size` to the indexed workers' cache block or page size. It is a compatibility value, not an independent tuning knob. The current vLLM aggregate beta example selects KV routing without worker event publication; add `--no-router-kv-events` unless the recipe adds a compatible publisher. The current vLLM disaggregated beta example also lacks worker event publication but already passes `--router-kv-events`; replace that flag with `--no-router-kv-events` unless the recipe adds a compatible publisher.

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
| [SGLang ComputeDomain beta](sglang/agg/deploy-v1beta1-compute-domain.template.yaml) | [Qwen 3.8 ComputeDomain aggregate](../qwen3.8-2.4t-a95b-fp8/sglang/agg-gb300-chat/deploy.yaml) | ConfigMap, DRA chain, runtime bundle, node/GPU shape, offline operation, 7,200-second worker startup budget | 200Gi shared memory borrowed from the vLLM ComputeDomain source because the SGLang source did not size `/dev/shm`; reduced capability set | Source-derived except for shared-memory size; static checks; 200Gi requires SGLang-owner review |
| [TensorRT-LLM aggregate alpha](trtllm/agg/deploy-v1alpha1.template.yaml) | [GPT-OSS 120B aggregate](../gpt-oss-120b/trtllm/agg/deploy.yaml) | ConfigMap, runtime bundle, scale, GPU intent, 80Gi shared memory | Canonical names and a 60-minute worker startup budget; the source had no worker probe | Source-derived; static checks |
| [TensorRT-LLM disaggregated alpha](trtllm/disagg/deploy-v1alpha1.template.yaml) | [Nemotron 3 Super FP8 disaggregated](../nemotron-3-super-fp8/trtllm/disagg/deploy.yaml) | ConfigMaps, runtime and transfer bundle, scale, 16Gi shared memory, 6,000-second worker startup budget | Canonical names and standard probe shape | Source-derived; static checks |
| [TensorRT-LLM aggregate beta](trtllm/agg/deploy-v1beta1.template.yaml) | [Nemotron 3.5 Lightning aggregate](../nemotron-3.5-lightning/trtllm/agg-b200-bf16/deploy.yaml) | ConfigMap, runtime bundle except model identity, scale, GPU intent, 40Gi shared memory, 3,600-second worker startup budget | NVFP4 model substitution, matching parser flags, canonical names, online model access, and reduced cluster policy | Source-derived except for model identity; static checks; the NVFP4 substitution requires TensorRT-LLM owner review |
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
- Review canonical names, roles, ordering, and hook positions before contribution. An ordinary Kustomize build has no patch precondition that checks them. Only a guarded JSON Patch Component with explicit `test` operations can fail the build when a name, role, position, or replaced value differs from its expectation. This prevents a rename from silently creating an extra component through merge behavior.

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
