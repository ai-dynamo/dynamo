# AWS EFA addon (Kustomize Component)

Adds the **EFA / libfabric KV-transfer layer** to the disaggregated workers of any
Dynamo `DynamoGraphDeployment`, so you don't have to hand-copy the EFA env block
into every recipe.

## What it patches

Onto the `VllmPrefillWorker` and `VllmDecodeWorker` services:

| Piece | Value |
|---|---|
| EFA device | `vpc.amazonaws.com/efa: "16"` (limits + requests) |
| libfabric provider | `FI_PROVIDER=efa`, `FI_EFA_USE_DEVICE_RDMA=1`, `FI_EFA_ENABLE_SHM_TRANSFER=0`, `FI_LOG_LEVEL=warn` |
| NIXL backend | `DYN_KVBM_NIXL_BACKEND_LIBFABRIC=true`, `DYN_KVBM_NIXL_BACKEND_UCX=false` |
| securityContext | `privileged: true` + `IPC_LOCK`, `SYS_PTRACE` (required for `fi_mr_reg` on CUDA VRAM) |

The env is **appended** (your existing env is preserved); the EFA resource is **merged**
into `resources` (your `gpu` request is preserved).

## Usage

```yaml
# my-deploy/kustomization.yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
resources:
  - my-dgd.yaml
components:
  - <path-or-git-url>/recipes/addons/kustomize/aws-efa
```

```bash
kubectl kustomize my-deploy | kubectl apply -f -
```

## Adjust for your DGD

- **Service names.** The patches target `VllmPrefillWorker` and `VllmDecodeWorker`
  — the worker service names used by the Dynamo vLLM disagg recipes. Kustomize
  cannot patch a CRD's nested per-service fields without naming the service, so if
  your DGD uses different worker names (e.g. the TRT-LLM recipes use `prefill` /
  `decode`), copy this dir and retarget:
  ```bash
  sed -i 's/VllmPrefillWorker/prefill/g; s/VllmDecodeWorker/decode/g' kustomization.yaml
  ```
- **EFA count.** `16` = the NUMA-local EFA NICs for a TP=4 worker on `p5.48xlarge`
  (32 EFA / 8 GPU = 4 per GPU). Match it to your worker's GPU count.

## NOT included (engine-specific — set these in your base DGD)

- An **EFA-capable image** (libfabric + the EFA provider baked in). This component
  does not change the image.
- The **engine's NIXL backend**:
  - vLLM: `--kv-transfer-config '{"kv_connector":"NixlConnector",…,"kv_connector_extra_config":{"backends":["LIBFABRIC"]}}'`
  - TRT-LLM: `cache_transceiver_config.backend` in the engine config.

## Verified

`kubectl kustomize` applies cleanly — EFA env appended (existing env preserved), EFA
resource merged (gpu preserved), privileged securityContext created/merged — to the
`VllmPrefillWorker`/`VllmDecodeWorker` vLLM disagg DGDs (`qwen3-32b-fp8`, `llama-3-70b`,
`deepseek-v4`), and to the `qwen3-235b-a22b-fp8` TRT-LLM DGD after the `prefill`/`decode`
retarget above.
