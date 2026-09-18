# Materializing a fabric profile (agent guide)

The `*.yaml` here are **specs, not kustomize**. To put a base `DynamoGraphDeployment` on a
CSP's RDMA fabric, materialize the profile onto it:

## Steps
1. Pick `<provider>.yaml` (aws-efa | aks-ib | gke-roce | nebius-ib | nscale-ib).
2. Target the base's **worker services by `subComponentType: prefill|decode`** (any name).
3. For each worker, apply the profile fields:
   - `device` → `resources.{limits,requests}.custom`
   - `securityContext` → merge into `mainContainer.securityContext`
   - `env` → **append** to `mainContainer.env` (don't clobber existing)
   - `annotations` (gke-roce) → `extraPodMetadata.annotations`
   - `image` → apply the `rule`, then **verify the resulting tag exists in the registry**
   - `backend` → apply **only the entry matching the base's engine** (`vllm`|`trtllm`|`sglang`).
     Only `aws-efa` has a non-empty `backend` (LIBFABRIC); the UCX fabrics inherit UCX (empty).
4. Resolve every `{{...}}`/`@param` from the base + target cluster — **never copy the default blind**:
   - EFA count = `gpus_per_worker * 4` (p5.48xlarge); read GPUs from `limits.gpu`
   - image `arch` = `amd64` (H100/B200) | `arm64` (GB200), from the base `nodeSelector`/cluster
   - `UCX_NET_DEVICES` (aks/nscale) = compute-fabric NICs from `ibv_devinfo`
   - gke-roce rdma networks = the A4X pool's network names (`device` + `annotations` must match)
5. Leave `skip` items in the base recipe (base image, storage PVC, volume mounts).
6. Deploy, then **verify libfabric actually engaged** — UCX is a silent ~100s-TTFT fallback, not an error.

## Verify (startup log, per engine)
- vLLM / SGLang / KVBM → `NIXL ... Backend LIBFABRIC was instantiated`
- TRT-LLM → `NixlTransferAgent ... using NIXL backend: LIBFABRIC`
  (ignore the generic `Backend UCX was instantiated` line — it's a secondary agent, not the KV transfer)

## Gotchas
- Verify the EFA image tag exists before deploying (the `-efa[-arch]` suffix isn't uniform across releases).
- DGD name + worker service name must be **≤ 45 chars** (admission webhook).
- The storage PVC name (e.g. `model-cache` vs `shared-model-cache`) is cluster-specific and lives in the base.

Canonical reference: `docs/kubernetes/cloud-providers/eks/efa.md` (per-engine LIBFABRIC selectors, FI_* env, EFA image build).
