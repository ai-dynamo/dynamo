# CSP fabric profiles

One self-contained YAML profile per cloud provider that describes how to put a generic
Dynamo `DynamoGraphDeployment`'s disaggregated workers onto that provider's RDMA fabric.

These are **agent-materialized specs**, not `kustomize` artifacts. An agent reads a
profile plus your base recipe, resolves the parameters, and adds the fabric layer to the
worker services. **Step-by-step procedure: [AGENTS.md](AGENTS.md).** (We started with kustomize Components but dropped them: kustomize is
template-free and schema-blind, so it can't derive the EFA count from the GPU count,
target workers by role, or reach the backend selection that lives in a vLLM arg string /
a TRT-LLM ConfigMap. One agent-applied format does all of it.)

| Profile | Fabric | KV backend | NIC pin |
|---|---|---|---|
| [`aws-efa`](aws-efa.yaml)   | AWS EFA              | **LIBFABRIC** | n/a (EFA provider) |
| [`aks-ib`](aks-ib.yaml)     | Azure AKS InfiniBand | UCX           | required (`UCX_NET_DEVICES`) |
| [`gke-roce`](gke-roce.yaml) | GKE A4X RoCEv2       | UCX           | auto-probe |
| [`nebius-ib`](nebius-ib.yaml) | Nebius InfiniBand  | UCX           | auto-probe |
| [`nscale-ib`](nscale-ib.yaml) | Nscale InfiniBand  | UCX           | required (`UCX_NET_DEVICES`) |

## Schema

| Field | Meaning |
|---|---|
| `target` | which services to patch — by `subComponentType: prefill\|decode` (any worker name) |
| `image` | a **rule** for the worker image (e.g. `{base_tag}-efa-{arch}`), plus `verify: registry` |
| `device` | entries added to `resources.{limits,requests}.custom` |
| `annotations` | entries added to `extraPodMetadata.annotations` (GKE only) |
| `securityContext` | `privileged` + `IPC_LOCK`/`SYS_PTRACE` (VRAM registration) |
| `env` | appended to each worker's `env` — the engine-neutral fabric transport layer |
| `kv_backend` | the KV-transport backend value (`UCX` or `LIBFABRIC`) |
| `backend` | per-engine wiring to *select* `kv_backend` — `{}` when none is needed |
| `skip` | things the profile deliberately does NOT touch (stay in the base recipe) |

## Conventions

- **`{{ ... }}` / `@param`** mark values that must be **materialized**, never copied blind:
  - `aws-efa`: `vpc.amazonaws.com/efa = gpus_per_worker * 4` (read GPU count from the base);
    image `arch` ∈ `amd64` (H100/B200) / `arm64` (GB200).
  - `aks-ib` / `nscale-ib`: `UCX_NET_DEVICES` = this cluster's compute-fabric NICs (`ibv_devinfo`).
  - `gke-roce`: the RDMA network names + count for your A4X pool (in both `device` and `annotations`).
- **`verify: registry`** — confirm a derived image tag actually exists before using it (the
  `-efa` suffix isn't uniform across releases).
- **`rdma/ib: "1"`** is a device-plugin slot count (grants all HCAs), **not** GPU-derived — leave it `"1"`.

## Backend is special only for EFA

`kv_backend` is `UCX` for the four IB/RoCE fabrics, and UCX needs **no** selection:
vLLM defaults to it, and every TRT-LLM disagg base already sets
`cache_transceiver_config.backend: UCX`. So their `backend:` block is empty. Only
`aws-efa` uses a non-default backend (`LIBFABRIC`), so it's the **only** profile with a
populated `backend:` block — and the engine entries there (vLLM env+arg, TRT-LLM
ConfigMap) are the only backend-specific content in the whole set:

```
            vllm        trtllm      sglang
aws-efa     LIBFABRIC   LIBFABRIC   LIBFABRIC     <- only row with backend wiring
aks-ib        —           —           —
gke-roce      —           —           —
nebius-ib     —           —           —
nscale-ib     —           —           —
```

## What a profile never includes (stays in the base recipe)

- the base image choice (the profile only applies an image *rule*),
- the storage PVC name + mounts,
- volumes/host mounts (GKE GIB libs; the IB recipes' `/dev/infiniband` + `/dev/shm`).
