# CSP RDMA addons (Kustomize Components)

Composable [Kustomize Components](https://kubectl.docs.kubernetes.io/guides/config_management/components/)
that add a cloud provider's RDMA KV-transfer layer to a generic Dynamo
`DynamoGraphDeployment`, so you don't have to hand-copy the provider-specific env,
device requests, and securityContext into every recipe.

| Component | Fabric | KV transport | NIC pin |
|---|---|---|---|
| [`aws-efa`](aws-efa)   | AWS EFA              | NIXL libfabric | n/a (EFA provider) |
| [`aks-ib`](aks-ib)     | Azure AKS InfiniBand | UCX over IB    | **required** (`UCX_NET_DEVICES`, cluster-specific) |
| [`gke-roce`](gke-roce) | GKE A4X RoCEv2       | UCX over RoCE  | auto-probe (no pin) |
| [`nebius-ib`](nebius-ib) | Nebius InfiniBand  | UCX over IB    | auto-probe (no side fabric) |
| [`nscale-ib`](nscale-ib) | Nscale InfiniBand  | UCX over IB    | **required** (`UCX_NET_DEVICES`, cluster-specific) |

## Usage

Wrap your DGD in a kustomization and pull in one component:

```yaml
# my-deploy/kustomization.yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
resources:
  - my-dgd.yaml
components:
  - <path-or-git-url>/recipes/addons/kustomize/aks-ib
```

```bash
kubectl kustomize my-deploy | kubectl apply -f -
```

## What a component patches

Onto the `VllmPrefillWorker` and `VllmDecodeWorker` services:

- **RDMA device request** — the provider's `resources.limits/requests.custom` entry
  (`vpc.amazonaws.com/efa`, `rdma/ib`, or the GKE `networking.gke.io.networks/*`).
- **Transport env** — appended to the worker's existing `env` (preserved), e.g.
  `UCX_NET_DEVICES`, `FI_*`, `NCCL_SOCKET_IFNAME`.
- **`securityContext`** — `privileged: true` + `IPC_LOCK`, `SYS_PTRACE` (required to
  register CUDA VRAM with the HCA/NIC). Created if absent, merged if present.
- **(GKE only)** the `networking.gke.io/interfaces` pod annotation the GKE webhook
  reads to attach the RDMA NICs.

The env is a **JSON6902 append** (your existing env is kept); the resource and
securityContext are a **strategic merge** (your `gpu` request is kept).

## What is NOT included (set these in your base DGD)

- **An RDMA-capable image** — the component does not change the image.
- **The engine's NIXL backend** — vLLM's UCX backend is the default; EFA needs
  `--kv-transfer-config '… "kv_connector_extra_config":{"backends":["LIBFABRIC"]}'`.
- **Volume mounts** — the GKE GIB/driver host mounts (`/usr/local/gib`,
  `/usr/local/nvidia`) and the IB-hostPath recipes' `/dev/infiniband` + `/dev/shm`
  are lists; Kustomize can't safely inject list items into an arbitrary base, so add
  them to your DGD directly.

## Adjust for your DGD

- **Service names.** The patches target `VllmPrefillWorker` / `VllmDecodeWorker`.
  Kustomize cannot patch a CRD's nested per-service fields without naming the service,
  so if your DGD uses different worker names (the TRT-LLM recipes use `prefill` /
  `decode`), copy the component dir and retarget:
  ```bash
  sed -i 's/VllmPrefillWorker/prefill/g; s/VllmDecodeWorker/decode/g' kustomization.yaml
  ```
- **Cluster-specific values.** `aks-ib` / `nscale-ib` pin `UCX_NET_DEVICES` to a
  specific compute-fabric NIC list — derive yours from `ibv_devinfo`. `gke-roce`'s
  RDMA network names and `aws-efa`'s EFA count likewise match a specific instance type.
- **Apply to a base without preexisting RDMA config.** These layer cleanly onto a
  vanilla DGD. If the base already sets `UCX_NET_DEVICES` / `UCX_TLS` / an RDMA
  resource, the component's appended env still wins (Kubernetes takes the last
  duplicate) and its resource value overrides — but you'll see duplicate env entries,
  and a base `UCX_TLS` (which these recipes intentionally avoid) is left in place.
  Remove the base's RDMA env first for a clean result.

## Verified

`kubectl kustomize` builds cleanly for every component against the generic
`VllmPrefillWorker`/`VllmDecodeWorker` vLLM disagg recipes (`qwen3-32b-fp8`,
`llama-3-70b`, `deepseek-v4`): the RDMA resource merges (gpu preserved), the transport
env appends (existing env preserved), and the privileged securityContext is
created/merged. `aws-efa` also verified against the `qwen3-235b-a22b-fp8` TRT-LLM DGD
after the `prefill`/`decode` retarget.
