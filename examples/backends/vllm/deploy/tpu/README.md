<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Google TPU Deployment Examples

Hardware-specific deployment template for Google TPU using Kubernetes Dynamic Resource
Allocation (DRA), via the [kubernetes-sigs/dra-driver-google-tpu](https://github.com/kubernetes-sigs/dra-driver-google-tpu)
driver and the [vLLM TPU (tpu-inference)](https://docs.vllm.ai/projects/tpu/en/latest/) plugin.

Supported scope: vLLM, aggregated serving, single-host TPU slices. Disaggregated
prefill/decode, KVBM, and the GPU Memory Service are not available on TPU.

## Available Templates

| File | Pattern | Description |
|------|---------|-------------|
| `agg_tpu_dra.yaml` | Aggregated | Single worker requesting all TPU chips on a single-host slice |

## Prerequisites

1. **Any conformant Kubernetes cluster with TPU nodes**, version 1.34 or later so the
   `resource.k8s.io/v1` Dynamic Resource Allocation (DRA) API is GA and enabled by default.
   A plain kubeadm cluster on a TPU VM is sufficient — see
   [Running on a vanilla Kubernetes cluster](#running-on-a-vanilla-kubernetes-cluster).
2. **CDI enabled in containerd** — how the DRA driver injects `/dev/vfio` into the
   container. containerd 2.x enables it by default; 1.x needs `enable_cdi = true`.
   Managed clusters usually preconfigure this; a hand-rolled one may not.
3. **An unlimited memlock limit on every TPU node.** libtpu locks large regions of memory
   when it initialises the chip, and with the systemd default of 8 MB the engine aborts
   with `TPU initialization failed: Couldn't mmap: Resource temporarily unavailable`.
   The limit has to be raised on the container runtime: rlimits are per-process and
   inherited from the runtime, and Kubernetes has no pod-spec field for them. Adding
   `IPC_LOCK` is not an alternative either, because the runtime image runs as a non-root
   user and added capabilities never reach a non-root process's effective set.
   ```bash
   sudo mkdir -p /etc/systemd/system/containerd.service.d
   printf '[Service]\nLimitMEMLOCK=infinity\n' \
     | sudo tee /etc/systemd/system/containerd.service.d/memlock.conf
   sudo systemctl daemon-reload && sudo systemctl restart containerd
   ```
   Verify from inside a pod on that node: `ulimit -l` must print `unlimited`. Some
   distributions already set this. To set the limit per pod instead of node-wide, see
   containerd's [ulimit-adjuster NRI plugin](https://github.com/containerd/nri/tree/main/plugins/ulimit-adjuster),
   which applies `RLIMIT_MEMLOCK` from a pod annotation.
4. **[kubernetes-sigs/dra-driver-google-tpu](https://github.com/kubernetes-sigs/dra-driver-google-tpu)
   v0.2.0 or later**, registering the `tpu.google.com` `DeviceClass`. Released artifacts are
   `registry.k8s.io/dra-driver-google/dra-driver-google-tpu:v0.2.0` and the Helm chart
   `oci://registry.k8s.io/dra-driver-google/charts/dra-driver-google-tpu:0.2.0`; that repo's
   `demo/scripts/install-dra-driver.sh` wraps both.

   v0.2.0 and later discover the chips from the host, so no node labelling is required.
   Earlier releases need the accelerator labels applied to the node by hand.
5. **Custom TPU runtime image** built with the vLLM TPU (tpu-inference) plugin:
   ```bash
   python container/render.py --framework=vllm --device=tpu --target=runtime
   docker build -t nvcr.io/nvidia/ai-dynamo/vllm-runtime-tpu:1.4.0 \
     -f container/vllm-runtime-tpu-amd64-rendered.Dockerfile .
   ```
   See [container/README.md](../../../../../container/README.md) for complete build instructions.
   The tag must be a semantic version; the operator rejects arbitrary tags such as
   `:my-tag` when it derives component versions.
6. **HuggingFace token secret**:
   ```bash
   export HF_TOKEN=your_hf_token
   kubectl create secret generic hf-token-secret \
     --from-literal=HF_TOKEN=${HF_TOKEN} \
     -n ${NAMESPACE}
   ```

## Deploy

```bash
# Apply template (includes ResourceClaimTemplate)
kubectl apply -f tpu/agg_tpu_dra.yaml -n $NAMESPACE

# Verify TPU allocation
kubectl get resourceclaim -n $NAMESPACE
kubectl get resourceslices

# Check deployment status
kubectl get dynamographdeployment -n $NAMESPACE
kubectl get pods -n $NAMESPACE
```

## Testing

```bash
# Port forward to frontend
kubectl port-forward deployment/vllm-agg-tpu-dra-frontend 8000:8000 -n $NAMESPACE

# Test inference
curl localhost:8000/v1/models
curl localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen3-0.6B","prompt":"Hello","max_tokens":20}'
```

For a non-Kubernetes smoke test (single Cloud TPU VM, plain `docker run`), see the "TPU variant"
in [container/README.md](../../../../../container/README.md)'s Production Workflow section.

## Running on a vanilla Kubernetes cluster

The steps above assume a cluster where the `tpu.google.com` `DeviceClass` and
`ResourceSlice`s already exist. This section builds one from scratch with upstream
Kubernetes and the OSS DRA driver.

### 1. Bring up a cluster with TPU nodes and the DRA driver
On GCE, the DRA driver repo ships a script that does all of it — control-plane VM, a
TPU VM joined as a worker, and the driver itself:
```bash
git clone --branch v0.2.0 https://github.com/kubernetes-sigs/dra-driver-google-tpu.git
cd dra-driver-google-tpu
PROJECT=my-project ZONE=us-central1-a ./demo/clusters/gce/create-kubeadm-tpu-cluster.sh
```
See [demo/clusters/gce](https://github.com/kubernetes-sigs/dra-driver-google-tpu/tree/main/demo/clusters/gce)
for quota checks, machine-type overrides and teardown.

That script leaves containerd at the systemd default 8 MB memlock limit, so raise it on
the TPU node before deploying (see prerequisite 3).

Any conformant installer works instead; the driver's
[README](https://github.com/kubernetes-sigs/dra-driver-google-tpu#readme) covers the
other supported paths, and `demo/scripts/install-dra-driver.sh` installs the driver into
a cluster you already have. The cluster needs:

- **Kubernetes 1.34 or later**, for the GA `resource.k8s.io/v1` API (see
  [install-dynamo.md](../../../../../docs/fern/pages/kubernetes/installation/install-dynamo.md)).
- **containerd with CDI enabled**, which is how the driver injects the TPU device nodes.
  containerd 2.x enables it by default; 1.x needs `enable_cdi = true`.
- **An unlimited memlock limit**, set on the container runtime.
- **Nodes with TPU chips attached** (`ls /dev | grep -E 'accel|vfio'`).
- **A container registry the nodes can pull from**, for the image built in step 3.

### 2. Verify the driver sees the TPU chips
```bash
kubectl get pod -n dra-driver-google-tpu
kubectl get deviceclass tpu.google.com
kubectl get resourceslice -o jsonpath='{range .items[*]}{.spec.nodeName} {.spec.driver}{"\n"}{range .spec.devices[*]}  {.name} {.attributes}{"\n"}{end}{end}'
```

**Expected output**

```bash
kubectl get pod -n dra-driver-google-tpu
NAME                                        READY   STATUS    RESTARTS   AGE
dra-driver-google-tpu-kubeletplugin-mqlxz   3/3     Running   0          29m
dra-driver-google-tpu-kubeletplugin-vcgmp   3/3     Running   0          29m

kubectl get deviceclass tpu.google.com
NAME             AGE
tpu.google.com   29m

kubectl get resourceslice -o jsonpath=...
tpu-dra-tpu tpu.google.com
  0 {"accelerator":{"string":"tpu-v5-lite-podslice"},"brand":{"string":"Google"},"chipCount":{"int":1},"index":{"int":0},"topology":{"string":"1x1"},"tpuGen":{"string":"v5litepod"},"uuid":{"string":"tpu-585695ce-c265-152a-e04a-88b343209584"}}
```

The plugin runs on every node but publishes a `ResourceSlice` only where TPUs exist, so
one slice for a single TPU node is expected. The `uuid` and `tpuGen` values are read from
the hardware.

### 3. Build and publish the Dynamo vLLM+TPU runtime image
Push to any registry the cluster can pull from:
```bash
cd /path/to/dynamo
python3 container/render.py --framework=vllm --device=tpu --target=runtime
docker build -t gcr.io/$PROJECT/dynamo-vllm-tpu:1.4.0 \
  -f container/vllm-runtime-tpu-amd64-rendered.Dockerfile .
gcloud auth configure-docker gcr.io --quiet
docker push gcr.io/$PROJECT/dynamo-vllm-tpu:1.4.0
```
Verify the image before deploying:
```bash
docker run --rm --entrypoint python3 gcr.io/$PROJECT/dynamo-vllm-tpu:1.4.0 \
  -c "from vllm import LLM; from vllm.platforms import current_platform; import dynamo.vllm; print(current_platform.device_name)"
# expect: tpu
```

### 4. Install the Dynamo Platform
An aggregated deployment needs only the operator and etcd. Persistence is disabled here
because a kubeadm cluster has no default `StorageClass` for the PVC to bind to:
```bash
export NAMESPACE=dynamo-system
helm fetch https://helm.ngc.nvidia.com/nvidia/ai-dynamo/charts/dynamo-platform-<RELEASE_VERSION>.tgz
helm install dynamo-platform dynamo-platform-<RELEASE_VERSION>.tgz \
  --namespace $NAMESPACE --create-namespace \
  --set global.etcd.install=true \
  --set etcd.persistence.enabled=false
kubectl get pods -n $NAMESPACE
```

**Expected output**

```bash
kubectl get pods -n dynamo-system
NAME                                                              READY   STATUS    RESTARTS   AGE
dynamo-platform-dynamo-operator-controller-manager-76f6d7dwbbsd   1/1     Running   0          2m
dynamo-platform-etcd-0                                            1/1     Running   0          2m
```

### 5. Secrets, then deploy
```bash
export NS=dynamo-tpu
kubectl create namespace $NS
kubectl create secret generic hf-token-secret --from-literal=HF_TOKEN=$HF_TOKEN -n $NS

# Attaching the pull secret to the namespace's default ServiceAccount covers every
# pod the operator creates, so the manifest needs no imagePullSecrets.
kubectl create secret docker-registry gcr-pull \
  --docker-server=gcr.io --docker-username=oauth2accesstoken \
  --docker-password="$(gcloud auth print-access-token)" -n $NS
kubectl patch serviceaccount default -n $NS \
  -p '{"imagePullSecrets":[{"name":"gcr-pull"}]}'

sed "s#image: nvcr.io/nvidia/ai-dynamo/vllm-runtime-tpu:.*#image: gcr.io/$PROJECT/dynamo-vllm-tpu:1.4.0#; \
     s#image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:.*#image: gcr.io/$PROJECT/dynamo-vllm-tpu:1.4.0#" \
  agg_tpu_dra.yaml | kubectl apply -n $NS -f -
```
The access token expires after about an hour; recreate the secret if a later pull fails.

### 6. Verify allocation and scheduling
```bash
kubectl get resourceclaim -n $NS     # status.allocation should name your real accel device
kubectl get pods -n $NS -o wide
kubectl logs -n $NS -l nvidia.com/dynamo-component-type=worker -f
```

**Expected output**

```bash
kubectl get pods -n dynamo-tpu -o wide
NAME                                                          READY   STATUS    NODE
vllm-agg-tpu-dra-frontend-5b7797df9d-nbrdt                    1/1     Running   tpu-dra-cpu
vllm-agg-tpu-dra-vllmdecodeworker-2aed0841-78fc5b7774-bbcrn   1/1     Running   tpu-dra-tpu

kubectl get resourceclaim -n dynamo-tpu
NAME                                                           STATE                AGE
vllm-agg-tpu-dra-vllmdecodeworker-2aed0841-78fc5b77-tpuqzh7x   allocated,reserved   8m
```

The worker lands on the TPU node and the frontend on the CPU node. In the worker log,
`Registered base model 'Qwen/Qwen3-0.6B' MDC` means vLLM finished loading on the TPU
(first start includes XLA compilation, so allow several minutes).

A worker stuck in `Pending` with `cannot allocate all claims` means the chip is still
held by another pod — `allocationMode: All` is exclusive, so only one claimant at a time.

### 7. Smoke test
```bash
kubectl port-forward -n dynamo-tpu svc/vllm-agg-tpu-dra-frontend 8000:8000 &
curl localhost:8000/v1/models
curl localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen3-0.6B","prompt":"The capital of France is","max_tokens":20,"temperature":0}'
```

**Expected output**

```bash
curl localhost:8000/v1/models
{"object":"list","data":[{"id":"Qwen/Qwen3-0.6B","object":"model","owned_by":"nvidia","context_window":2048}]}

curl localhost:8000/v1/completions ...
{"id":"cmpl-6e337253-...","choices":[{"text":" Paris. The capital of France is also the
 capital of the Republic of France. The capital of France","index":0,"finish_reason":"length"}],
 "model":"Qwen/Qwen3-0.6B","object":"text_completion",
 "usage":{"prompt_tokens":5,"completion_tokens":20,"total_tokens":25}}
```

### Troubleshooting
- Pod stuck `Pending` → `kubectl describe pod` for DRA scheduling errors; re-check
  `resourceslice`/`deviceclass` from step 2.
- Pod `CrashLoopBackOff` on the vLLM container → check logs first for the
  tpu-inference/JAX device-init error; confirm `/dev/accel*` or `/dev/vfio/*` actually
  got injected (`kubectl exec ... -- ls /dev`).
- `TPU initialization failed: Couldn't mmap: Resource temporarily unavailable` → the host
  memlock limit is too low. Confirm with `ulimit -l` inside a pod on that node and fix the
  container runtime limit, not the pod spec (see prerequisite 3).

## Automated test coverage

[tests/serve/test_vllm_tpu.py](../../../../../tests/serve/test_vllm_tpu.py) runs the same
aggregated scenario (chat completion, text completion, and a Prometheus metrics sanity
check) through Dynamo's `EngineConfig`/`run_serve_deployment` test framework, using
[examples/backends/vllm/launch/tpu/agg_tpu.sh](../../../launch/tpu/agg_tpu.sh) as the
bare-metal (non-Kubernetes) launch script. It's marked `tpu_1` (see `pyproject.toml`) and
only runs where a TPU-attached test runner is available — there is no CI runner for it
yet, so it must currently be run manually on real TPU hardware:
```bash
python3 -m pytest tests/serve/test_vllm_tpu.py -v
```

## Further Reading

- [Main Deployment README](../README.md) - Overview of all deployment patterns
- [kubernetes-sigs/dra-driver-google-tpu](https://github.com/kubernetes-sigs/dra-driver-google-tpu)
- [vLLM TPU documentation](https://docs.vllm.ai/projects/tpu/en/latest/)
