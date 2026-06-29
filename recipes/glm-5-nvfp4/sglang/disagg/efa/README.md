# GLM-5 NVFP4 — Disaggregated Prefill/Decode on GB200 over AWS EFA

Serves [nvidia/GLM-5-NVFP4](https://huggingface.co/nvidia/GLM-5-NVFP4) using SGLang with disaggregated prefill/decode, EAGLE speculative decoding, and KV cache transfer over AWS Elastic Fabric Adapter (EFA) RDMA via NIXL's LIBFABRIC backend.

Compared to the non-EFA recipe (UCX/RoCE), this variant:

- Carries inter-node KV transfer over EFA via NIXL LIBFABRIC backend.
- Bakes [ofiwg/libfabric#12019](https://github.com/ofiwg/libfabric/pull/12019) into the image so `fi_mr_reg` on CUDA VRAM succeeds on GB200's 64K-page arm64 kernel.
- Runs containers `privileged: true` so `fi_mr_reg` can pin VRAM for RDMA.
- Sets `SGLANG_DISAGGREGATION_NIXL_BACKEND=LIBFABRIC` — without this, SGLang defaults to UCX, which silently falls back to TCP on kernel ≥ 6.8.

Please see the [Dynamo on EFA](../../../../../docs/kubernetes/cloud-providers/eks/efa.md) for more on EFA.

## Topology

Identical to the non-EFA recipe.


| Role    | Nodes | GPUs/node | EFA NICs/node | Total GPUs | Parallelism        |
| ------- | ----- | --------- | ------------- | ---------- | ------------------ |
| Decode  | 4     | 4         | 4             | 16         | TP16 / DP16 / EP16 |
| Prefill | 1     | 4         | 4             | 4          | TP4                |


## Prerequisites

- A Kubernetes cluster with **5 × p6e-gb200.36xlarge** nodes (or equivalent GB200 in an MNNVL domain) — each node provides 4× GB200 + 4× EFA NICs.
- AWS EFA driver ≥ 3.0.0g on the nodes (default on modern AWS EKS AMIs).
- Kernel ≥ 5.12 (modern AWS EKS AMIs ship kernel 6.14 — fine). On older kernels, the host also needs the `efa_nv_peermem` module loaded; on ≥ 5.12 the dmabuf path is the default and peermem is not required.
- A Kubernetes cluster with the Dynamo Operator installed.
- The NVIDIA `ComputeDomain` operator (for the MNNVL ResourceClaim used here).
- Shared NFS PVC for model weights (same as the non-EFA recipe).

The libfabric is built into the image — no cluster-side DaemonSet is required.

## Step 1: Build the Container

```bash
docker buildx build \
  --platform linux/arm64 \
  --build-arg ARCH=arm64 \
  -t <your-registry>/sglang-dynamo-glm5-efa:latest \
  -f recipes/glm-5-nvfp4/sglang/disagg/efa/Dockerfile.efa \
  --push .
```

Quick sanity check on the resulting image:

```bash
docker run --rm <your-registry>/sglang-dynamo-glm5-efa:latest \
  bash -c '
    /opt/amazon/efa/bin/fi_info --version &&
    ldd /opt/amazon/efa/lib/libfabric.so.1 | grep -i cuda &&
    grep -F "efa_mr_is_cuda(efa_mr)" \
      /opt/amazon/efa/share/doc/libfabric/efa_mr.c.diff 2>/dev/null ||
      echo "(patch verified at build time via Dockerfile RUN grep)"'
```

## Step 2: Download the Model

Same as the non-EFA recipe:

```bash
kubectl apply -f recipes/glm-5-nvfp4/model-cache/model-cache.yaml

kubectl create secret generic hf-token-secret \
  --from-literal=HF_TOKEN=<your-hf-token>

kubectl apply -f recipes/glm-5-nvfp4/model-cache/model-download.yaml
kubectl wait --for=condition=complete job/model-download --timeout=3600s
```

## Step 3: Deploy

Edit `sglang/disagg/efa/deploy.yaml` and replace all `<placeholder>` values:

- `<your-namespace>` — your Kubernetes namespace
- `<your-registry>/sglang-dynamo-glm5-efa:latest` — your built container image

```bash
kubectl apply -f recipes/glm-5-nvfp4/sglang/disagg/efa/deploy.yaml
```

Monitor startup:

```bash
kubectl get pods -n <your-namespace> -l app.kubernetes.io/part-of=glm5-sglang-efa -w
```

## Step 4: Verify EFA Engaged

Three quick checks confirm the LIBFABRIC backend is actually carrying KV traffic (not silent TCP fallback):

```bash
NS=<your-namespace>
POD=$(kubectl -n $NS get pods -l nvidia.com/dynamo-component=decode \
        -o jsonpath='{.items[0].metadata.name}')

# 1. NIXL selected LIBFABRIC at startup.
kubectl -n $NS logs $POD | grep -i 'NIXL.*backend.*LIBFABRIC' | head -1
# Expected: "NIXL INFO Backend LIBFABRIC was instantiated" or similar.
# WRONG:    "NIXL agent uses UCX backend".

# 2. libplugin_LIBFABRIC.so is actually executing (not just dlopen'd).
kubectl -n $NS exec $POD -- bash -c '
  grep libplugin_LIBFABRIC /proc/$(pgrep -f sglang | head -1)/maps |
    grep "r-xp"' | head -3
# Expected: at least one line ending in "r-xp" (executable mapping).

# 3. NIXL transfer metrics show no failures.
kubectl -n $NS exec $POD -- curl -s localhost:19090/metrics |
  grep -E 'nixl_(bytes_transferred|num_failed)'
# Expected: nixl_num_failed_transfers_total stays 0; bytes_transferred grows.
```

If check 1 shows UCX, the most likely cause is `SGLANG_DISAGGREGATION_NIXL_BACKEND=LIBFABRIC` not being applied (verify the env block in the running pod). If check 2 is empty, `LD_LIBRARY_PATH` ordering is wrong — `/opt/amazon/efa/lib` must come before any other libfabric on the path.

## Step 5: Test

```bash
kubectl port-forward svc/glm5-sglang-efa-frontend 8000:8000 -n <your-namespace> &
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"nvidia/GLM-5-NVFP4","messages":[{"role":"user","content":"Hello!"}],"max_tokens":128}'
```

## Step 6: Benchmark

Edit `sglang/disagg/efa/perf.yaml` to set your namespace and PVC name, then run:

```bash
kubectl apply -f recipes/glm-5-nvfp4/sglang/disagg/efa/perf.yaml
kubectl logs -f -l job-name=glm5-disagg-efa-bench -n <your-namespace>
```

Workload shape is identical to the non-EFA `../perf.yaml`: ISL=1000, OSL=8192, concurrency=512 (32 req/GPU × 16 decode GPUs).

## Performance (ISL=1k, OSL=8k, concurrency=512)

Measured on 5 × p6e-gb200.36xlarge, EFA driver 3.0.0g, kernel 6.14.0-1018-aws-64k. Benchmark workload identical to the non-EFA recipe's `perf.yaml` (1,536 requests at concurrency 512 = 32 req/GPU × 16 decode GPUs).


| Metric                     | EFA (this recipe) | Non-EFA (UCX/RoCE) baseline |
| -------------------------- | ----------------- | --------------------------- |
| Output token throughput    | **19,131 tok/s**  | ~19,000 tok/s               |
| Total token throughput     | 21,468 tok/s      | —                           |
| TTFT p50                   | **621 ms**        | ~850 ms                     |
| TTFT avg                   | 2,786 ms          | —                           |
| ITL avg                    | **24.5 ms/token** | ~24 ms/token                |
| Output tokens / user / sec | **41.0**          | ~41                         |
| Request count              | 1,536             | —                           |
| Benchmark duration         | 657 s             | —                           |


Baseline numbers reproduced from [../README.md](../../../README.md). EFA achieves parity on all four published metrics. TTFT p50 is lower (better) on EFA, though TTFT avg is higher with a long tail (p99 ≈ 12.5 s).

## Long-context performance (ISL=20k, OSL=2k, concurrency=64)

The benefit of the LIBFABRIC backend grows with per-request KV cache size. Running the same recipe with `SGLANG_DISAGGREGATION_NIXL_BACKEND=LIBFABRIC` set vs. unset (NIXL defaults to UCX) on the same 5 × p6e-gb200.36xlarge hardware:


| Metric                  | LIBFABRIC (this recipe) | UCX (default) |
| ----------------------- | ----------------------- | ------------- |
| Output token throughput | **2,023 tok/s**         | 1,454 tok/s   |
| TTFT p50                | **20,273 ms**           | 46,369 ms     |
| TTFT avg                | **20,452 ms**           | 41,746 ms     |
| ITL avg                 | 17.56 ms                | 17.20 ms      |
| Benchmark duration      | **194 s**               | 270 s         |
| Request count           | 192                     | 192           |


With 20× larger per-request KV cache, LIBFABRIC is **39% higher throughput** and **56% lower TTFT p50** than UCX. ITL is essentially unchanged because steady-state decoding never touches the KV transfer path — only the prefill→decode KV hand-off does.


## References

- [ofiwg/libfabric#12019](https://github.com/ofiwg/libfabric/pull/12019) — `efa_mr_is_cuda` patch
- [README.md](../../../README.md) — non-EFA variant of this recipe
