# Troubleshooting

Quick index of the failures users most commonly hit, with the fix and a pointer to the engineering follow-up where one is tracked.

## Contents

- [Image pull failures](#image-pull-failures)
- [Dynamo CRDs missing](#dynamo-crds-missing)
- [PVC stuck Pending](#pvc-stuck-pending)
- [Driver mismatch — cryptic PyTorch error](#driver-mismatch--cryptic-pytorch-error)
- [`/v1/health/ready` returns 404](#v1healthready-returns-404)
- [CUDA dev headers missing for source build](#cuda-dev-headers-missing-for-source-build)
- [HF_TOKEN / gated model access](#hf_token--gated-model-access)
- [Multimodal streaming fails on vLLM](#multimodal-streaming-fails-on-vllm)
- [DGDR stuck in `Failed`](#dgdr-stuck-in-failed)
- [`disagg.sh` 503 on first requests](#disaggsh-503-on-first-requests)

---

## Image pull failures

Symptom: `ImagePullBackOff` / `ErrImagePull` on a Dynamo pod.

Cause: the image tag doesn't exist in NGC, or the cluster lacks a pull secret for NGC.

Fix:

1. Confirm the pinned tag exists — [release-artifacts.md](reference/release-artifacts.md) lists the canonical `:1.0.1` tag.
2. If using a top-of-tree Kimi-k2.5 recipe, you must build and push your own image. See [`recipes/kimi-k2.5/README.md`](../recipes/kimi-k2.5/README.md).
3. NGC pull secret: `kubectl create secret docker-registry ngc-cred --docker-server=nvcr.io --docker-username=\$oauthtoken --docker-password=$NGC_API_KEY`.

## Dynamo CRDs missing

Symptom: `kubectl apply -f deploy.yaml` returns `no matches for kind "DynamoGraphDeployment"`.

Fix: install the Dynamo operator first. `kubectl get crd | grep dynamo` should list three CRDs (`dynamocomponentdeployments`, `dynamographdeployments`, `dynamographdeploymentrequests`). See [Dynamo Operator](kubernetes/dynamo-operator.md).

## PVC stuck Pending

Symptom: `kubectl get pvc` shows `Pending`. `kubectl describe pvc <name>` shows "no persistent volumes available" or "provisioner not found."

Cause: recipes default to `storageClassName: standard`, which doesn't exist on every cluster.

Fix: pick the right class for your environment. See the [storage-class table in `recipes/README.md`](../recipes/README.md#storage-class).

## Driver mismatch — cryptic PyTorch error

Symptom:

```
RuntimeError: The NVIDIA driver on your system is too old (found version 570). Please update your GPU driver ...
```

Cause: the container's CUDA version needs a newer driver than the host ships.

Fix: check the minimum driver for your tag in the [TRT-LLM container/driver matrix](backends/trtllm/README.md#container--driver-matrix) and upgrade, or pull a lower-CUDA variant. The validation error message itself is being improved as a separate engineering follow-up.

## `/v1/health/ready` returns 404

Symptom: `curl localhost:8000/v1/health/ready` → `404 Not Found`.

Cause: the `/v1/health/ready` route is not yet wired. `/health` does work.

Workaround: use `/health` for liveness probes until `/v1/health/ready` ships. Tracked as a separate engineering follow-up.

## CUDA dev headers missing for source build

Symptom: `pip install`/`uv pip install` of Dynamo wheels fails with `fatal error: cuda_runtime.h: No such file or directory`.

Cause: CUDA runtime is installed but dev headers are not.

Fix: install the matching CUDA toolkit dev package (`cuda-toolkit-12-9` / `cuda-toolkit-13-1`), or use the pre-built NGC container. See [Local Installation — System Requirements](getting-started/local-installation.md#system-requirements).

## HF_TOKEN / gated model access

Symptom: model download 401/403 on Llama / Kimi / Qwen-VL.

Fix: accept the model card's license on huggingface.co, then `export HF_TOKEN=hf_…`. For K8s, create a secret and mount it as `HF_TOKEN` on every pod. If running multi-node, set it on every node.

## Multimodal streaming fails on vLLM

Symptom: streaming responses on vision/audio models are truncated or fail mid-response.

Status: known issue in vLLM multimodal streaming at the current pinned version. Workaround: run non-streaming for multimodal workloads. Tracked as a separate engineering follow-up.

## DGDR stuck in `Failed`

Symptom: `DynamoGraphDeploymentRequest` transitions to `Failed` with `pareto_analysis.py` in the logs producing NaN.

Workaround: re-run with a narrower sweep; narrow sweeps bypass the NaN path in practice. Tracked as a separate engineering follow-up. See also [DGDR Known Issues](kubernetes/dgdr.md#known-issues).

## `disagg.sh` 503 on first requests

Symptom: `examples/backends/sglang/launch/disagg.sh` — first 2–3 requests return `503 Service Unavailable`.

Cause: race between frontend startup and prefill/decode worker registration.

Workaround: wait ~10 seconds after the script exits, then retry. Tracked as a separate engineering follow-up.
