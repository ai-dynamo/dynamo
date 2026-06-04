# Nemotron Ultra Model Cache

## Required Model View

Expected worker path (all B200 + H200 recipes share this):

```text
/opt/models/patched/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4
```

Source repo and revision:

```text
nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4
https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4
revision: main, or set `MODEL_REVISION` to the release-pinned commit when provided
```

The DGD recipes expose the same public model ID as the served model name:

```text
nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4
```

Minimum files validated before vLLM startup:

```text
config.json
tokenizer.json
tokenizer_config.json
generation_config.json
ultra_v3_reasoning_parser.py
```

## Manifests

| Manifest | Purpose |
|---|---|
| `model-cache.yaml` | Namespace-local PVC contract for fresh namespaces. Edit `storageClassName` for the target cluster before apply. Do not apply over an existing platform-managed `shared-model-cache` PVC because PVC storage class and capacity are immutable. |
| `model-download.yaml` | No-GPU model population job. Downloads the pinned revision into the exact patched model path. |
| `model-validate.yaml` | No-GPU validation job. Fails with an actionable message if the model view is missing. |

## Usage

Then dry-run and apply in the target namespace. If the namespace already has a platform-managed `shared-model-cache`, skip `model-cache.yaml`; server-side dry-run against the existing PVC may warn about immutable field changes even though the platform cache itself is valid.

```bash
kubectl -n "${NAMESPACE}" apply --dry-run=server -f model-cache.yaml
kubectl -n "${NAMESPACE}" apply --dry-run=server -f model-download.yaml
kubectl -n "${NAMESPACE}" apply --dry-run=server -f model-validate.yaml

kubectl -n "${NAMESPACE}" apply -f model-cache.yaml
kubectl -n "${NAMESPACE}" apply -f model-download.yaml
kubectl -n "${NAMESPACE}" wait --for=condition=Complete job/nemotron-ultra-model-download --timeout=12h
kubectl -n "${NAMESPACE}" apply -f model-validate.yaml
kubectl -n "${NAMESPACE}" wait --for=condition=Complete job/nemotron-ultra-model-validate --timeout=30m
```

If the platform already provides `shared-model-cache`, skip `model-cache.yaml` and `model-download.yaml`, then run `model-validate.yaml` before applying a server recipe.
