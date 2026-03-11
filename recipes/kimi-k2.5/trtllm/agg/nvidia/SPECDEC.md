# Kimi-K2.5 Aggregated Deployment on Kubernetes with EAGLE Speculative Decoding

## Prerequisites

- A Kubernetes cluster with the [Dynamo Operator](https://docs.nvidia.com/dynamo/) installed
- 8 x NVIDIA B200 GPUs
- A `hf-token-secret` Secret containing your Hugging Face token
- A pre-existing `model-cache` PVC
- An image, either:
  - A Dynamo + TRT-LLM runtime image built from `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc5`, or:
  - A Dynamo 1.0.0 release image with the Kimi-K2.5 patch. Follow the [patch guide](./patch/README.md) to learn more about how to patch the Dynamo image.
- Replace the placeholder image tag `nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime:my-tag-patched` in `deploy-eagle-specdec.yaml` with your actual patched image.

## Build the Runtime Image

To build a Dynamo+TensorRT-LLM image for this recipe:

1. Start from `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc5`.
2. Clone Dynamo at our tested commit `3dc2d053ebe7f3b858bd6955dac52cc373e730b1`.
3. Build and install the `ai_dynamo_runtime` Rust wheel with `maturin`.
4. Install the Dynamo Python package plus `nixl[cu12]`.
5. Append the Kimi TensorRT-LLM patch from `patch/kimi.patch`.

An example Dockerfile is available in [dockerfile.from-trtllm](dockerfile.from-trtllm). You should be able to run it with:

```bash
cd recipes/kimi-k2.5/trtllm/agg/nvidia
docker build -f dockerfile.from-trtllm -t <your tag here> .
```

## Additional Model Assets

`deploy-eagle-specdec.yaml` uses a separate EAGLE draft-head checkpoint in addition to the base Kimi weights. By default, the deployment recipe file loads the EAGLE checkpoint from path `/opt/models/kimi-eagle-layers` mounted from the `model-cache` PVC, so you should populate this path before deploying.

Expected contents under this folder:
- `config.json`
- `model.safetensors`

The worker config references this path via:

```yaml
speculative_config:
  decoding_type: Eagle
  max_draft_len: 3
  speculative_model_dir: /opt/models/kimi-eagle-layers
```

## Deploy

```bash
kubectl -n ${NAMESPACE} apply -f deploy-eagle-specdec.yaml
```

This creates:
- A **ConfigMap** (`llm-config-kimi-k25-agg-eagle-specdec`) with the Eagle speculative decoding config
- A **DynamoGraphDeployment** (`kimi-k25-agg-eagle-specdec`) with a Frontend and TrtllmWorker serving `nvidia/Kimi-K2.5-NVFP4`
