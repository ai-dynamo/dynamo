# Kimi-K2.5 Aggregated Deployment on Kubernetes with EAGLE Speculative Decoding 

## Prerequisites

- A Kubernetes cluster with the [Dynamo Operator](https://docs.nvidia.com/dynamo/) installed
- 8 x NVIDIA B200 GPUs
- A `hf-token-secret` Secret containing your Hugging Face token
- A pre-existing `model-cache` PVC
- A [patched container image](patch/) built from `nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime:<tag>`
- Replace the placeholder image tag `nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime:my-tag-patched` in `deploy-eagle-specdec.yaml` with your actual patched image

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
  speculative_model_dir: /opt/models/kimi-eagle-layers <- path is loaded here
```

## Deploy

```bash
kubectl apply -f deploy-eagle-specdec.yaml
```

This creates:
- A **ConfigMap** (`llm-config-kimi-k25-agg-eagle-specdec`) with the Eagle speculative decoding config
- A **DynamoGraphDeployment** (`kimi-k25-agg-eagle-specdec`) with a Frontend and TrtllmWorker serving `nvidia/Kimi-K2.5-NVFP4`
