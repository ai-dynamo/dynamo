# Building local dev container with local TRTLLM.

1. Download a post-merge wheel close to your local TRTLLM's commit. [Link](https://urm.nvidia.com/artifactory/sw-tensorrt-generic/llm-artifacts/LLM/main/L0_PostMerge/).
    1. Find the corresponding commit ID in the `waive_list` folder (it's in the `.txt` filenames).
1. Build the dynamo dev container:


```bash
$ ./build.sh \
  --framework TRTLLM \
  --target local-dev \
  --tensorrtllm-pip-wheel-dir <wheel_dir> \
  --tensorrtllm-commit <wheel_commit_id>
```

Grab some ☕, this can take a while.

3. Run the container using:

```bash
$ ./run.sh \
--image dynamo:latest-trtllm-local-dev \
-v <mounts_you_want_to_forward> \
--gpus device=${NV_GPU} \
--hf-cache ${HF_HOME} \
--user $(id -u):$(id -g) \
-it \
--mount-workspace \
-- \
/bin/bash
```

⚠️ The first things you will want to do when this container is spun up is to `docker exec -it` into it, and run the `etcd` and `nats-server` commands printed to console, and leave them running the background. You only need to do this once per lifetime of the dynamo container (unless you somehow kill them afterwards manually).

# Local edits to TRTLLM.

If you plan to edit your local TRTLLM code, and have it be reflected in the dynamo container:

1. Go to your repo directory inside the container.
1. Set `TRTLLM_PRECOMPILED_LOCATION=<wheel_used_during_dynamo_container_build>`.
1. Run `pip uninstall -y tensorrt_llm` and then `pip install -e .[devel]`.

# Local edits to dynamo.

Since we mount the dynamo repo via `--mount-workspace` into the dynamo dev container, we can rebuild dynamo in the container using:

```bash
# 1) Rebuild Rust -> Python bindings.
$ cd /workspace/lib/bindings/python
$ maturin develop --uv

# 2) Reinstall the Python package in editable mode.
$ cd /workspace
$ uv pip install -e .
```

# Hosting a dynamo server (qwen3 VL MoE example).

```bash
$ export MODEL=Qwen/Qwen3-VL-30B-A3B-Instruct-FP8
```

## Aggregated mode.

You need to run 2 commands, one for the frontend, and another for the engine.

```bash
$ python -m dynamo.frontend --http-port 8123
```

Take note of the `--modality` argument - without it, the model will think it gets no images, even if you
include them in the requests.

```bash
$ python -m dynamo.trtllm \
  --modality multimodal \
  --model-path $MODEL \
  --extra-engine-args perf_config.yaml
```

where the `perf_config.yaml` contains:

```yaml
kv_cache_config:
  enable_block_reuse: false
enable_chunked_prefill: true
```

## EPD disagg.

Adjust the `_CUDA_VISIBLE_DEVICES` environment variables as appropriate. Note that this script
internally also runs the frontend, so if you already have one running, you will likely want
to kill that process first.

```bash
MODEL_PATH=$MODEL \
SERVED_MODEL_NAME=$MODEL \
PREFILL_CUDA_VISIBLE_DEVICES=0 \
DECODE_CUDA_VISIBLE_DEVICES=1 \
ENCODE_CUDA_VISIBLE_DEVICES=2 \
./examples/backends/trtllm/launch/epd_multimodal_image_and_embeddings.sh
```

# OCRBench.

## Run the VLMEValkit container.

```bash
$ export BENCHMARK_IMAGE=nvcr.io/nvidia/eval-factory/vlmevalkit:25.11

$ docker run -it --rm \
--net=host \
--entrypoint=/bin/bash \
${BENCHMARK_IMAGE}
```

Then, in the container:

```bash
nemo-evaluator run_eval \
--run_config run_config.yml \
--eval_type ocrbench \
--output_dir ./ocrbench_results \
--model_id Qwen/Qwen3-VL-30B-A3B-Instruct-FP8 \
--model_type vlm \
--model_url http://localhost:8123/v1/chat/completions
```

where `run_config.yml` contains:

```yaml
target:
  api_endpoint:
    adapter_config:
      interceptors:
        - name: "payload_modifier"
          enabled: true
          config:
            params_to_remove:
              - "dataset"
        - name: "caching"
          enabled: true
          config:
            cache_dir: /results/cache
            save_requests: true
            save_responses: true
            reuse_cached_responses: false
        - name: "endpoint"
          enabled: true
          config: {}
```

The `payload_modifier` removes the extra `"dataset": "OCRBench"` from the request payload, which
is not part of the OAI API, and therefore rejected by `dynamo` / `trtllm-serve` (`vllm serve` allows extras, but ignores them).

Note that `nemo-evaluator` will try to download the `OCRBench.tsv` file from a URL that is deny-listed by the NVIDIA network.

A workaround is to download the file from [here](https://huggingface.co/ambivalent02/openencompass_tsv/blob/5c04f3d6230103b6cad5ebe39c27b1c478e70235/OCRBench.tsv), and then place it in `~/LMUData` in the eval container.
