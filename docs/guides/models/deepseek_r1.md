# Deploying DeepSeek R1 with Dynamo and TensorRT-LLM on GB200

This guide provides step-by-step instructions for deploying DeepSeek R1 using
Dynamo with TensorRT-LLM backend on GB200 systems.

## Table of Contents

- [Setup](#setup)
- [Deployment](#deployment)
  - [Aggregated Deployment](#aggregated-deployment)
  - [Disaggregated Deployment](#disaggregated-deployment)
- [Benchmarking](#benchmarking)
  - [Setup](#setup-1)
  - [Running Benchmarks](#running-benchmarks)
- [Troubleshooting](#troubleshooting)
  - [Common Issues](#common-issues)
  - [Performance Tuning](#performance-tuning)
- [Next Steps](#next-steps)

## Setup

### Pre-requisites

Before starting, ensure you have:

- NVIDIA GB200 GPU nodes
- Access to Docker containers on all nodes
- Network connectivity between all nodes

### Container Setup

1. Clone the Dynamo repository:
    ```bash
    git clone -b release/0.3.1 https://github.com/ai-dynamo/dynamo.git
    cd dynamo
    ```

2. Build the container:

    ```bash
    # Build for ARM architecture (required for GB200)
    # For x86 machines, remove the `--platform` flag
    ./container/build.sh --framework tensorrtllm --platform linux/arm64
    ```

> [!Tip]
>
> For the latest experimental features and bug fixes in TRTLLM that may not be in a released pip wheel yet, use the experimental TensorRT-LLM build to build TensorRT-LLM from source:
>
> ```bash
> # Install git-lfs if not already installed
> apt-get update && apt-get -y install git git-lfs
>
> # Build TRTLLM from source
> ./container/build.sh --framework tensorrtllm --platform linux/arm64 --use-default-experimental-tensorrtllm-commit
> ```
>
> Note that building from source may not go through the same testing as released pre-built wheels,
> and that the build may take several hours. However, it can be handy when there is a bug fix
> on the repository that hasn't yet been released in a pre-built wheel yet.

3. On all nodes, run the container:

    ```bash
    docker run -it \
        --gpus all \
        --network host \
        -v /tmp/hf_cache:/root/.cache/huggingface \
        dynamo:latest-tensorrtllm
    ```

> [!Tip]
>
> Docker Recommendations:
> 1. Mounting HF cache to host is optional, but recommended for repeated runs.
>    You can replace /tmp/hf_cache with ~/.cache/huggingface or any path you'd like to mount.
> 2. Publishing your Dynamo-TRTLLM image to a container registry, and pulling that
>    image instead of `dynamo:latest-tensorrtllm` will make it easier to distribute and
>    re-use across nodes and environments.

### Environment Setup

1. On the head node, start core services:
    ```bash
    # Start NATS and etcd for discovery and communication
    nats-server -js &
    etcd --listen-client-urls http://0.0.0.0:2379 \
        --advertise-client-urls http://0.0.0.0:2379 \
        --data-dir /tmp/etcd &
    ```

2. On worker nodes, set environment variables so they can communicate with the head node:
    ```bash
    export HEAD_NODE_IP="<head-node-ip>"
    export NATS_SERVER="nats://${HEAD_NODE_IP}:4222"
    export ETCD_ENDPOINTS="${HEAD_NODE_IP}:2379"
    ```



## Deployment

For simplicity, these steps assume that the model being deployed fits
on a single node, and that every worker (whether aggregated, prefill-only,
or decode-only) will be launched on independent nodes.

For example, `nvidia/DeepSeek-R1-FP4` with Tensor Parallelism and Expert
Parallelism set to 4 on a 4xGB200 node.

### Aggregated Deployment

1. On the head node:
    ```bash
    cd /workspace/examples/tensorrt_llm
    dynamo serve graphs.agg:Frontend -f configs/deepseek_r1/agg.yaml
    ```

    This will start:
    - HTTP frontend on port 8000
    - TensorRT-LLM worker handling both prefill and decode

2. (Optional) On any additional worker nodes, you can start an additional replica aggregated worker that Dynamo will load balance and route to automatically:
    ```bash
    cd /workspace/examples/tensorrt_llm
    dynamo serve components.worker:TensorRTLLMWorker \
        -f configs/deepseek_r1/agg.yaml \
        --service-name TensorRTLLMWorker
    ```

### Disaggregated Deployment

1. On the head node:
    ```bash
    cd /workspace/examples/tensorrt_llm
    dynamo serve graphs.agg:Frontend -f configs/deepseek_r1/disagg.yaml &
    ```

    This will start:
    - HTTP frontend on port 8000
    - TensorRT-LLM worker handling both decode-only
      ("decode-only" mode controlled by `remote-prefill: true` in `configs/deepseek_r1/disagg.yaml`)

2. On prefill worker nodes:
    ```bash
    cd /workspace/examples/tensorrt_llm
    dynamo serve components.prefill_worker:TensorRTLLMPrefillWorker \
        -f configs/deepseek_r1/disagg.yaml \
        --service-name TensorRTLLMPrefillWorker
    ```

3. On decode worker nodes:
    ```bash
    cd /workspace/examples/tensorrt_llm
    dynamo serve components.worker:TensorRTLLMWorker \
        -f configs/deepseek_r1/disagg.yaml \
        --service-name TensorRTLLMWorker
    ```

## Benchmarking

### Setup

1. Install genai-perf:
    ```bash
    pip install genai-perf
    ```

2. (Optional) Verify the model is registered and available:
    ```bash
    curl ${HEAD_NODE_IP}:8000/v1/models
    ```

3. (Optional) Verify the model is functional:
    ```bash
    curl localhost:8000/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{
        "model": "nvidia/DeepSeek-R1-FP4",
        "messages": [
        {
            "role": "user",
            "content": "Give me an idea for a dungeons & dragons campaign."
        }
        ],
        "stream": false,
        "max_tokens": 256
      }'
    ```

### Running Benchmarks

Basic benchmark of a specific scenario:

```bash
ISL=8000
OSL=256
MODEL="nvidia/DeepSeek-R1-FP4"
CONCURRENCY=64
genai-perf profile \
  --model "${MODEL}" \
  --tokenizer "${MODEL}" \
  --url "${HEAD_NODE_IP}:8000" \
  --endpoint-type chat \
  --streaming \
  --synthetic-input-tokens-mean ${ISL} \
  --output-tokens-mean ${OSL} \
  --extra-inputs max_tokens:${OSL} \
  --extra-inputs min_tokens:${OSL} \
  --extra-inputs ignore_eos:true \
  --extra-inputs "{\"nvext\":{\"ignore_eos\":true}}" \
  --concurrency ${CONCURRENCY} \
  --request-count $(($CONCURRENCY*10)) \
  --warmup-request-count $(($CONCURRENCY*2)) \
  --num-dataset-entries $(($CONCURRENCY*12)) \
  --random-seed 100 \
  -- \
  -v \
  --max-threads ${CONCURRENCY}
```

## Troubleshooting

### Common Issues

1. Memory Issues:
- Check GPU memory usage
- Adjust `free_gpu_memory_fraction` or `max_batch_size` in config
- Consider reducing batch sizes

2. Performance Issues:
- Monitor GPU utilization
    - For disaggregated serving in particular, finding the right ratio of P/D
      instances is crucial to overall performance and efficiency.
- Monitor server-side metrics
    ```bash
    curl ${HEAD_NODE_IP}:8000/metrics
    ```
- Verify worker configuration (`configs/deepseek_r1/*.yaml`)
- Check network connectivity between nodes and GPUs (RDMA, Infiniband, etc.)

3. (Slurm Clusters Only) MPI Spawn Failure:

    If you encounter errors related to MPI Spawn failure from TRTLLM when running
    on a Slurm cluster interactively, try temporarily unsetting these variables:

    ```bash
    unset SLURM_JOBID SLURM_JOB_ID SLURM_NODELIST
    ```

    The more proper solution may be to launch any MPI based jobs with
    `srun` rather than interactively.

### Performance Tuning

1. Expriment with Ratio of # Prefill/Decode instances:

    This is crucial to overall performance/efficiency of disaggregated serving,
    and is sensitive to the input/output sequence lengths received by the server.

    For example, for very long inputs and very short outputs, it
    may require multiple prefill instances to generate enough load for
    a single decode instance to be fully saturated, and vice versa.

2. DP Attention (enable for max throughput, disable for min latency):

    ```yaml
    # configs/deepseek/*_llm_api_config.yaml
    enable_attention_dp: true
    ```

3. Tensor or Expert Parallel Size:

    If using different hardware like DGX B200 (ex: 8xB200 GPUs instead of 4xGB200), you can
    also experiment with the parallelism settings to use all the available GPUs:

    ```yaml
    # configs/deepseek/*_llm_api_config.yaml
    tensor_parallel_size: 4     # Adjust based on your GPU count
    moe_expert_parallel_size: 4 # Adjust based on your GPU count
    ```

For TRTLLM-specific configuration recommendations, see the guide on
[how to get best performance on DeepSeek-R1 in TensorRT-LLM
](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/Best_perf_practice_on_DeepSeek-R1_in_TensorRT-LLM.md).

## Next Steps

- Explore Multi-Token Prediction (MTP) with a custom dataset
- Explore WideEP for large-scale deployments
- Monitor and tune performance for specific use case

For more detailed information about specific features or configurations, refer to the [TensorRT-LLM examples](../../../examples/tensorrt_llm/README.md).
