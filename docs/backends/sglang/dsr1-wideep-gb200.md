<!--
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Running DeepSeek-R1 Disaggregated with WideEP on GB200s

Dynamo supports SGLang's GB200 implementation of wide expert parallelism and large scale P/D for DeepSeek-R1! You can read their blog post [here](https://lmsys.org/blog/2025-06-16-gb200-part-1/) for more details. We provide a Dockerfile for this in `container/Dockerfile.sglang-wideep` and a sample configuration that demonstrates WideEP and P/D  disaggregation. To run the exact configuration shown in the blog post, you can view the commands created by the SGLang team [here](https://github.com/sgl-project/sglang/issues/7227). In this example, we will run 1 prefill worker on 2 GB200 nodes (4 GPUs each) and 1 decode worker on 2 GB200 nodes (total 8 GPUs).

## Instructions

1. Build the Dynamo container using the latest published dynamo version and stable sglang version. If you want to build from a local dynamo repo, you can add `--build-arg BRANCH_TYPE=local` to the build command. If you want to build from a remote dynamo repo, you can add `--build-arg BRANCH_TYPE=remote` to the build command. If you want to use a specific tag for the default sglang version, you can add `--build-arg SGLANG_IMAGE_TAG=<tag>` to the build command.

> [!Note]
> Please ensure that you are building this on an ARM64 machine. The correct SGLang image will be selected automatically via the multi-arch manifest.

> [!Note]
> Please use `--build-arg SGLANG_IMAGE_TAG=nightly-dev-20251019-fda0cb2a` to build the container due to a bug that we found with the DeepEP version being installed. This was fixed in [PR 11773](https://github.com/sgl-project/sglang/pull/11773). When SGLang releases a version > `0.5.3.post3` we will update these instructions.

```bash
cd $DYNAMO_ROOT
docker build \
  -f container/Dockerfile.sglang-wideep \
  -t dynamo-wideep-gb200 \
  --build-arg SGLANG_IMAGE_TAG=nightly-dev-20251019-fda0cb2a \
  --no-cache \
  .
```

2. You can run this container on each 4xGB200 node using the following command.

> [!IMPORTANT]
> We recommend downloading DeepSeek-R1 and then mounting it to the container. You can find the model [here](https://huggingface.co/deepseek-ai/DeepSeek-R1)

```bash
docker run \
    --gpus all \
    -it \
    --rm \
    --network host \
    --volume /PATH_TO_DSR1_MODEL/:/model/ \
    --shm-size=10G \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --ulimit nofile=65536:65536 \
    --cap-add CAP_SYS_PTRACE \
    --ipc host \
    dynamo-wideep-gb200:latest
```

3. Run the ingress and prefill worker

```bash
# run ingress
python3 -m dynamo.frontend --http-port=8000 &
# run prefill worker
DYN_SKIP_SGLANG_LOG_FORMATTING=1 \
MC_TE_METRIC=true \
SGLANG_DISAGGREGATION_HEARTBEAT_MAX_FAILURE=100000 \
SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=100000 \
SGLANG_DISAGGREGATION_WAITING_TIMEOUT=100000 \
MC_FORCE_MNNVL=1 \
NCCL_MNNVL_ENABLE=1 \
NCCL_CUMEM_ENABLE=1 \
SGLANG_USE_MESSAGE_QUEUE_BROADCASTER=0 \
SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK=1 \
PYTHONUNBUFFERED=1 \
python3 -m dynamo.sglang \
  --served-model-name deepseek-ai/DeepSeek-R1 \
  --model-path /model/ \
  --skip-tokenizer-init \
  --trust-remote-code \
  --disaggregation-mode prefill \
  --dist-init-addr ${HEAD_PREFILL_NODE_IP}:29500 \
  --disaggregation-bootstrap-port 30001 \
  --nnodes 2 \
  --node-rank 0 \
  --tp-size 8 \
  --dp-size 8 \
  --enable-dp-attention \
  --host 0.0.0.0 \
  --decode-log-interval 1 \
  --max-running-requests 6144 \
  --context-length 2716 \
  --disable-radix-cache \
  --moe-a2a-backend deepep \
  --load-balance-method round_robin \
  --deepep-mode normal \
  --moe-dense-tp-size 1 \
  --enable-dp-lm-head \
  --disable-shared-experts-fusion \
  --ep-num-redundant-experts 32 \
  --ep-dispatch-algorithm static \
  --eplb-algorithm deepseek \
  --attention-backend cutlass_mla \
  --watchdog-timeout 1000000 \
  --disable-cuda-graph \
  --chunked-prefill-size 16384 \
  --max-total-tokens 32768 \
  --mem-fraction-static 0.82 \
  --log-level debug \
  --disaggregation-transfer-backend nixl
```

On the other prefill nodes (this example has 2 total prefill nodes), run the same command but change `--node-rank` to 1

4. Run the decode worker on the head decode node

```bash
SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=768 \
MC_TE_METRIC=true \
SGLANG_DISAGGREGATION_HEARTBEAT_MAX_FAILURE=100000 \
SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=100000 \
SGLANG_DISAGGREGATION_WAITING_TIMEOUT=100000 \
SGLANG_HACK_SEQ_BOOTSTRAP_ROOM=1 \
SGLANG_MOONCAKE_CUSTOM_MEM_POOL=True \
NCCL_MNNVL_ENABLE=1 \
MC_FORCE_MNNVL=1 \
NCCL_CUMEM_ENABLE=1 \
SGLANG_USE_MESSAGE_QUEUE_BROADCASTER=0 \
SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK=1 \
PYTHONUNBUFFERED=1 \
python3 -m dynamo.sglang \
  --served-model-name deepseek-ai/DeepSeek-R1 \
  --model-path /model/ \
  --skip-tokenizer-init \
  --trust-remote-code \
  --disaggregation-mode decode \
  --dist-init-addr ${HEAD_DECODE_NODE_IP}:29500 \
  --disaggregation-bootstrap-port 30001 \
  --nnodes 2 \
  --node-rank 0 \
  --tp-size 8 \
  --dp-size 8 \
  --enable-dp-attention \
  --host 0.0.0.0 \
  --decode-log-interval 1 \
  --max-running-requests 36864 \
  --context-length 2716 \
  --disable-radix-cache \
  --moe-a2a-backend deepep \
  --prefill-round-robin-balance \
  --deepep-mode low_latency \
  --moe-dense-tp-size 1 \
  --enable-dp-lm-head \
  --cuda-graph-max-bs 256 \
  --disable-shared-experts-fusion \
  --ep-num-redundant-experts 32 \
  --ep-dispatch-algorithm static \
  --eplb-algorithm deepseek \
  --attention-backend cutlass_mla \
  --watchdog-timeout 1000000 \
  --chunked-prefill-size 36864 \
  --mem-fraction-static 0.82 \
  --log-level debug \
  --disaggregation-transfer-backend nixl
```

On the other decode nodes (this example has 2 total decode nodes), run the same command but change `--node-rank` to 1.