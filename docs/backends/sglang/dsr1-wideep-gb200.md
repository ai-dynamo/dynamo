<!--
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Running DeepSeek-R1 Disaggregated with WideEP on GB200s

Dynamo supports SGLang's GB200 implementation of wide expert parallelism and large scale P/D for DeepSeek-R1! You can read their blog post [here](https://lmsys.org/blog/2025-06-16-gb200-part-1/) for more details. We provide a sample configuration that demonstrates WideEP and P/D  disaggregation. To run the exact configuration shown in the blog post, you can view the commands created by the SGLang team [here](https://github.com/sgl-project/sglang/issues/7227). In this example, we will run 1 prefill worker on 2 GB200 nodes (4 GPUs each) and 1 decode worker on 2 GB200 nodes (total 8 GPUs).

## Instructions

1. Build the Dynamo container for ARM64 (GB200) using the `build.sh` script.

> [!Note]
> Please ensure that you are building this on an ARM64 machine. The build script will automatically configure the correct platform and build arguments for SGLang on ARM64/GB200.

```bash
cd $DYNAMO_ROOT
./container/build.sh \
  --framework SGLANG \
  --platform linux/arm64 \
  --tag dynamo-wideep-gb200:latest
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

In each container, you should be in the /sgl-workspace/dynamo/examples/backends/sglang directory.

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

> [!IMPORTANT]
> If you encounter random CPU recv timeout issues during the warm-up phase in multi-GPU or multi-node setups, they are likely caused by DeepGEMM kernel compilation overhead.
> To avoid these non-deterministic timeouts, it's strongly recommended to precompile the DeepGEMM kernels before launching the SGLang engine. This ensures all kernels are cached and ready, preventing long initialization delays or distributed timeout errors. To precompile and use cached kernels, please execute the following commands:

```bash
# 1. Precompile DeepGEMM kernels
export SGLANG_DG_CACHE_DIR="/configs/dgcache/3p1dcache"
python3 -m sglang.compile_deep_gemm <ServerArgs>

# 2. Launch the engine with the same cache directory
export SGLANG_DG_CACHE_DIR="/configs/dgcache/3p1dcache"
python3 -m dynamo.frontend <ServerArgs>
```

> [!NOTE]
> There's a known issue where the compile request may fail due to missing bootstrap information, but the kernels are still successfully cached.
> Using a gradual warm-up phase and enabling caching for FlashInfer (similar to DeepGEMM) can further improve stability and reduce startup time.
> See https://github.com/sgl-project/sglang/issues/9867#issuecomment-3336551174 for more details.

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
