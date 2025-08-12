<!--
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Running DeepSeek-R1 Disaggregated with WideEP on GB200s (NVFP4)

Together with the SGLang team, we have implemented NVFP4 support for DeepSeek-R1 on Blackwell GPUs! While many performance optimization are still ongoing, we wanted to provide an initial recipe that allows you to deploy the WideEP + P/D disaggregation setup in NVFP4! For now we provide instructions for 1 node prefill and 1 node decode. If you want to manually test a larger setup, you can set and edit the following CLI flags. Take a look at the [GB200-FP8 instructions](dsr1-wideep-gb200-fp8.md) for more details.

```bash
--dist-init-addr
--nnodes
--node-rank
--tp-size
--dp-size
```

<details>
<summary>Some SGLang PRs that should improve performance</summary>

- [#7667](https://github.com/sgl-project/sglang/pull/7667): Add `--enable-flashinfer-fp4-allgather` for FlashInfer cutlass MoE DP (max throughput)
  _10% e2e speedup_
- [#8638](https://github.com/sgl-project/sglang/pull/8638): FP8 KV Cache
- [#8588](https://github.com/sgl-project/sglang/pull/8588): Use FlashInfer's TRTLLM FP8 Blockscale GEMM
  _6% e2e improvement for low latency_
- [#8280](https://github.com/sgl-project/sglang/pull/8280): Enables all-gather for DP when using padding
  [#8539](https://github.com/sgl-project/sglang/pull/): Similar PR enables reducescatter instead of all-reduce following MOE/MLP layers in DeepSeek
  _5% e2e improvement_
- [#8811](https://github.com/sgl-project/sglang/pull/8811): Fix trtllm_fp4_block_scale_moe API routing_logits dim check (num_experts consistency)
- [#8770](https://github.com/sgl-project/sglang/pull/8770): Add changes required for unit tests in fused routed scaling
- [#8690](https://github.com/sgl-project/sglang/pull/8690): [2/2] SGLang: Fuse routed scaling factor into select_experts
  _Fuse multiply by routed_scaling_factor into select_experts, following TRT-LLM. 10% speed up on low latency_
- [#8330](https://github.com/sgl-project/sglang/pull/8330): Add unit test for flashinfer fp4 moe

</details>

## Instructions

1. Apply the following diff to the SGLang GB200 Dockerfile (temporary until [#8811](https://github.com/sgl-project/sglang/pull/8811) is merged)

```diff
diff --git a/docker/Dockerfile.gb200 b/docker/Dockerfile.gb200
index 90a80ec0b..6e1cd9245 100644
--- a/docker/Dockerfile.gb200
+++ b/docker/Dockerfile.gb200
@@ -53,10 +53,11 @@ RUN mkdir -p /tmp/gdrcopy && cd /tmp \
 RUN ln -sf /usr/lib/$(uname -m)-linux-gnu/libmlx5.so.1 /usr/lib/$(uname -m)-linux-gnu/libmlx5.so

 # Clone and install SGLang
+# Hack to get PR 8811
 WORKDIR /sgl-workspace
 RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel html5lib six \
- && git clone --depth 1 https://github.com/sgl-project/sglang.git \
- && cd sglang \
+ && git clone https://github.com/wenscarl/sglang.git \
+ && cd sglang && git checkout fix_trtllm_moe_create_weights \
  && case "$CUDA_VERSION" in \
       12.6.1) CUINDEX=126 ;; \
       12.8.1) CUINDEX=128 ;; \
```

2. Build the SGLang dockerfile

```bash
docker build \
  -f docker/Dockerfile.gb200 \
  -t sgl-blackwell-wideep-8811 \
  --build-arg BUILD_TYPE=blackwell \
  --build-arg CUDA_VERSION=12.8.1 \
  .
```

3. Build the Dynamo container

```bash
cd $DYNAMO_ROOT
docker build \
  -f container/Dockerfile.sglang-wideep \
  -t dynamo-wideep-gb200 \
  --build-arg MODE=blackwell \
  --build-arg SGLANG_IMAGE=sgl-blackwell-wideep-8811:latest \
  --build-arg ARCH=arm64 \
  --build-arg ARCH_ALT=aarch64 \
  .
```

4. You can run this container on each 4xGB200 node using the following command.

> [!IMPORTANT]
> We recommend downloading DeepSeek-R1 and then mounting it to the container. You can find the FP4 model [here](https://huggingface.co/nvidia/DeepSeek-R1-0528-FP4)

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

5. On the head prefill node, run the helper script provided to generate commands to start the `nats-server`, `etcd`. This script will also tell you which environment variables to export on each node to make deployment easier.

```bash
./utils/gen_env_vars.sh
```

6. Run the ingress and prefill worker

```bash
# run ingress
python3 -m dynamo.frontend --http-port=8000 &
# run prefill worker
MC_TE_METRIC=true \
MC_FORCE_MNNVL=1 \
NCCL_MNNVL_ENABLE=1 \
NCCL_CUMEM_ENABLE=1 \
SGLANG_USE_MESSAGE_QUEUE_BROADCASTER=0 \
SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK=1 \
PYTHONUNBUFFERED=1 \
python3 -m dynamo.sglang.worker \
    --disaggregation-transfer-backend nixl \
    --disaggregation-mode prefill \
    --host 0.0.0.0 \
    --decode-log-interval 1 \
    --max-running-requests 1536 \
    --context-length 4224 \
    --disable-radix-cache \
    --disable-shared-experts-fusion \
    --attention-backend cutlass_mla \
    --watchdog-timeout 1000000 \
    --model-path /model/ \
    --served-model-name nvidia/DeepSeek-R1-0528-FP4 \
    --trust-remote-code \
    --tp-size 4 \
    --dp-size 4 \
    --enable-dp-attention \
    --disable-cuda-graph \
    --chunked-prefill-size 32768 \
    --max-total-tokens 131072 \
    --max-prefill-tokens 32768 \
    --quantization modelopt_fp4 \
    --enable-flashinfer-cutlass-moe \
    --enable-ep-moe
```

7. Run the decode worker

```bash
MC_TE_METRIC=1 \
NCCL_MNNVL_ENABLE=1 \
MC_FORCE_MNNVL=1 \
NCCL_CUMEM_ENABLE=1 \
SGLANG_USE_MESSAGE_QUEUE_BROADCASTER=0 \
PYTHONUNBUFFERED=1 \
python3 -m dynamo.sglang.decode_worker \
    --disaggregation-transfer-backend nixl \
    --disaggregation-mode decode \
    --host 0.0.0.0 \
    --decode-log-interval 1 \
    --max-running-requests 1536 \
    --context-length 4224 \
    --max-total-tokens=2048 \
    --disable-radix-cache \
    --disable-shared-experts-fusion \
    --attention-backend cutlass_mla \
    --watchdog-timeout 1000000 \
    --model-path /model/ \
    --served-model-name nvidia/DeepSeek-R1-0528-FP4 \
    --trust-remote-code \
    --dist-init-addr "$HOST_IP:$PORT" \
    --disaggregation-bootstrap-port 30001 \
    --tp-size 4 \
    --dp-size 4 \
    --enable-dp-attention \
    --cuda-graph-bs 64 \
    --port 30000 \
    --quantization modelopt_fp4 \
    --enable-flashinfer-trtllm-moe \
    --enable-ep-moe
```