<!--
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Running DeepSeek-R1 Disaggregated with WideEP on H100s

Dynamo supports SGLang's implementation of wide expert parallelism and large scale P/D for DeepSeek-R1! You can read their blog post [here](https://lmsys.org/blog/2025-05-05-large-scale-ep/) for more details. We provide a Dockerfile for this in `container/Dockerfile.sglang-wideep` and configurations to deploy this at scale. In this example, we will run 1 prefill worker on 4 H100 nodes and 1 decode worker on 9 H100 nodes (104 total GPUs).

## Instructions

1. Build the Dynamo container

```bash
cd $DYNAMO_ROOT
docker build -f container/Dockerfile.sglang-wideep . -t dynamo-wideep --no-cache
```

You can use a specific tag from the [lmsys dockerhub](https://hub.docker.com/r/lmsysorg/sglang/tags) by adding `--build-arg SGLANG_IMAGE_TAG=<tag>` to the build command.

2. You can run this container on each 8xH100 node using the following command.

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
    dynamo-wideep:latest
```

In each container, you should be in the `/sgl-workspace/dynamo/components/backends/sglang` directory.

3. Run the ingress and prefill worker

```bash
# run ingress
python3 -m dynamo.frontend --http-port=8000 &
# optionally run the http server that allows you to flush the kv cache for all workers (see benchmarking section below)
python3 utils/sgl_http_server.py --ns dynamo &
# run prefill worker
python3 -m dynamo.sglang \
  --model-path /model/ \
  --served-model-name deepseek-ai/DeepSeek-R1 \
  --skip-tokenizer-init \
  --disaggregation-mode prefill \
  --disaggregation-transfer-backend nixl \
  --disaggregation-bootstrap-port 30001 \
  --dist-init-addr ${HEAD_PREFILL_NODE_IP}:29500 \
  --nnodes 4 \
  --node-rank 0 \
  --tp-size 32 \
  --dp-size 32 \
  --enable-dp-attention \
  --decode-log-interval 1 \
  --enable-deepep-moe \
  --page-size 1 \
  --trust-remote-code \
  --moe-dense-tp-size 1 \
  --enable-dp-lm-head \
  --disable-radix-cache \
  --watchdog-timeout 1000000 \
  --enable-two-batch-overlap \
  --deepep-mode normal \
  --mem-fraction-static 0.85 \
  --deepep-config /configs/deepep.json \
  --ep-num-redundant-experts 32 \
  --ep-dispatch-algorithm dynamic \
  --eplb-algorithm deepseek
```

On the other prefill node (since this example has 4 total prefill nodes), run the same command but change `--node-rank` to 1,2, and 3

4. Run the decode worker on the head decode node

```bash
python3 -m dynamo.sglang \
  --model-path /model/ \
  --served-model-name deepseek-ai/DeepSeek-R1 \
  --skip-tokenizer-init \
  --disaggregation-mode decode \
  --disaggregation-transfer-backend nixl \
  --disaggregation-bootstrap-port 30001 \
  --dist-init-addr ${HEAD_DECODE_NODE_IP}:29500 \
  --nnodes 9 \
  --node-rank 0 \
  --tp-size 72 \
  --dp-size 72 \
  --enable-dp-attention \
  --decode-log-interval 1 \
  --enable-deepep-moe \
  --page-size 1 \
  --trust-remote-code \
  --moe-dense-tp-size 1 \
  --enable-dp-lm-head \
  --disable-radix-cache \
  --watchdog-timeout 1000000 \
  --enable-two-batch-overlap \
  --deepep-mode low_latency \
  --mem-fraction-static 0.835 \
  --ep-num-redundant-experts 32 \
  --cuda-graph-bs 128
```

On the other decode nodes (this example has 9 total decode nodes), run the same command but change `--node-rank` to 1, 2, 3, 4, 5, 6, 7, and 8

## Benchmarking

In the official [blog post repro instructions](https://github.com/sgl-project/sglang/issues/6017), SGL uses batch inference to benchmark their prefill and decode workers. They do this by pretokenizing the ShareGPT dataset and then creating a batch of 8192 requests with ISL 4096 and OSL 5 (for prefill stress test) and a batch of 40000 with ISL 2000 and OSL 100 (for decode stress test). If you want to repro these benchmarks, you will need to add the following flags to the prefill and decode commands:

prefill:

```bash
...
--max-running-requests 8192 \
--max-total-tokens 131072 \
--context-length 8192 \
--init-expert-location /configs/prefill_in4096.json \
--chunked-prefill-size 524288

```

decode:

```bash
...
--max-running-requests 18432 \
--context-length 4500 \
--init-expert-location /configs/decode_in2000out100.json
```

We currently provide 2 different ways to perform an end to end benchmark which includes using our OpenAI frontend and tokenization. We will continue to add better support for these sorts of large single batch workloads in the future.

1. **GenAI Perf to benchmark end to end performance with 8k ISL 256 OSL**
   We've found that 8k ISL 256 OSL provides a good baseline for measuring end to end disaggregated serving performance for DSR1. As WideEP allows for a higher throughput, we provide a script that runs this workload at high concurrencies. DeepGEMM kernels can sometimes take a while to warm up. We provide a short ramping warmup script that can be used.

Example usage:

```bash
# warmup
./utils/bench.sh HEAD_PREFILL_NODE_IP --type warmup
# if you ran the http server on the head prefill node, you can optionally flush the kv cache for all workers (similar to SGLangs benchmarking script)
curl -X POST http://${HEAD_PREFILL_NODE_IP}:9001/flush_cache
# run benchmark
./utils/bench.sh HEAD_PREFILL_NODE_IP --type e2e
```

2. **GenAI Perf to benchmark completions with custom dataset**
   We provide a script that generates a JSONL file of the ShareGPT dataset and then use GenAI Perf to benchmark the prefill and decode workers. We use ShareGPT in order to leverage the pre-existing EPLB distributions provided by the SGLang team. If you don't want to use ShareGPT - you can also use GenAI Perf's synthetic dataset setup But note you will have to use dynamic EPLB configurations or record your own as the `init-expert-location` provided by SGLang is tuned specifically for the ShareGPT dataset at a 4096 ISL and 5 OSL.

Example usage:

```bash
# generate data
python3 src/dynamo/sglang/utils/generate_bench_data.py --output data.jsonl --num-prompts 8192 --input-len 4096 --output-len 5 --model deepseek-ai/DeepSeek-R1
# if you ran the http server on the head prefill node, you can optionally flush the kv cache for all workers (similar to SGLangs benchmarking script)
curl -X POST http://${HEAD_PREFILL_NODE_IP}:9001/flush_cache
# run benchmark
./utils/bench.sh HEAD_PREFILL_NODE_IP --type custom_completions
```
