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

> **NOTE**: This example is based on an internal NVIDIA library that will soon be publicly released. The example won't work until the official release.

## Build docker

```
./container/build.sh --framework VLLM_NIXL --target dev --build-context nixl=<path to downloaded nixl repo @ c53bb19a6a114e9093071bd1f2904f996ae1839b>
```

## Run container

```
./container/run.sh --framework VLLM_NIXL --target dev -it
```

All of the commands below are run inside the same container.

## Run deployment

Add model to triton and start http server.

In terminal 0:
```
llmctl http add chat-models deepseek-ai/DeepSeek-R1-Distill-Llama-8B test-nixl.vllm.generate
TRT_LOG=DEBUG http --port 8181
```



### Monolithic deployment

In terminal 1:

```
cd /workspace/examples/python_rs/llm/vllm_nixl
CUDA_VISIBLE_DEVICES=0 python3 worker.py \
    --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --enforce-eager
```


### Disaggregated deployment

In terminal 1:

```
cd /workspace/examples/python_rs/llm/vllm_nixl
CUDA_VISIBLE_DEVICES=0 python prefill_worker.py \
    --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --enforce-eager \
    --kv-transfer-config \
    '{"kv_connector":"TritonNixlConnector"}'
```

In terminal 2:

```
cd /workspace/examples/python_rs/llm/vllm_nixl
CUDA_VISIBLE_DEVICES=1,2 python3 worker.py \
    --remote-prefill \
    --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --enforce-eager \
    --tensor-parallel-size 2 \
    --kv-transfer-config \
    '{"kv_connector":"TritonNixlConnector"}'
```


## Client

In another terminal:
```
curl localhost:8181/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "messages": [
      {"role": "user", "content": "What is the capital of France?"}
    ],
    "max_tokens": 10
  }'
```

## Run genai-perf

`genai-perf` is a tool for profiling and benchmarking LLM servers. It is already installed in the container. For more details, please refer to the [genai-perf README](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/perf_analyzer/genai-perf/README.html).

```
genai-perf profile \
  -m deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
  --url localhost:8181 \
  --endpoint-type chat \
  --streaming \
  --service-kind openai \
  --endpoint v1/chat/completions \
  --warmup-request-count 10 \
  --random-seed 123 \
  --synthetic-input-tokens-stddev 0 \
  --output-tokens-stddev 0 \
  --tokenizer deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
  --synthetic-input-tokens-mean 3000 \
  --output-tokens-mean 150 \
  --extra-inputs min_tokens:150 \
  --extra-inputs max_tokens:150 \
  --profile-export-file my_profile_export.json \
  --artifact-dir artifacts/ \
  --concurrency 10 \
  --request-count 40 \
  -- -v \
  --async
```

## Close deployment

Kill all python processes and clean up metadata files:

```
pkill -9 -f python
rm -r /tmp/nixl
```

## TODOs, limitations, known issues

- [ ] Add etcd for discovery
- [ ] Multi-node deployment support
- [ ] Enable chunked prefill
- [ ] Support mixed tp
- [ ] Process many remote prefill in one iteration
- [ ] Support recompute preemption
- [ ] Make sure decode does not preempt blocks before xfer finishes
- [ ] Layer wise transfer
- [ ] Non blocking send in prefill (cache manager should check xfer status)
- [ ] Test under load
- [ ] Support pp > 1
- [ ] Check why adding extra seed input is crashing vllm with remote prefill
- [ ] Unified worker for both prefill and decode
- [x] Require sending two parallel requests to start decode for the first time
- [x] Concurrency > 2 is not working
- [x] Parse cmdline args
- [x] Manual nixl example with tp1
- [x] Zero copy
- [x] Conditional remote prefill
- [x] Manual example with tp > 1
- [x] Run on triton distributed runtime
- [x] add oai http endpoint
- [x] Sample only on decode, do note return remote prefill response
- [x] Check if all transfers finished before moving to decode
- [x] Enable async output processing - could be working
