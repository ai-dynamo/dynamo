Benchmarks Documentation
-------------------------

## Quickstart

Download the [mooncake_trace.jsonl file](https://github.com/kvcache-ai/Mooncake/blob/d21da178bae8db9651cf18a76824c084145fc725/mooncake_trace.jsonl).

```bash
wget https://raw.githubusercontent.com/kvcache-ai/Mooncake/d21da178bae8db9651cf18a76824c084145fc725/mooncake_trace.jsonl
```

Next, use `datagen analyze --input-file mooncake_trace.jsonl` to analyze the dataset:

```bash
Analyzing dataset: mooncake_trace.jsonl
Using block size: 512

Loading dataset from mooncake_trace.jsonl...
Dataset loaded: 23608 examples
Hash counter built: 183166 unique hash IDs
+-----------------------+--------------------+--------------------+-------+--------+--------+--------------------+----------+
|                       |        Mean        |      Std Dev       |  Min  |  P25   | Median |        P75         |   Max    |
+-----------------------+--------------------+--------------------+-------+--------+--------+--------------------+----------+
|     Input Length      | 8589.956836665537  | 10994.026625418108 | 890.0 | 3227.0 | 6345.0 |       7470.0       | 125546.0 |
|    Context Length     | 5755.485936970518  | 8101.296710263215  | 512.0 | 2048.0 | 6144.0 |       6144.0       | 122368.0 |
| Unique Prompt Length  | 2834.4708996950185 | 8561.283290412863  |  0.0  | 266.0  | 560.0  |       1103.0       | 125034.0 |
|     Output Length     | 182.1338952897323  | 242.27604216357528 |  1.0  |  13.0  |  30.0  |       356.0        |  2000.0  |
| Theoretical Hit Rates | 0.6297856206978413 | 0.3256570779031306 |  0.0  |  0.36  |  0.75  | 0.9230769230769231 |   1.0    |
+-----------------------+--------------------+--------------------+-------+--------+--------+--------------------+----------+
```

### Breakdown of Summary Statistics

#### Input Length

The total number of input tokens in the request before generation starts. This is the concatenation of context + unique prompt, measured in tokens (not characters).

Example:

```bash
"Given the following document: [...long article...], summarize it."
```

Then the tokenized input includes both:
- Repeated prefix blocks (e.g., shared “system” or “article” intro),
- Unique prompt blocks (“summarize it”).

Formula: input_length = context_length + unique_prompt_length

#### Context Length

The portion of the input that has been seen before in previous requests, measured by overlapping `hash_ids`. This simulates prefix reuse and is fully cacheable in the KV cache.

It represents the input that could be preloaded into memory by the model.

Example:

```python
Current hash_ids = [0, 1, 2, 3]
Previously seen  = [0, 1, 2]
→ context_length = 3 * block_size
```

This is what contributes to KV cache hits during prefill.

#### Unique Prompt Length

The non-repeated suffix portion of the input. These are new `hash_ids` that have never been seen in prior requests.

These cannot be reused and must be freshly encoded and stored in the attention cache.

If input_length = 4096 and context_length = 3072, then:

```
unique_prompt_length = 4096 - 3072 = 1024
```

A high unique prompt length → lower cache reuse and higher compute cost for prefill.

#### Output Length

The number of tokens that the model is expected to generate after consuming the input. This determines how long the decoding phase will run.

Example:

```json
"output_length": 150
```

This means the model must generate 150 new tokens, usually with autoregressive decoding.

High output length increases:
- Latency (especially ITL),
- Memory usage during decoding.

#### Theoretical Hit Rates

The proportion of input blocks (`hash_ids`) that are already cached, assuming an infinite KV cache.

This metric estimates how reusable the prefix is and is useful for modeling upper-bound cache effectiveness.

Formula:

```
theoretical_hit_rate = # cached hash_ids / total hash_ids
```

Example:

```python
hash_ids = [0, 1, 2, 3, 4, 5]
cached = [0, 1, 2]
→ theoretical_hit_rate = 3 / 6 = 0.5
```

It ranges from 0.0 (all input is new) to 1.0 (fully cached input).

#### Summary Table

| Metric                  | Description                                            | Cacheable? |
|------------------------|--------------------------------------------------------|------------|
| input_length            | Total input (context + prompt)                        | Partially  |
| context_length          | Shared prefix seen before                            | ✅ Yes     |
| unique_prompt_length    | New, non-reused input blocks                          | ❌ No      |
| output_length           | Tokens to generate after input                        | ❌ No      |
| theoretical_hit_rate    | Ratio of reused input blocks to total blocks          | ✅ Yes     |

### Hashes

```bash
python minrex_prompt_to_hash_ids.py
```

```jsonl
{"prompt": "Summarize the following article about climate change..."}
{"prompt": "Write a haiku about GPU cache coherence."}
{"prompt": "Translate the sentence into French: 'Hello world'"}
```

```jsonl
{"prompt": "The solar system is composed of the sun, eight planets, moons, asteroids, and comets. Explain how gravity holds them together."}
{"prompt": "The solar system is composed of the sun, eight planets, moons, asteroids, and comets. Describe how they formed over time."}
{"prompt": "Write a haiku about a GPU that dreams of more VRAM."}
```

```bash
touch data/prompts.jsonl ## add above to here
python ./scripts/generate_trace_from_prompts.py ./data/prompts.jsonl ./data/output.jsonl 4
```


Install genai-perf:

```bash
uv pip install genai-perf
```

```bash
docker pull nvcr.io/nvidia/tritonserver:25.06-py3-sdk
docker run \
  --gpus all \
  --rm -it \
  --net host \
  -v $PWD:/workspace/benchmarks \
  nvcr.io/nvidia/tritonserver:25.06-py3-sdk
```

```bash
cargo run --bin dynamo-run in=http out=mistralrs deepseek-ai/DeepSeek-R1-Distill-Llama-8B

curl -L -o Llama-3.2-3B-Instruct-Q4_K_M.gguf "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf?download=true"

cargo run --bin dynamo-run in=http out=mistralrs Llama-3.2-3B-Instruct-Q4_K_M.gguf
cargo run --features cuda --bin dynamo-run in=http out=mistralrs Llama-3.2-3B-Instruct-Q4_K_M.gguf

uv pip install "ai-dynamo[vllm]"

curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
    "messages": [
      {
        "role": "user",
        "content": "What is the weather like in San Francisco today?"
      }
    ]
    }' | jq

```


```bash
export HF_TOKEN=...
huggingface-cli login

bash llm/perf.sh --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
                 --url http://localhost:8080 \
                 --mode aggregated \
                 --tp 2 \
                 --dp 2 \
                 --concurrency 1,2,4,8,16,32 \
                 --artifacts-root-dir artifacts_root

bash llm/perf.sh --model bartowski/Llama-3.2-3B-Instruct-Q4_K_M.gguf \
                 --url http://localhost:8080 \
                 --mode aggregated \
                 --tp 2 \
                 --dp 2 \
                 --concurrency 1,2,4,8,16,32 \
                 --artifacts-root-dir artifacts_root \
                 --deployment-kind vllm

```

```bash

uv pip install "ai-dynamo[vllm]"
docker compose -f deploy/docker-compose.yml up -d

sudo apt install libstdc++-12-dev
sudo add-apt-repository ppa:ubuntu-toolchain-r/test -y
sudo apt update
sudo apt install libstdc++6

export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
strings /usr/lib/x86_64-linux-gnu/libstdc++.so.6 | grep GLIBCXX_3.4.32
echo 'set -x LD_LIBRARY_PATH /usr/lib/x86_64-linux-gnu $LD_LIBRARY_PATH' >> ~/.config/fish/config.fish

cd components/backends/vllm
dynamo serve graphs.agg:Frontend -f configs/agg.yaml

### still did not work


curl localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "messages": [
    {
        "role": "user",
        "content": "Hello, how are you?"
    }
    ],
    "stream":false,
    "max_tokens": 300
  }' | jq
```


```bash
docker pull nvcr.io/nvidia/ai-dynamo/vllm-runtime:0.3.2

docker run \
  -it --rm \
  --gpus all \
  -p 8000:8000 \
  -p 8080:8080 \
  nvcr.io/nvidia/ai-dynamo/vllm-runtime:0.3.2 \
  bash

./container/build.sh --framework VLLM

docker run \
  -it --rm \
  --gpus all \
  -p 8080:8080 \
  dynamo:latest-vllm \
  bash
```

```bash
Try the following to begin interacting with a model:
> dynamo --help
> dynamo run Qwen/Qwen2.5-3B-Instruct

To run more complete deployment examples, instances of etcd and nats need to be
accessible within the container. This is generally done by connecting to
existing etcd/nats services from the host or other containers. For simple
cases, you can start them in the container as well:

nats-server -js &
etcd --listen-client-urls http://0.0.0.0:2379 --advertise-client-urls http://0.0.0.0:2379 --data-dir /tmp/etcd &

With etcd/nats accessible, run the examples:
> cd examples/hello_world
> dynamo serve hello_world:Frontend

apt update
apt install curl jq

curl -X GET localhost:8080/v1/models


nats-server -js &
etcd --listen-client-urls http://0.0.0.0:2379 --advertise-client-urls http://0.0.0.0:2379 --data-dir /tmp/etcd &
dynamo run in=http out=vllm Qwen/Qwen2.5-3B-Instruct

time curl localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [
    {
        "role": "user",
        "content": "Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?Hello, how are you?"
    }
    ],
    "stream":false,
    "max_tokens": 300
  }'


time curl localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [
    {
        "role": "user",
        "content": "Can you write the alphabet?Can you write the alphabet?Can you write the alphabet?Can you write the alphabet?Can you write the alphabet?Can you write the alphabet?Can you write the alphabet?Can you write the alphabet?Can you write the alphabet?Can you write the alphabet?Can you write the alphabet?Can you write the alphabet?Can you write the alphabet?Can you write the alphabet?Can you write the alphabet?Can you write the alphabet?Can you write the alphabet?Can you write the alphabet?Can you write the alphabet?Can you write the alphabet?Can you write the alphabet?Can you write the alphabet?Can you write the alphabet?Can you write the alphabet?Can you write the alphabet?Can you write the alphabet?Can you write the alphabet?Can you write the alphabet?Can you write the alphabet?Can you write the alphabet?Can you write the alphabet?Can you write the alphabet?Can you write the alphabet?Can you write the alphabet?Can you write the alphabet?Can you write the alphabet?Can you write the alphabet?Can you write the alphabet?Can you write the alphabet?Can you write the alphabet?Can you write the alphabet?Can you write the alphabet?Can you write the alphabet?Can you write the alphabet?Can you write the alphabet?Can you write the alphabet?Can you write the alphabet?Can you write the alphabet?Can you write the alphabet?Can you write the alphabet?Can you write the alphabet?Can you write the alphabet?Can you write the alphabet?Can you write the alphabet?Can you write the alphabet?Can you write the alphabet?Can you write the alphabet?Can you write the alphabet?Can you write the alphabet?Can you write the alphabet?Can you write the alphabet?Can you write the alphabet?Can you write the alphabet?Can you write the alphabet?Can you write the alphabet?Can you write the alphabet?Can you write the alphabet?Can you write the alphabet?Can you write the alphabet?Can you write the alphabet?Can you write the alphabet?Can you write the alphabet?Can you write the alphabet?Can you write the alphabet?Can you write the alphabet?Can you write the alphabet?Can you write the alphabet?Can you write the alphabet?Can you write the alphabet?Can you write the alphabet?Can you write the alphabet?Can you write the alphabet?Can you write the alphabet?Can you write the alphabet?"
    }
    ],
    "stream":false,
    "max_tokens": 300
  }'




curl localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-3B-Instruct",
    "messages": [
    {
        "role": "user",
        "content": "Hello, how are you?"
    }
    ],
    "stream":false,
    "max_tokens": 300
  }' | jq

curl localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-3B-Instruct",
    "messages": [
    {
        "role": "user",
        "content": "Hello, how are you?"
    }
    ],
    "stream":false,
    "max_tokens": 300
  }' | jq


bash llm/perf.sh --model Qwen/Qwen2.5-3B-Instruct \
                 --url http://localhost:8080 \
                 --mode aggregated \
                 --tp 2 \
                 --dp 2 \
                 --concurrency 1,2,4,8,16,32 \
                 --artifacts-root-dir artifacts_root

python llm/perf.py --model Qwen/Qwen2.5-3B-Instruct \
                   --url http://localhost:8000 \
                   --mode aggregated \
                   --tp 1 \
                   --dp 1 \
                   --concurrency 1,2,4,8,16,32 \
                   --artifacts-root-dir artifacts_root

python llm/perf.py --model Qwen/Qwen2.5-3B-Instruct \
                   --url http://localhost:8080 \
                   --mode aggregated \
                   --tp 1 \
                   --dp 1 \
                   --concurrency 1,2,4,8,16,32 \
                   --artifacts-root-dir artifacts_root


bash llm/perf.sh --model Qwen/Qwen2.5-3B-Instruct \
                 --url http://localhost:8000 \
                 --mode aggregated \
                 --tp 1 \
                 --dp 1 \
                 --concurrency 1,2,4,8,16,32 \
                 --artifacts-root-dir artifacts_root \
                 --deployment-kind vllm


uv pip install matplotlib seaborn
python llm/plot_pareto.py --artifacts-root-dir artifacts_root --title "Dynamo vs. vLLM"

uv pip install vllm==0.7.3
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-3B-Instruct \
  --tokenizer Qwen/Qwen2.5-3B-Instruct \
  --max-model-len 8192 \
  --tensor-parallel-size 1

cd components/backends/vllm
bash launch/disagg_router.sh


https://huggingface.co/neuralmagic/DeepSeek-R1-Distill-Llama-70B-FP8-dynamic
https://github.com/ai-dynamo/dynamo/issues/402

vllm serve neuralmagic/DeepSeek-R1-Distill-Llama-70B-FP8-dynamic \
    --tensor-parallel-size 4

genai-perf profile -m neuralmagic/DeepSeek-R1-Distill-Llama-70B-FP8-dynamic \
    --tokenizer neuralmagic/DeepSeek-R1-Distill-Llama-70B-FP8-dynamic \
    --endpoint-type chat \
    --endpoint v1/chat/completions \
    --url 127.0.0.1:8000 \
    --streaming --concurrency 32 \
    --num-dataset-entries 128 \
    --warmup-request-count 128 \
    --request-count 128 \
    --synthetic-input-tokens-mean 3000 \
    --synthetic-input-tokens-stddev 0 \
    --output-tokens-mean 150 \
    --output-tokens-stddev 0 \
    --extra-inputs min_tokens:150 \
    --extra-inputs max_tokens:150 \
    --extra-inputs ignore_eos:true \
    --random-seed 0 \
    --artifact-dir concurrency_32 \
    --profile-export-file profile_export_concurrency_32.json \
    -- --max-threads 32


bash container/build.sh --framework VLLM
bash container/run.sh --framework VLLM -it


# etcd &
nats-server --js &
etcd --listen-client-urls http://0.0.0.0:2379 --advertise-client-urls http://0.0.0.0:2379 --data-dir /tmp/etcd &

dynamo serve graphs.disagg:Frontend -f <your_configuration>.yaml
dynamo serve graphs.disagg:Frontend -f agg.yaml

# container version 0.3.2
dynamo run in=http out=vllm neuralmagic/DeepSeek-R1-Distill-Llama-70B-FP8-dynamic

dynamo run in=http out=vllm neuralmagic/DeepSeek-R1-Distill-Llama-70B-FP8-dynamic --tensor-parallel-size=4

```

Final workstream:

```bash
docker run \
  -it --rm \
  --gpus all \
  -p 8000:8000 \
  -p 8080:8080 \
  -v /home/nvidia/.cache:/root/.cache \
  --ipc=host \
  --shm-size=2g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --cap-add=SYS_ADMIN \
  nvcr.io/nvidia/ai-dynamo/vllm-runtime:0.3.2 \
  bash

nats-server --js &
etcd --listen-client-urls http://0.0.0.0:2379 --advertise-client-urls http://0.0.0.0:2379 --data-dir /tmp/etcd &

dynamo run in=http out=vllm Qwen/Qwen2.5-3B-Instruct

sudo rm -rf artifacts_root

docker run \
  --gpus all \
  --rm -it \
  --net host \
  -v $PWD:/workspace/benchmarks \
  nvcr.io/nvidia/tritonserver:25.06-py3-sdk

python llm/perf.py --model Qwen/Qwen2.5-3B-Instruct \
                   --url http://localhost:8000 \
                   --mode aggregated \
                   --tp 1 \
                   --dp 1 \
                   --concurrency 1,2,4,8,16,32 \
                   --artifacts-root-dir artifacts_root

bash llm/perf.sh --model Qwen/Qwen2.5-3B-Instruct \
                 --url http://localhost:8080 \
                 --mode aggregated \
                 --tp 1 \
                 --dp 1 \
                 --concurrency 1,2,4,8,16,32 \
                 --artifacts-root-dir artifacts_root

rm -rf .venv
uv venv .venv --python 3.12 --seed
source .venv/bin/activate.fish
uv pip install matplotlib seaborn

uv pip install vllm==0.7.3
uv pip install vllm==0.10.0
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-3B-Instruct \
  --tokenizer Qwen/Qwen2.5-3B-Instruct \
  --max-model-len 8192 \
  --tensor-parallel-size 1

python llm/perf.py --model Qwen/Qwen2.5-3B-Instruct \
                   --url http://localhost:8000 \
                   --mode aggregated \
                   --tp 1 \
                   --dp 1 \
                   --concurrency 1,2,4,8,16,32 \
                   --artifacts-root-dir artifacts_root \
                   --deployment-kind vllm

bash llm/perf.sh --model Qwen/Qwen2.5-3B-Instruct \
                 --url http://localhost:8000 \
                 --mode aggregated \
                 --tp 1 \
                 --dp 1 \
                 --concurrency 1,2,4,8,16,32 \
                 --artifacts-root-dir artifacts_root \
                 --deployment-kind vllm


python llm/plot_pareto.py --artifacts-root-dir artifacts_root --title "Dynamo vs. vLLM"

dynamo run in=http out=vllm neuralmagic/DeepSeek-R1-Distill-Llama-70B-FP8-dynamic --tensor-parallel-size=4

bash llm/perf.sh --model neuralmagic/DeepSeek-R1-Distill-Llama-70B-FP8-dynamic \
                 --url http://localhost:8080 \
                 --mode aggregated \
                 --tp 4 \
                 --dp 1 \
                 --concurrency 1,2,4,8,16,32 \
                 --artifacts-root-dir artifacts_root

# TODO deploy on 0,1,2,3 and 4,5,6,7 and Nginx round robin
vllm serve neuralmagic/DeepSeek-R1-Distill-Llama-70B-FP8-dynamic \
    --tensor-parallel-size 4


bash llm/perf.sh --model neuralmagic/DeepSeek-R1-Distill-Llama-70B-FP8-dynamic \
                 --url http://localhost:8000 \
                 --mode aggregated \
                 --tp 4 \
                 --dp 1 \
                 --concurrency 1,2,4,8,16,32,64,128,256 \
                 --artifacts-root-dir artifacts_root \
                 --deployment-kind vllm

python llm/plot_pareto.py --artifacts-root-dir artifacts_root --title "Dynamo vs. vLLM Llama 70B FP8"

bash container/build.sh --framework VLLM
bash container/run.sh --framework VLLM -it


# etcd &
nats-server --js &
etcd --listen-client-urls http://0.0.0.0:2379 --advertise-client-urls http://0.0.0.0:2379 --data-dir /tmp/etcd &

cd components/backends/vllm
bash launch/disagg_router.sh

curl localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "neuralmagic/DeepSeek-R1-Distill-Llama-70B-FP8-dynamic",
    "messages": [
    {
        "role": "user",
        "content": "Hello, how are you?"
    }
    ],
    "stream":false,
    "max_tokens": 300
  }' | jq
```

```bash
#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -e

trap 'echo Cleaning up...; kill 0' EXIT

# run ingress
python -m dynamo.frontend --router-mode kv &

# routing will happen between the two decode workers
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m dynamo.vllm \
    --model neuralmagic/DeepSeek-R1-Distill-Llama-70B-FP8-dynamic \
    --enforce-eager \
    --tensor-parallel-size 4 &

CUDA_VISIBLE_DEVICES=4,5 python3 -m dynamo.vllm \
    --model neuralmagic/DeepSeek-R1-Distill-Llama-70B-FP8-dynamic \
    --enforce-eager \
    --is-prefill-worker \
    --tensor-parallel-size 2 &

CUDA_VISIBLE_DEVICES=6,7 python3 -m dynamo.vllm \
    --model neuralmagic/DeepSeek-R1-Distill-Llama-70B-FP8-dynamic \
    --enforce-eager \
    --is-prefill-worker \
    --tensor-parallel-size 2 &

sudo rm -rf artifacts_root

docker run \
  --gpus all \
  --rm -it \
  --net host \
  -v $PWD:/workspace/benchmarks \
  nvcr.io/nvidia/tritonserver:25.06-py3-sdk

cd benchmarks
bash llm/perf.sh --model neuralmagic/DeepSeek-R1-Distill-Llama-70B-FP8-dynamic \
                 --url http://localhost:8080 \
                 --mode disaggregated \
                 --prefill-tensor-parallelism 4 \
                 --prefill-data-parallelism 2 \
                 --decode-tensor-parallelism 4 \
                 --decode-data-parallelism 1 \
                 --concurrency 1,2,4,8,16,32,64,128,256 \
                 --artifacts-root-dir artifacts_root \
                 --deployment-kind dynamo


```
