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

```bash
python generate_trace_from_prompts.py prompts.jsonl output.jsonl 4
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
cargo run --bin dynamo-run in=http out=vllm deepseek-ai/DeepSeek-R1-Distill-Llama-8B
```

```bash
bash llm/perf.sh --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
                 --url http://localhost:8000 \
                 --mode aggregated \
                 --tp 2 \
                 --dp 2 \
                 --concurrency 1,2,4,8,16,32 \
                 --artifacts-root-dir artifacts_root
```

