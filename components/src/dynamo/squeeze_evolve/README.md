<!-- # SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 -->

# `dynamo.squeeze_evolve` (experimental)

> **Experimental: not a released component.** Run it from a source checkout (see [Install](#install)); the CLI flags and `--tiers` schema are unstable and will change.

A standalone Dynamo component for **multi-model orchestration**: it serves [Squeeze-Evolve](https://arxiv.org/abs/2604.07725) (verifier-free evolutionary test-time scaling) as a chat model backed by any number of model tiers, from small cheap models up to large expensive ones. Each `/v1/chat/completions` request becomes one Squeeze-Evolve *problem*: an evolutionary loop generates a population of candidates and, each round, routes every group to the cheapest tier that can still handle it. Easy groups stay on the cheap models and only the hardest reach the most expensive, so the fleet matches or beats single-model accuracy at a fraction of the cost.

## How it works

Each tier is a separate `dynamo.vllm` deployment, ordered cheapest to most expensive, and the component owns one `KvRouter` per tier.

- **Loop 0**: the most expensive tier generates `--population` candidates.
- **Loops 1..T**: select `--groups` random `--k`-subsets, score each by answer diversity (number of unique answers), route each group to one of the tiers by its diversity (lowest-diversity consensus groups to the cheapest tier, highest-diversity to the most expensive), then recombine each tier's groups in parallel and replace the population.

The first candidate of the final population is the answer.

## Install

Run it from a source checkout. Squeeze-Evolve is experimental and ships no wheel extra, so install its two dependencies (`numpy` for the percentile routing thresholds, `uvloop` for the entrypoint's event loop) explicitly, alongside the vLLM extra for the tier workers:

```bash
git clone https://github.com/ai-dynamo/dynamo
cd dynamo
(cd lib/bindings/python && maturin develop --uv)   # build the Rust bindings
uv pip install -e '.[vllm]' numpy uvloop
```

## Usage

Each command below is a long-running process, so run them in separate terminals (or background them with `&` as shown). Start one `dynamo.vllm` worker per tier, each pinned to its own GPU with `CUDA_VISIBLE_DEVICES`, then the orchestrator, then the frontend. This example uses two tiers; add more tier processes and `--tiers` entries for N tiers (and pass N-1 `--confidence-percentiles`).

```bash
# Tier 0 (cheap) on GPU 0
CUDA_VISIBLE_DEVICES=0 python -m dynamo.vllm --model Qwen/Qwen3-4B-Instruct-2507 \
    --endpoint dynamo.tier1.generate \
    --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:20080","enable_kv_cache_events":true}' &

# Tier 1 (expensive) on GPU 1
CUDA_VISIBLE_DEVICES=1 python -m dynamo.vllm --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --endpoint dynamo.tier2.generate \
    --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:20081","enable_kv_cache_events":true}' &

# Orchestrator: registers the chat model "squeeze-evolve/aime25"
python -m dynamo.squeeze_evolve \
    --tiers '[{"endpoint":"dynamo.tier1.generate","model":"Qwen/Qwen3-4B-Instruct-2507"},
              {"endpoint":"dynamo.tier2.generate","model":"Qwen/Qwen3-30B-A3B-Instruct-2507"}]' \
    --model-name squeeze-evolve/aime25 --confidence-percentiles 50 &

# Frontend
python -m dynamo.frontend --http-port 8000 --router-mode round-robin
```

`--tiers` is a JSON array, cheapest first; each entry needs `endpoint` and `model`, plus optional `temperature` / `top_p` / `max_tokens` / `block_size` / `tokenizer` / `trust_remote_code`. For N tiers, pass N-1 `--confidence-percentiles`. The model card served to clients defaults to the most expensive tier's model; pass `--model-path` to override. Run `python -m dynamo.squeeze_evolve --help` for the other knobs.

Then call `--model-name` like any chat model:

```bash
curl localhost:8000/v1/chat/completions -H 'content-type: application/json' -d '{
  "model": "squeeze-evolve/aime25",
  "messages": [{"role": "user", "content": "Find the remainder when 7^100 is divided by 13."}]
}'
```

## Citation

If you use this package for research, please cite the Squeeze-Evolve paper:

```bibtex
@misc{maheswaran2026squeezeevolveunifiedmultimodel,
      title={Squeeze Evolve: Unified Multi-Model Orchestration for Verifier-Free Evolution},
      author={Monishwaran Maheswaran and Leon Lakhani and Zhongzhu Zhou and Shijia Yang and Junxiong Wang and Coleman Hooper and Yuezhou Hu and Rishabh Tiwari and Jue Wang and Harman Singh and Qingyang Wu and Yuqing Jian and Ce Zhang and Kurt Keutzer and Tri Dao and Xiaoxia Wu and Ben Athiwaratkun and James Zou and Chenfeng Xu},
      year={2026},
      eprint={2604.07725},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2604.07725},
}
```
