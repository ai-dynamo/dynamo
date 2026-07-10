# Token-in/token-out parity runner

This runner captures `GenerateRequest` payloads from upstream vLLM 0.23.0's
render endpoints, sends the identical token and multimodal feature payloads to
upstream vLLM and Dynamo, and compares every stable response field exactly.
Only the intrinsically generated top-level `request_id` value is normalized.

Run it from the persistent worktree container's `vllm` tmux session:

```bash
python tests/serve/tito_parity/run_parity.py \
  --model Qwen/Qwen3.5-2B \
  --suite smoke

python tests/serve/tito_parity/run_parity.py \
  --model Qwen/Qwen3.5-35B-A3B-FP8 \
  --suite full

python tests/serve/tito_parity/run_pd_three_way.py \
  --model Qwen/Qwen3.5-2B \
  --suite smoke

python tests/serve/tito_parity/run_pd_three_way.py \
  --model Qwen/Qwen3.5-35B-A3B-FP8 \
  --suite full

# One mixed-reward VLM cohort, replayed unchanged through native vLLM,
# Dynamo aggregated, and Dynamo P/D.
python tests/serve/tito_parity/run_seeded_rl_parity.py \
  --model Qwen/Qwen3-VL-4B-Instruct \
  --seed-base 144

# Scrape and verify Generate frontend metrics in each Dynamo topology.
python tests/serve/tito_parity/verify_generate_metrics.py \
  --model Qwen/Qwen3.5-2B \
  --topology aggregated

python tests/serve/tito_parity/verify_generate_metrics.py \
  --model Qwen/Qwen3.5-2B \
  --topology disaggregated

# Qwen3.5 supports video input.
python tests/serve/tito_parity/run_pd_three_way.py \
  --model Qwen/Qwen3.5-2B \
  --suite video

# Gemma 4 supports audio input. vLLM needs its audio extras (av and soundfile).
python tests/serve/tito_parity/run_pd_three_way.py \
  --model google/gemma-4-E2B-it \
  --suite audio
```

`run_pd_three_way.py` performs a true P/D comparison. It cold-starts two
independent `vllm serve` prefill/decode pairs to establish that the upstream
reference is reproducible, then starts one `dynamo.vllm` prefill/decode
deployment on the same visible GPU. Every deployment receives the same
rendered requests and uses NIXL, distinct side-channel ports, and the same
hybrid-state layout. The checks are:

1. the repeated `vllm serve` result is exact;
2. `dynamo.vllm` is exact against each upstream reference.

Prompt logprobs are composed at the P/D boundary: the prefill result supplies
prompt logprobs, while decode supplies generated tokens and generated
logprobs. Decode must not recompute prompt logprobs from transferred KV. The
full suite includes a request that enables both fields together and requires
both to be populated before comparing their values exactly.

The `full` suite runs the two-request smoke stage first, followed by two
sequential repetitions of nine longer text/VLM cases. This is the exact-parity
gate: the current Qwen hybrid kernels can change greedy tokens when batch
composition changes, even within upstream vLLM itself.

`run_seeded_rl_parity.py` reproduces task 7 from `color-codeword-v1` and gives
the 16 rollouts distinct, fixed sampling seeds. Native vLLM's render endpoint
produces every multimodal token/feature request once; those exact request bodies
are then replayed through Dynamo aggregated and Dynamo P/D. The runner requires
a mixed correct/incorrect reward vector, exact tokens and stable metadata, and
logprob equality within a narrow numerical tolerance. Request IDs and the known
Python-null/Rust-populated completion-logprob `bytes` representation are
normalized. It requests zero alternative logprobs, which still returns each
sampled token's logprob and avoids a known Python/Rust formatter difference in
the number of alternatives returned for a non-top-1 sampled token. Each replay
also verifies that the render result contains the VLM feature tensors, hashes,
and placeholders, and that the server remains healthy after the request wave.

Pass `--max-concurrency 4` to append an exact concurrent-batching stress stage.
That mode intentionally reports any upstream/Dynamo drift instead of weakening
the comparison. The aggregated runner writes beneath `logs/tito-parity/`; the
P/D runner writes metadata, requests, per-deployment logs, raw responses,
field-level checks, and summaries beneath `logs/tito-pd-three-way/`.

`verify_generate_metrics.py` enables prefix caching, sends deterministic text
and VLM Generate requests, and verifies Prometheus deltas for request lifecycle,
input/output tokens, TTFT/ITL, cached tokens, live queue/active gauges, and
P/D per-worker timing attribution. Its P/D run opts the existing launcher into
KV routing so the request tracker receives both selected worker IDs; ordinary
parity runs retain their default router. The verifier also checks that the
token-in/token-out path does not report tokenizer latency. Artifacts are written
beneath `logs/tito-generate-metrics/`.
