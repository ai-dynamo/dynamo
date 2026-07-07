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
```

`run_pd_three_way.py` performs a true P/D comparison. It cold-starts two
independent upstream vLLM prefill/decode pairs, then starts one Dynamo
prefill/decode deployment on the same visible GPU. Every deployment receives
the same rendered requests and uses NIXL, distinct side-channel ports, and the
same hybrid-state layout. The comparison matrix is:

1. upstream P/D run 1 versus upstream P/D run 2;
2. Dynamo P/D versus upstream P/D run 1; and
3. Dynamo P/D versus upstream P/D run 2.

Prompt logprobs are composed at the P/D boundary: the prefill result supplies
prompt logprobs, while decode supplies generated tokens and generated
logprobs. Decode must not recompute prompt logprobs from transferred KV. The
full suite includes a request that enables both fields together and requires
both to be populated before comparing their values exactly.

The `full` suite runs the two-request smoke stage first, followed by two
sequential repetitions of nine longer text/VLM cases. This is the exact-parity
gate: the current Qwen hybrid kernels can change greedy tokens when batch
composition changes, even within upstream vLLM itself.

Pass `--max-concurrency 4` to append an exact concurrent-batching stress stage.
That mode intentionally reports any upstream/Dynamo drift instead of weakening
the comparison. The two-way runner writes beneath `logs/tito-parity/`; the
three-way runner writes metadata, requests, per-deployment logs, raw responses,
field-level checks, and summaries beneath `logs/tito-pd-three-way/`.
