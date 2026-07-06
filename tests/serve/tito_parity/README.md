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

python tests/serve/tito_parity/run_parity.py \
  --model Qwen/Qwen3.5-2B \
  --dynamo-topology disaggregated \
  --suite smoke

python tests/serve/tito_parity/run_parity.py \
  --model Qwen/Qwen3.5-35B-A3B-FP8 \
  --dynamo-topology disaggregated \
  --suite full
```

The disaggregated mode launches one prefill and one decode worker on the same
visible GPU, waits for both worker health endpoints and frontend discovery, and
then replays the same rendered requests. Worker logs in `dynamo.log` provide
the topology evidence: every request must enter both `[PREFILL]` and `[DECODE]`,
with NIXL KV-transfer parameters on the handoff.

The `full` suite runs the two-request smoke stage first, followed by two
sequential repetitions of eight longer text/VLM cases. This is the exact-parity
gate: the current Qwen hybrid kernels can change greedy tokens when batch
composition changes, even within upstream vLLM itself.

Pass `--max-concurrency 4` to append an exact concurrent-batching stress stage.
That mode intentionally reports any upstream/Dynamo drift instead of weakening
the comparison. Requests, server logs, raw responses, and failure diffs are
written beneath `logs/tito-parity/`.
