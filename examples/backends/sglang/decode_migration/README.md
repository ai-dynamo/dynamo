# SGLang Decode-to-Decode Migration Prototype

This example migrates an active decode request between ordinary SGLang workers
behind Dynamo's normal frontend. Workers keep their public model cards and
normal `generate` endpoint; there is no coordinator model or alternate HTTP
frontend.

## Architecture

Start the frontend with:

```text
--router-mode kv --enable-decode-migration
```

The Rust migration operator runs after preprocessing and before
`PrefillRouter`. It selects and pins the source, forwards its stream until the
trigger, selects the destination, and dispatches the prepared continuation
directly to that worker.

Every migration-enabled SGLang worker exposes `generate`,
`migration_prepare`, `migration_sync`, and `migration_finalize`. Generic taints
such as `decode/fast` and `decode/slow` describe scheduling policy; every worker
can send and receive.

The source remains authoritative until the destination produces valid output
and the source commit succeeds. Earlier failures abort the destination and
resume the retained source. Cancellation follows both streams during handoff.

## Request Policy

Migration is opt-in through `nvext.decode_migration`:

```json
{
  "nvext": {
    "decode_migration": {
      "source": {"required_taints": ["decode/fast"]},
      "destination": {"required_taints": ["decode/slow"]},
      "trigger": {"type": "token_id", "token_id": 151668}
    }
  }
}
```

Supported triggers are `token_id` and `sequence_length`. Qwen3 token `151668`
is `</think>`. Trigger scanning covers every token in a coalesced stream chunk,
including when `--stream-interval` is greater than one.

## Run

The default two-GPU Qwen3-0.6B suite covers deterministic parity, stream
intervals, finish races, cancellation, cleanup, and concurrent triggers:

```bash
./examples/backends/sglang/decode_migration/run_container.sh
STREAM_INTERVAL=4 \
  ./examples/backends/sglang/decode_migration/run_container.sh
```

Choose GPUs with `SOURCE_GPUS` and `DESTINATION_GPUS`. Results are written to
`RESULT_DIR/stream-N`.

For Qwen3-8B TP4 to TP1:

```bash
MODEL_ROOT=/root/models/qwen3-8b \
MODEL_PATH_IN_CONTAINER=/models/qwen3-8b \
SERVED_MODEL_NAME=Qwen/Qwen3-8B \
SOURCE_TP=4 DESTINATION_TP=1 \
SOURCE_GPUS=0,1,2,3 DESTINATION_GPUS=4 \
SGLANG_DISAGG_STAGING_BUFFER=1 \
  ./examples/backends/sglang/decode_migration/run_container.sh
```

To run the included paired GSM8K check, add `TEST_MODE=gsm8k`,
`GSM8K_DATA_PATH=/results/gsm8k-test.jsonl`, and the desired
`GSM8K_NUM_QUESTIONS` and `GSM8K_MAX_TOKENS`.

## Constraints

- SGLang currently requires `--disable-overlap-schedule` for exact quiescence.
- Source quiescence pauses the worker scheduler. Dynamo queues concurrent
  migrations per source rank.
- Transfer is one-shot; the range-based prepare/sync/finalize protocol is
  designed to extend to incremental synchronization.
- Destination reservation is not yet a durable KV-capacity lease.
- Source and destination need compatible model, page size, PP layout, KV
  dtype/layout, and NIXL transport. Heterogeneous TP requires a supported direct
  or staging path.
- Current live coverage is DP=1 and excludes advanced generation state.

See [ROADMAP.md](ROADMAP.md) for the remaining upstream work.
