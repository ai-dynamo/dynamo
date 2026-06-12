# SGLang Decode-to-Decode Migration Prototype

This prototype migrates an actively decoding request between ordinary SGLang
workers behind the normal Dynamo frontend. There is no coordinator model and no
alternate HTTP frontend.

## Architecture

Start Dynamo with `--router-mode kv --enable-decode-migration`. The Rust
`DecodeMigration` operator runs after preprocessing and before `PrefillRouter`.
It selects the source through the normal KV router, evaluates the request trigger
while forwarding the source stream, and directly dispatches the continuation to
the selected destination so the destination is not booked as a second unrelated
KV-router request.

Every migration-enabled SGLang worker keeps its normal model card and exposes:

- `generate`
- `migration_prepare`
- `migration_sync`
- `migration_finalize`

Workers publish generic taints such as `decode/fast` and `decode/slow`, plus
migration compatibility metadata. Every worker can source and receive a
migration. Aggregated workers initialize SGLang's existing decode-side NIXL
receiver queues only for requests carrying migration bootstrap metadata; normal
requests continue through the ordinary prefill path.

The transaction is:

1. Select and pin a source worker.
2. Forward source output until the single trigger fires.
3. Ask the destination to reserve an opaque bootstrap room.
4. Quiesce the source and obtain its exact committed KV frontier.
5. Arm a parked destination request with the room, explicit source rank, and
   continuation request.
6. Transfer the committed KV range with NIXL.
7. Activate the destination after its first valid output, then commit and release
   the source.

Before source commit, any failure aborts the destination and resumes the retained
source. Client cancellation is propagated to whichever source and destination
streams exist at that transaction stage.

## Request Policy

Migration is opt-in through `nvext.decode_migration`. The policy has independent
source and destination routing constraints and exactly one tagged trigger.

```json
{
  "nvext": {
    "decode_migration": {
      "source": {"required_taints": ["decode/fast"]},
      "destination": {"required_taints": ["decode/slow"]},
      "trigger": {"type": "sequence_length", "tokens": 256}
    }
  }
}
```

A semantic boundary can use a token trigger instead:

```json
{"type": "token_id", "token_id": 151668}
```

For Qwen3, token ID `151668` is `</think>`. The trigger implementation scans all
tokens in a coalesced stream chunk, so `--stream-interval > 1` does not skip the
boundary.

## Run Locally

The default harness uses Qwen3-0.6B on two GPUs:

```bash
./examples/backends/sglang/decode_migration/run_container.sh
STREAM_INTERVAL=4 ./examples/backends/sglang/decode_migration/run_container.sh
```

Select different free GPUs when needed:

```bash
SOURCE_GPUS=2 DESTINATION_GPUS=3 \
  ./examples/backends/sglang/decode_migration/run_container.sh
```

The black-box suite verifies:

- a request finishing before the trigger never migrates;
- migrated deterministic output matches a source-only baseline;
- stream intervals 1 and 4 preserve token accounting;
- a request finishing immediately after handoff completes correctly;
- cancellation before destination prepare and after commit clean up correctly;
- workers accept a new migrated request after cancellation;
- two requests crossing the trigger concurrently both complete correctly.

Logs are written under `RESULT_DIR/stream-N` (default:
`/tmp/decode-migration-results/stream-N`).

## Mixed TP

The transport state is range-based and preserves the existing heterogeneous-TP
NIXL staging path. A Qwen3-8B TP4 to TP1 run can use:

```bash
MODEL_ROOT=/root/models/qwen3-8b \
MODEL_PATH_IN_CONTAINER=/models/qwen3-8b \
SERVED_MODEL_NAME=Qwen/Qwen3-8B \
SOURCE_TP=4 DESTINATION_TP=1 \
SOURCE_GPUS=0,1,2,3 DESTINATION_GPUS=4 \
SGLANG_DISAGG_STAGING_BUFFER=1 \
  ./examples/backends/sglang/decode_migration/run_container.sh
```

## Qwen3 Thinking-Boundary Accuracy

The paired GSM8K harness sends each prompt once to the fast worker only and once
with migration triggered by Qwen3's `</think>` token. It requires an observed
migration commit before scoring the migrated response. A completion that reaches
`max_tokens` without `</think>` is skipped because no migration was attempted.

```bash
mkdir -p /tmp/qwen3-migration-gsm8k
cp /tmp/gsm8k-test.jsonl /tmp/qwen3-migration-gsm8k/gsm8k-test.jsonl

MODEL_ROOT=/root/models/qwen3-8b \
MODEL_PATH_IN_CONTAINER=/models/qwen3-8b \
SERVED_MODEL_NAME=Qwen/Qwen3-8B \
SOURCE_TP=4 DESTINATION_TP=1 \
SOURCE_GPUS=0,1,2,3 DESTINATION_GPUS=4 \
SGLANG_DISAGG_STAGING_BUFFER=1 \
ENABLE_DETERMINISTIC_INFERENCE=1 \
TEST_MODE=gsm8k \
GSM8K_NUM_QUESTIONS=20 GSM8K_MAX_ATTEMPTS=60 \
GSM8K_MAX_TOKENS=1536 \
GSM8K_DATA_PATH=/results/gsm8k-test.jsonl \
RESULT_DIR=/tmp/qwen3-migration-gsm8k \
  ./examples/backends/sglang/decode_migration/run_container.sh
```

The result is written to
`RESULT_DIR/stream-1/gsm8k_accuracy_results.json`. `questions` is the number of
completed migrated pairs; `attempted_questions` also includes skipped
completions without a thinking boundary.

## Prototype Constraints

- Exact source quiescence currently requires `--disable-overlap-schedule`.
- Source quiescence uses a scheduler-wide pause, so concurrent migrations may be
  serialized or one may fall back to its retained source.
- Transfer is currently one-shot. The prepare/sync/finalize lifecycle and
  explicit `[start, end)` range state are intended to support incremental KV
  synchronization without changing the frontend ownership protocol.
- Source and destination must have compatible model, page size, PP layout, KV
  dtype, and transfer backend. Heterogeneous TP requires a supported direct or
  staging transfer layout.
- Destination reservation is currently local worker state rather than a durable
  router lease with TTL.

See [ROADMAP.md](ROADMAP.md) for the state model and incremental-transfer path.
