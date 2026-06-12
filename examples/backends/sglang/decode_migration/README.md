# SGLang Decode-to-Decode Migration Prototype

This prototype migrates an actively decoding request from a fast SGLang worker
to a slow SGLang decode worker through Dynamo. It uses SGLang's current NIXL
prefill/decode transfer protocol, but treats the live decode worker as the KV
sender.

Every worker publishes its normal Dynamo model card, enables decode migration,
and exposes the same generation and migration endpoints. The destination first
reserves an opaque
NIXL room, then arms an actual parked SGLang decode request and receiver with
the source's exact KV frontier. The source rank is passed explicitly; it is not
encoded in the room ID.

The source remains authoritative until the prepared destination returns its
first output and the coordinator activates it. Only then is source state
released. A destination failure resumes the untouched source request. Client
cancellation aborts whichever side currently owns or retains state.

## Migration trigger policy

The transfer protocol is independent of the policy that decides when to migrate.
The prototype coordinator supports either a generated-token count or one or more
semantic output token IDs. If `--migrate-on-token-id` is present, it takes
precedence over `--migrate-after-tokens`.

For Qwen3 thinking requests, token ID `151668` is `</think>`. The coordinator
forwards that boundary token from the fast worker, then reserves and arms the
destination, snapshots the exact SGLang KV frontier, and performs the handoff. A
request that finishes before the boundary never enters migration. The
coordinator model card also carries `reasoning_parser=qwen3` and the configured
stream interval so Dynamo preserves reasoning/content parsing across the single
client stream.

## Run locally

The launcher uses GPU 0 for the aggregated source and GPU 1 for the decode
destination. It builds a small derived image with the SGLang kernel version
required by the checked-out source tree.

```bash
./examples/backends/sglang/decode_migration/run_container.sh
STREAM_INTERVAL=4 ./examples/backends/sglang/decode_migration/run_container.sh
```

Mixed-TP examples used for validation:

```bash
# Qwen3-8B, GQA KV reshaping through the heterogeneous-TP staging path
MODEL_ROOT=/root/models/qwen3-8b \
MODEL_PATH_IN_CONTAINER=/models/qwen3-8b \
SERVED_MODEL_NAME=Qwen/Qwen3-8B \
SOURCE_TP=4 DESTINATION_TP=1 \
SOURCE_GPUS=0,1,2,3 DESTINATION_GPUS=4 \
SGLANG_DISAGG_STAGING_BUFFER=1 \
./examples/backends/sglang/decode_migration/run_container.sh

# DeepSeek-V2-Lite, replicated MLA KV through the direct NIXL path on B200
MODEL_ROOT=/root/models/deepseek-v2-lite \
MODEL_PATH_IN_CONTAINER=/models/deepseek-v2-lite \
SERVED_MODEL_NAME=deepseek-ai/DeepSeek-V2-Lite \
SOURCE_TP=4 DESTINATION_TP=1 \
SOURCE_GPUS=0,1,2,3 DESTINATION_GPUS=4 \
ATTENTION_BACKEND=triton ENABLE_DETERMINISTIC_INFERENCE=1 \
./examples/backends/sglang/decode_migration/run_container.sh
```

A paired Qwen3-8B thinking/GSM8K check starts a TP4 source on GPUs 0-3,
a TP1 destination on GPU 4, and source-only plus migration coordinator model
aliases behind the same Dynamo frontend:

```bash
NUM_EXAMPLES=20 MAX_TOKENS=4096 \
./examples/backends/sglang/decode_migration/run_qwen3_gsm8k.sh
```

The harness uses SGLang's GSM8K prompt and scorer, enables Qwen thinking,
requires every migrated request to produce reasoning, and proves each handoff
from boundary, reservation, receiver arm, NIXL completion, activation, and
source commit log events. It writes paired JSON and HTML reports under
`RESULT_DIR`.

On June 12, 2026, the 20-example deterministic smoke run completed all 20
handoffs at `</think>`. Baseline and migrated accuracy were both 95%; hidden
reasoning matched exactly for 20/20 requests and extracted answers matched for
20/20. Final visible text matched exactly for 15/20; the other five retained the
same answer with small post-handoff wording differences from TP4 versus TP1
numerics. This is a focused paired regression check, not a statistically
authoritative full GSM8K result.

The threshold/lifecycle `run_container.sh` harness starts two model workers, one
Dynamo frontend, and three lightweight coordinator components exposed as
distinct model aliases:

- `decode-migration-baseline`: source-only deterministic reference
- `Qwen/Qwen3-0.6B`: normal decode migration
- `decode-migration-rollback`: injected destination-start failure and source resume

Logs are written to `/tmp/decode-migration-results/stream-N` on the host. The
test covers deterministic output equality, a request finishing before the
threshold, a request finishing immediately after handoff, rollback, client
disconnect, and a successful request after cancellation.

The test image includes a Dynamo runtime binding built from this worktree. If
the image does not exist, `build_image.sh` builds that wheel with `maturin` and
uses a temporary Docker build context. The host therefore needs Rust, `maturin`,
and `protoc` for a fresh image build.

## Prototype constraints

- Source overlap scheduling must be disabled.
- The source uses a scheduler-wide pause, so only one migration can be active.
- Source and destination must use compatible model, TP/PP layout, page size,
  KV dtype, and transfer backend.
- The destination is selected statically by the prototype coordinator.
- The transfer is one-shot; incremental range synchronization is the next step.
- Destination capacity accounting is represented by a session reservation; a
  production router lease and per-request scheduler accounting remain follow-up
  work.

The current B200 image cannot use its bundled FlashInfer for DeepSeek MLA
because it is one patch below SGLang's minimum version. FlashMLA dense decode is
SM90a-only. The verified DeepSeek-V2-Lite run therefore uses Triton attention.

See [ROADMAP.md](ROADMAP.md) for the state model and production path.
