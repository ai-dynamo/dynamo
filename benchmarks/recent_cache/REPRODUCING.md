<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Reproduce recent-cache experiments

Run these commands from the selected Dynamo worktree, using its `.venv`.
Generated traces and reports belong outside the repository. No dataset is
included here.

For the recorded headline numbers, use commit
`c3266067365b53630cdaf4fd6a29ba95775385c3` before building the worktree's environment.
That revision pins AISimulate 0.12.0.dev1. The later PR ancestry merge includes an
upstream 0.12.0.dev2 bump; running the current head evaluates that newer dependency,
not an exact reproduction of the recorded campaign.

The first-32-request example below introduces the tools. Its 300-second TTL is
not the main cache-pressure experiment. For that comparison, start with the
first **128 requests, four shared copies, 20 cycles, a 60-second period, and a
15-second TTL**, using the recipes under [Experiment workloads](#experiment-workloads).

The [public Mooncake trace at revision
`d21da178bae8db9651cf18a76824c084145fc725`](https://github.com/kvcache-ai/Mooncake/blob/d21da178bae8db9651cf18a76824c084145fc725/mooncake_trace.jsonl)
contains 23,608 requests over one hour and uses 512-token trace blocks. Its
SHA256 is `b434f1816a707f4bac697235588184ebc374c9907cb981bb65fb0643471fe711`.

```bash
scratch=$(mktemp -d /tmp/dynamo-recent-cache.XXXXXX)
source_url=https://raw.githubusercontent.com/kvcache-ai/Mooncake/d21da178bae8db9651cf18a76824c084145fc725/mooncake_trace.jsonl
curl --fail --location "$source_url" --output "$scratch/mooncake.jsonl"
shasum -a 256 "$scratch/mooncake.jsonl"

for sharing in shared private; do
  .venv/bin/python benchmarks/recent_cache/make_workload.py \
    --source "$scratch/mooncake.jsonl" --source-url "$source_url" \
    --output-dir "$scratch/$sharing" \
    --first 32 --copies 4 --cycles 20 --period-seconds 60 \
    --copy-offset-ms 10 --sharing "$sharing" \
    --trace-block-size 512 --engine-block-size 64
done
```

Each output directory contains `trace.jsonl` and `manifest.json`. The manifest
records the source and output SHA256 values, selection, configuration, request
count, arrival span, and prefix counts. Existing output directories are rejected.
The source file remains untouched.

The generator preserves prompt/output lengths, other request fields, and all
arrival gaps within the selected window. It normalizes the window start to zero,
overlays the copies with the specified arrival offsets, and repeats the window.
The period must accommodate the entire window plus the last copy's offset.
The default produces 2,560 requests over more than 19 minutes, exceeding a
five-minute TTL. Prefixes touched every minute can nevertheless remain in the
recent set throughout the run. A longer period can exercise expiration between
cycles. Completing a long workload does not itself establish useful cache
pressure or a policy improvement.

`shared` keeps source hash IDs unchanged across copies and cycles. `private`
assigns each copy its own consistent hash namespace, preserving every prefix
equality within that copy across cycles. These variants have identical request
lengths and arrival times. They differ in cross-copy sharing and total distinct
working set. Both intentionally reuse content across cycles. Request and session
IDs, when supplied, become distinct per copy and cycle so separate repetitions
are not interpreted as one continuing session. Requests also include their
source-row index to guarantee unique IDs.

This generator accepts ordinary Mooncake rows with absolute millisecond
`timestamp` or `created_time` fields and complete `hash_ids` coverage. It rejects
agentic dependency traces and rows with only relative delays. Do not use it to
flatten Weka/AgentX trajectories or to truncate a dependency graph.

The manifest's chained complete-prefix count is exact at the **source trace
block size**: the same raw hash ID following different prefixes counts as
different blocks. Private copies multiply this count; cycles do not. The
engine-block estimate only scales that count by the trace/engine block-size
ratio. For 512-token source blocks and 64-token engine blocks, that is an
approximate eightfold conversion. It excludes partial source blocks, generated
outputs, live-request allocations, and placement-dependent replication. Use
measured engine/cache telemetry for capacity conclusions. `--engine-block-size`
only labels this estimate; it does not configure a replay engine.

Replay these derived traces with `trace_format: mooncake`,
`trace_block_size: 512`, and `arrival_speedup_ratio: 1.0` in the case manifest
consumed by `benchmarks/replay_recent_cache.py`. Keep engine settings identical
across policy comparisons, declare both the per-worker and aggregate cache
budget, and retain the generator manifest alongside the replay artifacts.

## Build and generate comparison cases

Build the bindings with AIC timing and the offline builtin policy adapter:

```bash
VIRTUAL_ENV="$PWD/.venv" .venv/bin/maturin develop --uv --release \
  --manifest-path lib/bindings/python/Cargo.toml \
  --features aic-forward-pass,replay-builtin
```

Use these features consistently for every policy. Do not enable `replay-bench`
for this comparison: that feature also changes replay identity and deterministic
selector plumbing. The normal selector is stochastic, and these reproducers do
not configure a seed. Repeat close comparisons and retain each independent run.

Supply the engine arguments as a JSON file. For example, the following is an
explicit Qwen3-32B/vLLM/H200 timing configuration with a 3,904-block per-worker
cache and 64-token engine blocks. It is an experiment configuration, not a
recommendation for arbitrary workloads. Ensure the selected AIC performance
data is available and the cache/model limits can admit the longest request.

```bash
cat > "$scratch/engine.json" <<'JSON'
{
  "block_size": 64,
  "aic_backend": "vllm",
  "aic_backend_version": "0.24.0",
  "aic_system": "h200_sxm",
  "aic_model_path": "Qwen/Qwen3-32B",
  "aic_tp_size": 1,
  "speedup_ratio": 1.0,
  "num_gpu_blocks": 3904
}
JSON

.venv/bin/python benchmarks/recent_cache/make_cases.py \
  --trace "$scratch/shared/trace.jsonl" --format mooncake \
  --expected-requests 2560 --workers 4 --ttl-seconds 300 \
  --threshold 0.5 --arrival-speedup 1 --trace-block-size 512 \
  --engine-args "$scratch/engine.json" --output-dir "$scratch/cases-shared"
```

The generator writes `cases.json`, `two-tier.yaml`, an engine configuration
snapshot, and `provenance.json` with input hashes and all mode parameters. It
does not copy the input trace or run a simulation, and rejects existing output
directories. For Weka, provide a complete single-model trace directory with
`--format weka --trace-block-size 64 --agentic-lanes N` and its total model-request
count. The recorded trace timing and graph remain unchanged.

The nine cases are:

| Cases | Overlap credit | TTL prediction | Selection |
| --- | ---: | --- | --- |
| `stock-default-credit1` | 1 | Off | Normal cost |
| `prediction-only-credit1` | 1 | On | Normal cost |
| `static-credit4`, `static-credit16`, `static-credit64` | 4 / 16 / 64 | On | Fixed credit |
| `hybrid-credit4`, `hybrid-credit16`, `hybrid-credit64` | 1 below threshold; 4 / 16 / 64 above | On | Adaptive credit |
| `smg` | Builtin two-tier policy | Off | Cache threshold 0.5; balance thresholds 32 and 1.1 |

Every case records recent-cache and residency telemetry. Observe mode with
prediction off does not change stock or SMG routing. Static and adaptive cases
use identical TTL prediction so their comparison measures the adaptive choice
of credit. SMG's own cache threshold stays at 0.5 even if `--threshold` changes
the adaptive policy's pressure threshold.

## Experiment workloads

The following commands reuse `$scratch`, `$source_url`, the downloaded source,
and the TP1 engine JSON above. They generate input and case manifests; run the
chosen manifests separately using the logging configuration in the next section.

### Repeated shared Mooncake windows

These are derived stress workloads: copies share prefixes, and each cycle
repeats the selected source window. They are not additional independent
production traces. All three use four TP1 workers, 3,904 engine blocks per
worker, a 0.5 pressure threshold, and a 15-second TTL. The generator retains
the source window's internal arrival gaps.

```bash
for spec in '128 60' '140 60' '128 45'; do
  read -r first period <<< "$spec"
  workload="shared${first}-p${period}"
  requests=$((first * 4 * 20))

  .venv/bin/python benchmarks/recent_cache/make_workload.py \
    --source "$scratch/mooncake.jsonl" --source-url "$source_url" \
    --output-dir "$scratch/$workload" \
    --first "$first" --copies 4 --cycles 20 --period-seconds "$period" \
    --copy-offset-ms 10 --sharing shared \
    --trace-block-size 512 --engine-block-size 64

  .venv/bin/python benchmarks/recent_cache/make_cases.py \
    --trace "$scratch/$workload/trace.jsonl" --format mooncake \
    --expected-requests "$requests" --workers 4 --ttl-seconds 15 \
    --threshold 0.5 --arrival-speedup 1 --trace-block-size 512 \
    --engine-args "$scratch/engine.json" \
    --output-dir "$scratch/cases-$workload"
done
```

The 128-request variants contain 10,240 requests each; the 140-request variant
contains 11,200. The 45-second period changes offered demand while preserving
the selected requests and sharing pattern. Compare policies within each
workload before comparing results across periods or window sizes.

### Full original Mooncake trace

This uses all 23,608 source requests without duplication or truncation. Arrival
speedup 1.2 compresses the one-hour arrival schedule to 50 minutes; engine
speedup remains 1.0. Use eight TP1 workers and TTL 25 seconds:

```bash
.venv/bin/python benchmarks/recent_cache/make_cases.py \
  --trace "$scratch/mooncake.jsonl" --format mooncake \
  --expected-requests 23608 --workers 8 --ttl-seconds 25 \
  --threshold 0.5 --arrival-speedup 1.2 --trace-block-size 512 \
  --engine-args "$scratch/engine.json" \
  --output-dir "$scratch/cases-full-mooncake"
```

### Public Weka/AgentX subset and private copies

`fetch_weka.py` reproduces seven complete plays from the pinned
[public capped Weka corpus](https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126-256k/tree/8fecd2fc56694469f758f0afbbb6335ad3043740),
checking the dataset revision and each selected play's SHA256. The selection
contains 443 model requests and nine explicit subagent groups. It selects
supported single-model plays among the first 32 source rows and excludes two
idle-heavy plays. It is a partial, deliberately selected sample, not a
representative evaluation of the full AgentX corpus. Original request lengths,
prefix identities, nested requests, and timing gaps remain intact.

```bash
.venv/bin/python benchmarks/recent_cache/fetch_weka.py \
  --output-dir "$scratch/weka7"

.venv/bin/python benchmarks/recent_cache/fetch_weka.py \
  --source-dir "$scratch/weka7/plays" --copies 4 \
  --output-dir "$scratch/weka4"

cat > "$scratch/engine-weka.json" <<'JSON'
{
  "block_size": 64,
  "aic_backend": "vllm",
  "aic_backend_version": "0.24.0",
  "aic_system": "h200_sxm",
  "aic_model_path": "Qwen/Qwen3-32B",
  "aic_tp_size": 2,
  "speedup_ratio": 1.0,
  "num_gpu_blocks": 11742
}
JSON

for ttl in 60 120; do
  .venv/bin/python benchmarks/recent_cache/make_cases.py \
    --trace "$scratch/weka4/plays" --format weka \
    --expected-requests 1772 --workers 2 --agentic-lanes 28 \
    --ttl-seconds "$ttl" --threshold 0.5 --arrival-speedup 5 \
    --trace-block-size 64 --engine-args "$scratch/engine-weka.json" \
    --output-dir "$scratch/cases-weka4-t$ttl"
done
```

Four private copies produce 28 plays and 1,772 requests. Distinct filenames
give each copy a separate cache namespace while retaining intra-play reuse.
This multiplies synthetic demand; the copies are not independent observations.
The source graph is preserved, and arrival speedup 5 scales its timing rather
than flattening its dependencies. Replay the `plays/` subdirectory, not the
parent directory containing download metadata.

For the seven original plays at source timing, use seven lanes, speedup 1, and
443 expected requests. The earlier native-subset reference used TTL 300:

```bash
.venv/bin/python benchmarks/recent_cache/make_cases.py \
  --trace "$scratch/weka7/plays" --format weka \
  --expected-requests 443 --workers 2 --agentic-lanes 7 \
  --ttl-seconds 300 --threshold 0.5 --arrival-speedup 1 \
  --trace-block-size 64 --engine-args "$scratch/engine-weka.json" \
  --output-dir "$scratch/cases-weka7-native"
```

Use separate output directories when varying this reference to TTL 60 or 120.
Both Weka variants simulate two TP2 workers, each with 11,742 engine blocks.
Qwen3-32B/vLLM/H200 provides the **timing proxy** for recorded Claude traffic;
these are comparisons of routing under that proxy, not measured Claude serving
performance or an official AgentX benchmark submission.

## Run and interpret the comparison

The scheduler debug events make engine preemption counts available to the
runner. Preserve this logging configuration when comparing those counts:

```bash
unset DYNAMO_SKIP_PYTHON_LOG_INIT
DYN_LOG='warn,aisimulate_core::engine::scheduler::vllm::core=debug' \
DYN_LOGGING_CONSOLE_FORMAT=jsonl \
  .venv/bin/python benchmarks/replay_recent_cache.py \
    --manifest "$scratch/cases-shared128-p60/cases.json" \
    --output "$scratch/results-shared128-p60" --parallel 4 --timeout-seconds 1800
```

Compare each adaptive candidate with **the best static credit among
1, 4, 16, and 64, and with SMG**, alongside stock routing. A gain over stock alone
does not establish an adaptive advantage. Check completed request counts before
comparing reuse, throughput, TTFT/tail latency, replication, or preemption.
Separate source-block estimates, recent TTL demand, and actual resident cache
measurements. Retain raw logs and provenance locally; publish code and concise
results without including downloaded traces or raw replay artifacts.

## Export aggregate results

After a run, export selected aggregate columns with the sanitizing helper:

```bash
.venv/bin/python benchmarks/recent_cache/summarize.py \
  --runs "$scratch/results-shared128-p60" \
  --output "$scratch/aggregate.csv"
```

`--runs` accepts multiple existing runner directories, or a parent containing
several groups. Pass only result directories you intend to summarize. The
export reads runner manifests and aggregate `row.json` files; it omits raw
traces, request records, logs, host details, full input paths, and original error
messages. Missing optional metrics remain blank, while measured zeros remain
zero. Review the aggregate CSV before publishing it. Preserve the local raw
artifacts for audit, and keep downloaded traces and raw results out of commits.
