# Frontend Performance Profiling

Unified observability and benchmarking suite for Dynamo frontend performance.

## Quick Start

```bash
cd ~/dev/dynamo
source dynamo/bin/activate

# Single run (mocker + frontend + aiperf + Prometheus)
cd benchmarks/frontend/scripts
./run_perf.sh --model Qwen/Qwen3-0.6B --concurrency 32 --num-requests 640 \
    --speedup-ratio 0 --skip-bpf --skip-nsys --skip-flamegraph --skip-perf

# Sweep (multiple config points)
python3 sweep_runner.py --tokenizers hf --concurrency 32 --isl 512 \
    --benchmark-duration 30 --speedup-ratio 0 \
    -- --skip-bpf --skip-nsys --skip-flamegraph --skip-perf
```

## Architecture

```text
sweep_runner.py          Python sweep orchestrator
  |
  +-- run_perf.sh        Per-run shell harness (one invocation per config point)
       |
       +-- Step 0: Start etcd + NATS (or reuse existing)
       +-- Step 1: Start mocker workers (N models x M workers)
       +-- Step 2: Start frontend (optionally under nsys)
       +-- Step 3: Wait for /v1/models readiness
       +-- Step 4: Parallel captures (perf stat, BPF, flamegraph, /proc, Prometheus)
       +-- Step 5: aiperf load
       +-- Step 6: Wait for captures (deadline timeout)
       +-- Step 7: Final Prometheus snapshot, nsys export
       +-- Step 8: Save config.json
       |
       +-- bpf/run.sh              BPF tracing (10 bpftrace scripts)
       +-- flamegraph/*.sh          CPU + off-CPU flamegraphs
       +-- analysis/create_report.py  Post-hoc markdown report
```

## Prerequisites

| Tool | Required | Install |
|------|----------|---------|
| etcd | Yes | `apt install etcd` or [releases](https://github.com/etcd-io/etcd/releases) |
| nats-server | Yes | `apt install nats-server` or [nats.io](https://nats.io/download/) |
| aiperf | Yes | `uv pip install "git+https://github.com/ai-dynamo/aiperf.git@main"` (in dynamo venv) |
| jq | Yes | `apt install jq` |
| perf | Optional | `apt install linux-tools-$(uname -r)` |
| bpftrace | Optional | `apt install bpftrace` (needs root or CAP_BPF + CAP_PERFMON) |
| inferno | Optional | `cargo install inferno` (for flamegraphs) |
| nsys | Optional | NVIDIA Nsight Systems |

## sweep_runner.py

The main entry point for running performance sweeps. Iterates over a grid of
configurations and delegates each point to `run_perf.sh`.

### Basic Usage

```bash
# Smoke test (1 run)
python3 sweep_runner.py --tokenizers hf --concurrency 32 --isl 512 \
    --benchmark-duration 30 --speedup-ratio 0 \
    -- --skip-bpf --skip-nsys --skip-flamegraph --skip-perf

# Full tokenizer comparison
python3 sweep_runner.py --tokenizers hf,fastokens \
    --concurrency 32,64 --isl 512,1024,2048 \
    --benchmark-duration 60 --speedup-ratio 0

# Transport saturation (vary workers and request count)
python3 sweep_runner.py --tokenizers hf --concurrency 4096 \
    --num-requests 16384,32768 --workers 1,2,4,8 --speedup-ratio 0

# Multi-model (2 model instances, 2 workers each)
python3 sweep_runner.py --tokenizers hf --concurrency 32 --isl 512 \
    --num-models 2 --workers 2 --benchmark-duration 30 --speedup-ratio 0

# Preview sweep plan without running
python3 sweep_runner.py --dry-run --tokenizers hf,fastokens \
    --concurrency 32,64 --isl 512,1024
```

### With Profilers

```bash
# With perf stat + flamegraphs (no root needed)
python3 sweep_runner.py --tokenizers hf --concurrency 64 --isl 1024 \
    --benchmark-duration 60 --speedup-ratio 0

# With everything including BPF (needs sudo)
sudo -E python3 sweep_runner.py --tokenizers hf --concurrency 64 --isl 1024 \
    --benchmark-duration 60 --speedup-ratio 0

# nsys profiling (needs nsys in PATH)
python3 sweep_runner.py --tokenizers hf --concurrency 64 --isl 1024 \
    --benchmark-duration 60 --speedup-ratio 0 \
    -- --nsys-path /opt/nvidia/nsight-systems/bin/nsys
```

Profiler controls are passed through to run_perf.sh after `--`:

| Flag | Effect |
|------|--------|
| `--skip-bpf` | Skip BPF tracing |
| `--skip-nsys` | Skip Nsight Systems |
| `--skip-flamegraph` | Skip CPU/off-CPU flamegraphs |
| `--skip-perf` | Skip perf stat hardware counters |

### All Options

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | `Qwen/Qwen3-0.6B` | HF model path |
| `--backend` | `mocker` | Engine: `mocker` (synthetic) or `vllm` |
| `--tokenizers` | `hf,fastokens` | Comma-separated tokenizer backends |
| `--concurrency` | `50,100,200` | Comma-separated concurrency levels |
| `--isl` | `512,1024,2048` | Comma-separated input sequence lengths |
| `--osl` | `256` | Output sequence length |
| `--workers` | `2` | Comma-separated worker counts per model |
| `--num-models` | `1` | Number of model instances (each gets `--workers` workers) |
| `--aiperf-targets` | `first` | `first`: model-1 only. `all`: run aiperf for each model |
| `--speedup-ratio` | `1.0` | Mocker speedup (0 = infinite) |
| `--benchmark-duration` | `60` | aiperf run duration (seconds) |
| `--num-requests` | - | Comma-separated request counts (overrides duration) |
| `--output-dir` | auto | Output directory |
| `--max-consecutive-fails` | `2` | Skip remaining ISLs after N failures |
| `--cooldown` | `3` | Seconds between runs |
| `--dry-run` | - | Print plan without executing |
| `--no-report` | - | Skip per-run report generation |

## run_perf.sh

Low-level per-run harness. Normally called by sweep_runner.py, but can be
used directly for single runs.

```bash
# Minimal (no profilers)
./run_perf.sh --model Qwen/Qwen3-0.6B --concurrency 32 --num-requests 640 \
    --speedup-ratio 0 --skip-bpf --skip-nsys --skip-flamegraph --skip-perf

# Full observability (needs sudo for BPF)
sudo -E ./run_perf.sh --model Qwen/Qwen3-0.6B --concurrency 64 \
    --benchmark-duration 60 --speedup-ratio 0

# Multi-model
./run_perf.sh --model Qwen/Qwen3-0.6B --num-models 2 --workers 2 \
    --concurrency 32 --benchmark-duration 30 --speedup-ratio 0 \
    --skip-bpf --skip-nsys --skip-flamegraph --skip-perf
```

## Analyzing Results

```bash
# Per-run report (generated automatically by sweep_runner.py)
python3 analysis/create_report.py analyze artifacts/sweep_<ts>/hf_c32_isl512_w2

# Auto-find latest run
python3 analysis/create_report.py analyze

# Prometheus delta (initial vs final snapshot)
diff <(grep "^dynamo_frontend" artifacts/.../prometheus/initial_snapshot.txt | sort) \
     <(grep "^dynamo_frontend" artifacts/.../prometheus/final_snapshot.txt | sort)

# nsys SQLite queries (when nsys was enabled)
sqlite3 artifacts/.../nsys/frontend.sqlite \
    "SELECT name, COUNT(*), ROUND(AVG(end-start)/1e3,1) as avg_us
     FROM NVTX_EVENTS WHERE end > start GROUP BY name ORDER BY avg_us DESC"
```

## Output Structure

```text
artifacts/sweep_YYYYMMDD_HHMMSS/
    results.csv                        Sweep results (all runs)
    summary.md                         Comparison table
    hf_c32_isl512_w2/                  Per-run directory
        config.json                    Run parameters
        report.md                      Analysis report
        aiperf/
            profile_export_aiperf.json aiperf metrics
        prometheus/
            initial_snapshot.txt        Pre-load metrics
            final_snapshot.txt          Post-load metrics
            timeseries.jsonl            Per-second scrapes
        system/
            thread_count.txt            Thread count over time
            fd_count.txt                FD count over time
            proc_status.txt             /proc/PID/status snapshots
        logs/
            frontend.log
            mocker_*.log
        perf/                           (if --with-perf)
            perf_stat.txt
            cpu_flamegraph.svg
        bpf/                            (if --with-bpf, needs root)
            runqlat.txt
            syscall_latency.txt
            ...
        nsys/                           (if --with-nsys)
            frontend.nsys-rep
            frontend.sqlite
```
