<!-- SPDX-License-Identifier: Apache-2.0 -->

# Flame Graph Scripts

Scripts for generating CPU, off-CPU, and differential flame graph SVGs from the Dynamo frontend. Each script auto-detects available profiling tools and picks the best one.

## Scripts

| Script | What it does | Requires root? |
|--------|-------------|----------------|
| `cpu_flamegraph.sh` | On-CPU sampling flame graph. Tries cargo-flamegraph, samply, then falls back to `perf record` + flamegraph.pl/inferno. | No (but `perf` needs `CAP_PERFMON` or `perf_event_paranoid=-1`) |
| `offcpu_flamegraph.sh` | Off-CPU flame graph via BPF. Shows what threads block on: mutexes, I/O, futex, socket waits. | Yes (BPF requires root or `CAP_BPF`) |
| `diff_flamegraph.sh` | Differential flame graph comparing two profiles. Red = regression, blue = improvement. | No |

## Quick Start

```bash
# Get the frontend PID from a running capture
FRONTEND_PID=$(pgrep -f "dynamo.frontend" | head -1)

# CPU flame graph (30s sample)
./cpu_flamegraph.sh --pid $FRONTEND_PID --duration 30

# Off-CPU flame graph (what's blocking threads)
sudo ./offcpu_flamegraph.sh --pid $FRONTEND_PID --duration 30

# Differential: compare before/after an optimization
./diff_flamegraph.sh before.perf.data after.perf.data
```

## Tool Priority

`cpu_flamegraph.sh` tries tools in order:

1. **cargo-flamegraph** — simplest, one-step SVG (only for launching a new binary, not `--pid`)
2. **samply** — generates a Firefox Profiler-compatible JSON (supports `--pid`)
3. **perf record** + **flamegraph.pl** or **inferno** — most common fallback

`offcpu_flamegraph.sh` tries:

1. **bpftrace** — inline BPF script capturing sched_switch stacks
2. **bcc offcputime-bpfcc** — BCC tools fallback

## Options

All scripts share a common option style:

| Option | Description | Default |
|--------|-------------|---------|
| `--pid PID` | Attach to running process | — |
| `--duration N` | Capture duration in seconds | 30 |
| `--output-dir DIR` | Output directory | `.` |
| `--freq HZ` | Sampling frequency (CPU only) | 99 |
| `--min-us N` | Minimum off-CPU time in us (off-CPU only) | 1000 |

## Interpreting Results

### CPU Flame Graph
- Wide towers = functions consuming the most CPU time
- Look for hot paths in `tokio-runtime-worker` threads
- Narrow, deep stacks = normal call chains; wide, flat = optimization targets

### Off-CPU Flame Graph
- `futex_wait_queue` → mutex/condvar contention
- `ep_poll` → epoll_wait (normal Tokio I/O loop)
- `schedule_timeout` → timer/sleep
- `tcp_sendmsg` / `tcp_recvmsg` → socket I/O blocking

### Differential Flame Graph
- **Red** frames got slower (regression)
- **Blue** frames got faster (improvement)
- Width difference shows magnitude of change

## Integration with Capture Script

The main capture script generates flame graphs automatically from `perf record` data:
```bash
sudo bash benchmarks/frontend/scripts/run_perf.sh \
  --skip-nsys \
  --model Qwen/Qwen3-0.6B --concurrency 64 --num-requests 4096

# Flame graph SVGs appear in artifacts/obs_<timestamp>/perf/
```

## Requirements

- **CPU**: `perf` (`apt install linux-tools-$(uname -r)`) or `cargo install flamegraph` or `cargo install samply`
- **Off-CPU**: `bpftrace` >= 0.16 or `bcc-tools`
- **SVG generation**: `cargo install inferno` (provides `inferno-collapse-perf`, `inferno-flamegraph`, `inferno-diff-folded`) or Brendan Gregg's [FlameGraph](https://github.com/brendangregg/FlameGraph) scripts
