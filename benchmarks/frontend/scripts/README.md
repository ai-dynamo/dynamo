# Frontend Profiling Scripts

Unified observability capture for Dynamo frontend performance analysis. Profiles
CPU usage, off-CPU blocking, BPF kernel traces, hardware counters, and Prometheus
metrics — all captured in parallel during a synthetic load run.

## Directory Layout

```
scripts/
├── run_perf.sh                   # Main orchestrator — runs everything
├── flamegraph/
│   ├── cpu_flamegraph.sh         # On-CPU sampling → SVG
│   ├── offcpu_flamegraph.sh      # Off-CPU blocking stacks → SVG
│   └── diff_flamegraph.sh        # Before/after differential flamegraph
├── bpf/
│   ├── run.sh                    # BPF script orchestrator (batch or single)
│   ├── setup.sh                  # Install bpftrace, configure kernel caps
│   └── traces/                   # 10 individual .bt scripts
├── tokio/
│   └── tokio_console.sh          # Connect tokio-console to running frontend
└── analysis/
    ├── create_report.py          # Unified markdown report generator
    ├── frontend_perf_analysis.py # Scalability curves, regression detection
    └── parsing_util.py           # Data extraction helpers
```

## Prerequisites

### Required (always needed)

| Tool | Purpose | Install |
|------|---------|---------|
| `etcd` | Config store for dynamo services | `apt install etcd` or [etcd releases](https://github.com/etcd-io/etcd/releases) |
| `nats-server` | Message bus for event plane | `apt install nats-server` or [nats.io](https://nats.io/download/) |
| `aiperf` | Synthetic load generator | `pip install git+https://github.com/ai-dynamo/aiperf.git` |
| `python3` | Analysis scripts, mocker, frontend | System python or venv |
| `jq` | JSON parsing in readiness checks | `apt install jq` |

### Profiling Tools (install what you need)

#### Tier 1: CPU Flamegraphs (`cpu_flamegraph.sh`)

The script auto-detects and uses the first available tool:

| Priority | Tool | Install | Notes |
|----------|------|---------|-------|
| 1 | `cargo-flamegraph` | `cargo install flamegraph` | Simplest — wraps perf + SVG in one step |
| 2 | `samply` | `cargo install samply` | Outputs Firefox Profiler JSON (interactive) |
| 3 | `perf` + renderer | See below | Most common on Linux servers |

When using `perf` (priority 3), you also need an SVG renderer:

| Renderer | Install |
|----------|---------|
| `inferno` (recommended) | `cargo install inferno` |
| Brendan Gregg's FlameGraph | `git clone https://github.com/brendangregg/FlameGraph && export PATH=$PWD/FlameGraph:$PATH` |

**perf setup:**
```bash
# Install perf for your kernel
apt install linux-tools-$(uname -r) linux-tools-common

# Allow non-root profiling (optional — avoids sudo)
echo -1 | sudo tee /proc/sys/kernel/perf_event_paranoid
```

#### Tier 2: Off-CPU Flamegraphs (`offcpu_flamegraph.sh`)

Shows *why* threads block (futex waits, epoll, I/O). Requires root or BPF capabilities.

| Priority | Tool | Install |
|----------|------|---------|
| 1 | `bpftrace` | `apt install bpftrace` or `bpf/setup.sh --install` |
| 2 | `offcputime-bpfcc` | `apt install bcc-tools` |

#### Tier 3: BPF Traces (`bpf/run.sh`)

Deep kernel-level tracing: scheduler latency, syscall costs, TCP lifetimes, context switches.

```bash
# Check if your system is BPF-ready
./bpf/run.sh --check

# Full automated setup (install + kernel config + capabilities)
sudo ./bpf/setup.sh --install --kernel --caps
```

Requirements: Linux kernel >= 4.18, `bpftrace` >= 0.16, CAP_BPF + CAP_PERFMON (or root).

#### Tier 4: Nsight Systems (`nsys`)

NVIDIA GPU + NVTX pipeline stage profiling.

| Tool | Install |
|------|---------|
| `nsys` | [NVIDIA Nsight Systems](https://developer.nvidia.com/nsight-systems) (part of CUDA toolkit) |

#### Tier 5: Tokio Console (`tokio/tokio_console.sh`)

Live async runtime introspection (task polls, waker counts, resource utilization).

| Tool | Install |
|------|---------|
| `tokio-console` | `cargo install tokio-console` |

Requires the frontend binary built with `tokio_unstable` cfg and `console-subscriber` enabled.

#### Optional: Packet Capture

| Tool | Install | Notes |
|------|---------|-------|
| `tcpdump` | `apt install tcpdump` | Enable with `--tcpdump` flag; captures to `system/capture.pcap` |

## Profiling Strategies

### Strategy 1: Quick CPU Profile (no root needed)

Best for: "Where is CPU time going?"

```bash
./run_perf.sh --skip-nsys --skip-bpf \
    --model Qwen/Qwen3-0.6B --concurrency 64 --num-requests 4096
```

**What runs:** `perf stat` + `cpu_flamegraph.sh`
**Output:** `perf/cpu_flamegraph.svg`, `perf/perf_stat.txt`
**Tools needed:** `perf` + `inferno` (or `cargo-flamegraph` / `samply`)

### Strategy 2: Full Flamegraphs (root required for off-CPU)

Best for: "CPU hotspots + what's blocking threads"

```bash
sudo ./run_perf.sh --skip-nsys \
    --model Qwen/Qwen3-0.6B --concurrency 64 --num-requests 4096
```

**What runs:** `perf stat` + `cpu_flamegraph.sh` + `offcpu_flamegraph.sh` + BPF traces
**Output:** `perf/cpu_flamegraph.svg`, `perf/offcpu_flamegraph.svg`, `bpf/*.txt`
**Tools needed:** `perf` + renderer + `bpftrace`

### Strategy 3: Deep Kernel Analysis (root required)

Best for: "Scheduler latency, syscall overhead, TCP connection issues"

```bash
sudo ./run_perf.sh --skip-nsys --skip-flamegraph \
    --model Qwen/Qwen3-0.6B --concurrency 64 --num-requests 4096
```

**What runs:** `perf stat` + all 10 BPF trace scripts
**Output:** `bpf/runqlat.txt`, `bpf/syscall_latency.txt`, `bpf/tcplife.txt`, etc.
**Tools needed:** `bpftrace`

### Strategy 4: Everything (root + NVIDIA GPU)

Best for: Full observability — correlate CPU, GPU, kernel, and application metrics.

```bash
sudo ./run_perf.sh \
    --model Qwen/Qwen3-0.6B --concurrency 64 --num-requests 4096
```

**What runs:** nsys + perf stat + flamegraphs + BPF + system stats + Prometheus
**Tools needed:** All of the above + `nsys`

### Strategy 5: Clean Baseline (no profiler overhead)

Best for: Getting accurate latency/throughput numbers without profiler noise.

```bash
./run_perf.sh --skip-nsys --skip-perf --skip-bpf --skip-flamegraph \
    --model Qwen/Qwen3-0.6B --concurrency 64 --num-requests 4096
```

**What runs:** System stats + Prometheus scraping + aiperf load only
**Output:** `aiperf/`, `prometheus/`, `system/`
**Tools needed:** None beyond required tools

### Strategy 6: Differential Flamegraph (compare two runs)

Best for: "Did my change make things better or worse?"

```bash
# Run baseline and experiment, then compare
./flamegraph/diff_flamegraph.sh \
    artifacts/obs_baseline/perf/cpu_flamegraph.perf.data \
    artifacts/obs_experiment/perf/cpu_flamegraph.perf.data
```

Red = regression (more CPU), blue = improvement (less CPU).
**Tools needed:** `perf` + `inferno` (or FlameGraph scripts)

## Standalone Script Usage

### CPU Flamegraph
```bash
# Attach to running process
./flamegraph/cpu_flamegraph.sh --pid 12345 --duration 30 --freq 99

# Custom output location
./flamegraph/cpu_flamegraph.sh --pid 12345 --output-dir /tmp --output my_flame
```

### Off-CPU Flamegraph
```bash
# Requires root/CAP_BPF
sudo ./flamegraph/offcpu_flamegraph.sh --pid 12345 --duration 30

# Filter stacks shorter than 100us
sudo ./flamegraph/offcpu_flamegraph.sh --pid 12345 --min-us 100
```

### BPF Traces
```bash
# List available scripts
./bpf/run.sh --list

# Run single script
./bpf/run.sh --pid 12345 offcputime

# Run all scripts in batch
sudo ./bpf/run.sh --batch --pid 12345 --output-dir /tmp/bpf --duration 30
```

## Analyzing Results

After a capture completes, generate a markdown report:

```bash
python3 analysis/create_report.py analyze artifacts/obs_YYYYMMDD_HHMMSS
```

The report includes:
- aiperf throughput and latency percentiles
- Prometheus stage durations and transport breakdown
- Hardware counters from `perf stat`
- BPF histogram summaries (scheduler, syscall, TCP)
- Flamegraph file references
- System resource trends (threads, FDs, sockets)
- Auto-generated key findings

## Common Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model PATH` | `nvidia/Llama-3.1-8B-Instruct-FP8` | Model path |
| `--concurrency N` | 64 | Parallel aiperf requests |
| `--num-requests N` | 640 | Total requests |
| `--isl N` | 1024 | Input sequence length |
| `--osl N` | 256 | Output sequence length |
| `--capture-duration N` | 60 | Seconds for parallel captures |
| `--workers N` | 2 | Number of mocker workers |
| `--speedup-ratio R` | 1.0 | Mocker speedup (0 = infinite) |
| `--skip-nsys` | | Skip Nsight Systems |
| `--skip-perf` | | Skip perf stat |
| `--skip-bpf` | | Skip BPF tracing |
| `--skip-flamegraph` | | Skip flamegraph generation |
| `--tcpdump` | | Enable packet capture via tcpdump |
| `--tcpdump-port PORT` | `--frontend-port` | Port filter for tcpdump |
| `--nsys-path PATH` | auto-detected | Path to nsys binary |
| `--output-dir DIR` | auto timestamped | Output directory |

## Output Structure

```
artifacts/obs_YYYYMMDD_HHMMSS/
├── config.json               # Capture parameters and tool availability
├── aiperf/                   # Load generator results
├── perf/
│   ├── perf_stat.txt         # Hardware counters
│   ├── cpu_flamegraph.svg    # On-CPU flamegraph
│   ├── cpu_flamegraph.perf.data
│   ├── offcpu_flamegraph.svg # Off-CPU flamegraph
│   └── offcpu_flamegraph.stacks
├── bpf/                      # BPF trace outputs
│   ├── runqlat.txt
│   ├── syscall_latency.txt
│   ├── offcputime.txt
│   ├── context_switches.txt
│   ├── cpudist.txt
│   ├── transport_latency.txt
│   ├── tcplife.txt
│   └── tcpretrans.txt
├── nsys/                     # Nsight Systems (if enabled)
│   ├── frontend.nsys-rep
│   └── frontend.sqlite
├── system/                   # /proc polling
│   ├── proc_status.txt
│   ├── proc_stat.txt         # Raw scheduler/CPU time (/proc/PID/stat)
│   ├── proc_statm.txt        # Page-level memory (/proc/PID/statm)
│   ├── thread_count.txt
│   ├── fd_count.txt
│   ├── ss_stats.txt
│   └── capture.pcap          # Packet capture (when --tcpdump enabled)
├── prometheus/               # Metrics scraping
│   ├── timeseries.jsonl
│   ├── final_snapshot.txt
│   └── mocker_*_snapshot.txt
└── logs/                     # Service logs
    ├── frontend.log
    └── mocker_*.log
```
