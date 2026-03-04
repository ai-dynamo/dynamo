<!-- SPDX-License-Identifier: Apache-2.0 -->

# BPF Tracing Scripts

eBPF/bpftrace scripts for low-overhead kernel-level tracing of the Dynamo frontend. These scripts attach to kernel tracepoints and kprobes to measure scheduling, syscall, TCP, and context-switch behavior without modifying the application.

## Setup

```bash
# Full setup: install bpftrace, configure kernel, grant capabilities
sudo bash setup.sh

# Or step by step:
sudo bash setup.sh --install    # install bpftrace
sudo bash setup.sh --kernel     # set perf_event_paranoid=-1, kptr_restrict=0
sudo bash setup.sh --caps       # grant capabilities (run bpftrace without sudo)

# Check current state
sudo bash setup.sh --check

# Undo everything
sudo bash setup.sh --reset
```

After granting capabilities, bpftrace runs **without sudo**.

## Quick Start

```bash
# Get the frontend PID from a running capture
FRONTEND_PID=$(pgrep -f "dynamo.frontend" | head -1)

# Run a single script
./run.sh --pid $FRONTEND_PID offcputime

# List all available scripts
./run.sh --list

# Check if BPF environment is ready
./run.sh --check
```

## Scripts

| Script | What it measures | Attach to PID? |
|--------|-----------------|----------------|
| `offcputime.bt` | Kernel stacks of threads going off-CPU (>1ms). Shows why threads block (futex, epoll, I/O). | Yes |
| `syscall_latency.bt` | Slow syscalls (>10us) by syscall ID, filtered to Tokio workers. | Yes |
| `runqlat.bt` | Scheduler run-queue latency — how long threads wait to be scheduled after wakeup. | Yes |
| `context_switches.bt` | Context switch rate and overhead per thread. | Yes |
| `cpudist.bt` | On-CPU time distribution per thread (how long a thread runs before being preempted). | Yes |
| `funclatency.bt` | Latency histogram for a specific kernel/user function (template — edit the probe). | Yes |
| `transport_latency.bt` | Socket read/write latency via syscall tracepoints. | Yes |
| `tcplife.bt` | TCP connection lifetimes — shows short-lived connections wasting setup cost. | No (system-wide) |
| `tcpretrans.bt` | TCP retransmission events. | No (system-wide) |

## Recommended Order for Frontend Analysis

**1. Start with off-CPU analysis** — identifies what's blocking Tokio workers:
```bash
bpftrace -p $FRONTEND_PID offcputime.bt
```
Look for `futex_wait_queue` (mutex contention), `ep_poll` (normal I/O), `schedule_timeout` (timers).

**2. Syscall latency** — find the expensive syscalls:
```bash
bpftrace -p $FRONTEND_PID syscall_latency.bt
```
`futex` with high avg latency = lock contention. `writev` = TCP send overhead.

**3. Run-queue latency** — check if threads are starved for CPU:
```bash
bpftrace -p $FRONTEND_PID runqlat.bt
```
p99 > 1ms means CPU contention is contributing to tail latency.

**4. TCP connection lifetimes** — verify connection reuse:
```bash
bpftrace tcplife.bt
```
Many short-lived connections (< 100ms) to localhost = connection pooling opportunity (Part 3 bottleneck).

**5. Context switches** — quantify scheduling overhead:
```bash
bpftrace -p $FRONTEND_PID context_switches.bt
```

## Interpreting Results

### Off-CPU Stacks
```
@blocked_us[tokio-runtime-w,
    futex_wait_queue        ← mutex/condvar contention
    futex / do_futex
    __x64_sys_futex
    entry_SYSCALL_64
]: [1ms, 10ms) = 4812
```
- `futex_wait_queue` → mutex blocked (check Prometheus registry, TCP endpoint table)
- `ep_poll` → epoll_wait (normal Tokio I/O loop — healthy)
- `schedule_timeout` → timer/sleep
- `do_wait` → join handle or channel receive

### Syscall Latency
```
@slow[futex]: count=6504, avg=110ms, total=717s
```
High `futex` total = lock contention dominates. Cross-reference with off-CPU stacks.

### TCP Lifetimes
```
PID   COMM        LADDR     LPORT  RADDR     RPORT  TX_KB  RX_KB  MS
12345 tokio-run   127.0.0.1 43210  127.0.0.1 8081   0      128    45
```
Many connections with lifetime < 100ms to mocker ports = no connection pooling.

## Directory Layout

```
bpf/
├── run.sh          # Script runner with capability detection
├── setup.sh        # Install bpftrace, configure kernel, grant caps
├── README.md
└── traces/         # bpftrace probe scripts (.bt files)
    ├── runqlat.bt
    ├── cpudist.bt
    ├── offcputime.bt
    ├── funclatency.bt
    ├── transport_latency.bt
    ├── tcplife.bt
    ├── tcpretrans.bt
    ├── syscall_latency.bt
    └── context_switches.bt
```

## Integration with Capture Script

The main capture script can run BPF traces automatically:
```bash
# Include BPF traces in the full capture (requires root)
sudo bash benchmarks/frontend/scripts/full_observability_run_perf.sh \
  --skip-nsys --skip-perf \
  --model Qwen/Qwen3-0.6B --concurrency 64 --num-requests 4096

# Compare event planes with BPF tracing:
sudo bash full_observability_run_perf.sh \
  --skip-nsys --skip-perf \
  --model Qwen/Qwen3-0.6B --concurrency 64 --num-requests 4096 \
  --event-plane zmq

sudo bash full_observability_run_perf.sh \
  --skip-nsys --skip-perf \
  --model Qwen/Qwen3-0.6B --concurrency 64 --num-requests 4096 \
  --event-plane nats

# BPF output appears in artifacts/obs_<timestamp>/bpf/
```

## Requirements

- Linux kernel >= 4.18 (for BPF CO-RE support)
- `bpftrace` >= 0.16
- Root or `CAP_BPF + CAP_PERFMON + CAP_NET_ADMIN + CAP_SYS_PTRACE` capabilities
- Kernel headers (for some kprobe scripts): `apt install linux-headers-$(uname -r)`
