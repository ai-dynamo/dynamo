# Tokenizer Cache Benchmark

This benchmark demonstrates the frontend throughput benefit of Dynamo's L1
tokenizer cache using an AgentX trace and fast mock workers.

## Requirements

- Dynamo is already enabled, so `python -m dynamo.frontend` and
  `python -m dynamo.mocker` work.
- The host provides `python`, `uvx`, `taskset`, and `curl`.
- CPU IDs are contiguous and start at 0.
- Port 8000 is available.

## Run

From this directory, start the cache-off topology:

```bash
./launch.sh off
```

In a second terminal, run the benchmark:

```bash
./benchmark.sh
```

Stop the topology with Ctrl-C. Then repeat with the cache enabled:

```bash
./launch.sh on
```

```bash
./benchmark.sh
```

Compare the two AIPerf summaries written under
`benchmarks/results/tokenizer-cache/<timestamp>/`.

## Configuration

- Frontend: CPU 0, fastokens, round-robin routing.
- Tokenizer cache: explicitly disabled with `DYN_TOKENIZER_CACHE=0`, or enabled
  with `DYN_TOKENIZER_CACHE=1` and an 8 GiB budget.
- Mockers: four workers on every CPU except CPU 0, with a 1,000,000x speedup
  ratio and a 1,024-token KV-cache block size.
- Discovery and transport: file discovery, TCP requests, and ZMQ events.
- Model: `Qwen/Qwen3-0.6B`.
- Workload: all 336 entries from
  `semianalysisai/cc-traces-weka-with-subagents-060526`, concurrency 220,
  32 warmup requests, 120 measured seconds, and a 90-second grace period.
- AIPerf: `cquil11/aiperf` pinned to commit
  `8473e1545476c1d91932aa2402b642b416a23df6`.

On the reference 24-CPU host, enabling the 8 GiB cache increased request
throughput from 107.49 to 189.01 requests/second (+75.8%) and input-token
throughput from 8.45M to 16.94M tokens/second (+100.6%). The frontend served
92.9% of input tokens from the tokenizer cache. Both runs completed with zero
request errors. Treat these as reference results; compare cache off and on on
the same otherwise-idle host.

AIPerf reconstructs and memory-maps the AgentX dataset. Ensure the host has
enough available memory before starting the benchmark.
