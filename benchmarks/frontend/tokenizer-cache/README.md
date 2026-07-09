# Tokenizer Cache Benchmark

This benchmark demonstrates the frontend throughput benefit of Dynamo's L1
tokenizer cache using an AgentX trace and fast mock workers.

## Build Dynamo

Dynamo v1.2.1 does not include the tokenizer cache. Use `v1.3.0-rc1`, which
includes the opt-in L1 cache and its multi-turn extension. The tag predates
these benchmark files, so restore this directory from the benchmark branch
before building:

```bash
git clone --branch jthomson04/tokenizer-cache-benchmark-main-20260709 \
    https://github.com/ai-dynamo/dynamo.git
cd dynamo
git checkout v1.3.0-rc1
git checkout jthomson04/tokenizer-cache-benchmark-main-20260709 -- \
    benchmarks/frontend/tokenizer-cache

# With your Python build environment activated:
uv pip install pip 'maturin[patchelf]'
(cd lib/bindings/python && maturin develop --uv --release)
uv pip install -e .
```

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

On the reference 24-CPU host running `v1.3.0-rc1` (`14a3b0d913`), enabling the
8 GiB cache increased request throughput from 111.15 to 163.31 requests/second
(+46.9%) and input-token throughput from 9.05M to 14.53M tokens/second
(+60.6%). Median time to first token fell from 1,769 ms to 1,200 ms. Both runs
completed with zero request errors. Treat these as reference results; compare
cache off and on on the same otherwise-idle host.

AIPerf reconstructs and memory-maps the AgentX dataset. Ensure the host has
enough available memory before starting the benchmark.
