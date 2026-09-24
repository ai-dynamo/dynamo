# Tyche response transport comparison — September 23, 2026

**Velo TCP improves throughput and CPU cost against Dynamo TCP. QUIC has the best throughput, CPU/request, and packet cost in this comparison. Velo RDMA gives no clear end-to-end gain over Velo TCP.** TCP remains the default.

All 16 measured runs are retained: four balanced repeats per mode, after one warmup/discard per mode. They exported **9,854,326 completed records**, including **five empty-content errors**. Each run has a complete export and all 2,048 KV sources active on both frontends. Every RDMA process used the selected `rc_mlx5` lane. The campaign used normal, non-preemptible allocations on the same five nodes; no backfill partition was used.

## Findings

- **Velo TCP versus TCP:** requests/s increased 71.3%, output tokens/s increased 66.5%, and frontend CPU/request fell 21.6%. RX wire packets/output token fell 46.6%. At the higher throughput, RX packets/s fell 10.9%, while combined RX+TX packets/s changed +0.2%.
- **QUIC versus Velo:** Velo TCP had 8.2% less output throughput, 18.4% more CPU/request, and 17.4% more RX packets/output token than QUIC. Velo RDMA also trailed QUIC on these measures. Velo's median TTFT was lower, but its p95/p99 TTFT was higher.
- **RDMA versus Velo TCP:** median output throughput differed by only 0.18%, with overlapping ranges. RDMA used 16.5% more combined wire packets/output token. It moved traffic off Ethernet, but added InfiniBand packets. Its total packet rate was +17.4% against TCP. Ethernet and InfiniBand packet counts are shown separately below.
- **Latency tradeoff:** Velo reduced p99 TTFT by about 74% against TCP, while p99 per-request mean ITL rose by 57–58%. Its p99 request latency rose by about 1–2%. QUIC also increased p99 mean ITL against TCP. These results exceed the provisional 5% p99 ITL limit; the throughput, p99 TTFT, p99 request latency, and CPU/request limits pass. This is a directional comparison, not a default-change qualification.

Values below are medians of four runs. [All metrics and ranges](all-metrics.csv) include changes against both TCP and QUIC. [Per-run values, counts, and errors](summary.json) and [the full metric table](metrics.md) are retained. Percent changes against a zero baseline are blank. For RDMA-only counters in modes without RDMA responses, use the absolute values; near-zero background counts make percentage changes unhelpful.

### Throughput and process CPU

| Metric | TCP | QUIC | Velo TCP | Velo RDMA |
|---|---:|---:|---:|---:|
| Requests/s | 2,475.6 | 4,581.6 | 4,241.2 | 4,259.4 |
| Output tokens/s (million) | 2.338 | 4.240 | 3.894 | 3.901 |
| Frontend CPU ms/request | 40.82 | 27.03 | 32.01 | 32.03 |
| Frontend CPU µs/output token | 43.16 | 29.24 | 35.02 | 34.94 |

### End-to-end latency

| Metric | TCP | QUIC | Velo TCP | Velo RDMA |
|---|---:|---:|---:|---:|
| TTFT p50 (ms) | 1,546.3 | 90.4 | 74.0 | 79.1 |
| TTFT p95 (ms) | 2,453.5 | 898.5 | 1,298.0 | 1,296.5 |
| TTFT p99 (ms) | 5,380.4 | 975.7 | 1,420.2 | 1,384.4 |
| Mean ITL per request p50 (ms) | 0.112 | 0.648 | 0.624 | 0.621 |
| Mean ITL per request p95 (ms) | 1.114 | 2.001 | 2.010 | 1.985 |
| Mean ITL per request p99 (ms) | 1.748 | 2.804 | 2.762 | 2.750 |
| Request latency p50 (ms) | 1,635.9 | 824.6 | 1,160.6 | 1,176.3 |
| Request latency p95 (ms) | 5,373.0 | 4,744.1 | 4,830.1 | 4,843.5 |
| Request latency p99 (ms) | 11,551.1 | 11,678.9 | 11,788.5 | 11,674.9 |

### Frontend physical packet load

| Metric | TCP | QUIC | Velo TCP | Velo RDMA |
|---|---:|---:|---:|---:|
| RX packets/s (million) | 3.608 | 2.987 | 3.215 | 3.396 |
| TX packets/s (million) | 2.106 | 2.472 | 2.513 | 3.315 |
| RX packets/completed request | 1,461.4 | 650.3 | 756.9 | 799.0 |
| TX packets/completed request | 857.6 | 538.9 | 591.5 | 777.4 |
| RX packets/output token | 1.546 | 0.704 | 0.826 | 0.869 |
| TX packets/output token | 0.908 | 0.583 | 0.646 | 0.845 |
| RX + TX packets/output token | 2.454 | 1.287 | 1.471 | 1.714 |
| Ethernet mean RX frame bytes | 437.7 | 897.5 | 790.9 | 727.7 |

### Packet rates by fabric

| Metric | TCP | QUIC | Velo TCP | Velo RDMA |
|---|---:|---:|---:|---:|
| ETHERNET RX packets/s (million) | 3.608 | 2.987 | 3.215 | 2.509 |
| ETHERNET TX packets/s (million) | 2.106 | 2.472 | 2.513 | 2.471 |
| RDMA RX packets/s (million) | 0.000 | 0.000 | 0.000 | 0.888 |
| RDMA TX packets/s (million) | 0.000 | 0.000 | 0.000 | 0.846 |

### Host and client cost

| Metric | TCP | QUIC | Velo TCP | Velo RDMA |
|---|---:|---:|---:|---:|
| Frontend process cores | 100.99 | 124.55 | 136.26 | 136.88 |
| Frontend process system cores | 53.24 | 36.51 | 38.89 | 37.29 |
| Whole-host system cores | 22.90 | 20.85 | 22.49 | 21.91 |
| Whole-host softirq cores | 34.04 | 17.20 | 18.02 | 16.84 |
| Frontend UCX progress cores | 0.00 | 0.00 | 0.00 | 1.79 |
| Frontend RSS (GiB) | 42.73 | 45.97 | 39.19 | 39.21 |
| Client process cores | 81.36 | 129.79 | 127.54 | 127.48 |
| Ethernet RX discards/s | 0.000 | 0.000 | 0.000 | 0.000 |
| Ethernet TX discards/s | 0.000 | 0.000 | 0.000 | 0.000 |
| Ethernet RX out-of-buffer/s | 0.204 | 0.042 | 0.000 | 0.000 |
| Host TCP retransmitted segments/s | 437.061 | 4,406.624 | 855.685 | 689.017 |

### Completed exports and retained errors

| Mode | Exported records | Errors | Drain cancellations |
|---|---:|---:|---:|
| TCP | 1,564,930 | 0 | 6,236 |
| QUIC | 2,926,350 | 1 | 0 |
| Velo TCP | 2,678,518 | 0 | 0 |
| Velo RDMA | 2,684,528 | 4 | 0 |

The host TCP retransmission counter includes HTTP and request-plane traffic; it does not measure QUIC retransmissions or total UCX retries. The selected InfiniBand ports recorded seven out-of-buffer events in the first RDMA run. The checked receive-error, transmit-discard, packet-sequence, local-ack-timeout, and transport-retry-exhaustion counters were zero. These hardware events are separate from request errors. The CSV includes their per-run ranges.

The packet total is the sum of the measured Ethernet interface and the two selected InfiniBand ports. It includes all traffic on those ports and does not isolate responses. Ethernet mean RX frame size rose from 437.7 bytes for TCP to 897.5 for QUIC and 790.9 for Velo TCP. The hardware size-bucket values are in the CSV. RDMA's Ethernet frame size describes its remaining Ethernet traffic; it is not an RDMA packet-size measurement. A larger application batch alone is not evidence of a larger wire packet.

Process CPU and host system/softirq counters are separate accounting views and are not added. Linux CPU accounting can attribute interrupt time through process system-time accounting; see the [kernel source](https://github.com/torvalds/linux/blob/master/kernel/sched/cputime.c).

The five errors were `InvalidInferenceResultError` records with no response content. They remain in the data. TCP also had 6,236 credits cancelled at the drain deadline, about 0.40% of its completed-plus-cancelled credits. Other modes had none. The pinned AIPerf does not export cancelled credits. Latency percentiles therefore describe successful exported requests; cancellation differences limit tail comparisons. No run was discarded for these counts.

### Throughput ranges across repeats

| Mode | Requests/s range | Output tokens/s range (million) |
|---|---:|---:|
| TCP | 2,435.1–2,493.1 | 2.302–2.360 |
| QUIC | 4,532.7–4,684.6 | 4.176–4.326 |
| Velo TCP | 4,233.9–4,280.5 | 3.880–3.902 |
| Velo RDMA | 4,213.1–4,308.3 | 3.870–3.979 |

## Method

The campaign uses five exclusive Grace nodes: frontend `ptyche0128`, workers `ptyche0130`, `ptyche0133`, `ptyche0135`, and client `ptyche0136`. Each frontend is bound to one NUMA domain and 72 CPUs. There are 2,048 speedup-10 mock workers in 12 processes across the three worker nodes. The separate client uses AgentX's 336 traces, seed 12345, concurrency 6,144, and 128 AIPerf workers.

Every run starts fresh services. It uses 32 warmup requests, a 30-second ramp, a 120-second measurement, and a 90-second drain allowance followed by complete record export. The measured order is balanced across four rounds, after one discarded run per mode. Tokenizer, routing, replica sync, request transport, and KV-event settings are common to all modes. No profiles overlap timing runs.

Frontend request and output-token rates, process CPU, and hardware network counters use the same observed measurement window. Counter boundaries are interpolated between adjacent samples. Latency percentiles use successful exported requests that started inside that window, including requests that completed during drain. Inter-token latency is AIPerf's mean inter-token latency per request; its p99 is not a percentile over all individual token gaps.

Ethernet packet counts and receive-size buckets come from physical NIC counters. InfiniBand packet counts come from the two selected physical ports. Host interface packet counters are reported separately; GRO/GSO aggregation and RDMA completions are not used as wire packet counts. The reported total sums the Ethernet interface and the two selected InfiniBand ports. These counters include HTTP, requests, response traffic, KV events, and background traffic on those ports. They do not isolate responses. Process CPU and host softirq figures are separate accounting views and are not added.

## Source and build

Dynamo main was checked at the start of the final campaign and frozen at `7472c23abfbbc56387d4fd44631b06a3f487f78e`. The compiled integration is `0c14c336c2547d2911d4cad5f530285fd91b10dd`, with Velo pinned to `da848eb78eec6331d670b8d80b5b0b8333b464cc`. The freeze time is September 23, 2026, 13:12:27 PDT.

The Python extension SHA-256 is `9d9d383ff77a12058373f01618a8eff3190e4b7671c5406d89a69a8ef18609b8`. Build features are `tracing/release_max_level_warn,velo-ucx`; Rust flags are `-C target-cpu=native -C force-frame-pointers=yes --cfg tokio_unstable`. The build uses Rust 1.96.1 on aarch64 with release debug information. The report harness is pinned at `5c51a09b7b78f9f23f9954c0575c6fdd91342794`. Source changes after the compiled revision are confined to the benchmark harness. The frozen manifest records all three lockfiles, source overlays, and binary hashes, which are checked before each run.

The existing benchmark ZMQ socket-capacity overlay is common to all modes. Its source patch and dependency-file hashes are retained. Selected registry dependencies were checked against their checksum-pinned crate archives. Harness-only changes after the freeze correct exported-record counting, move analysis after the timing matrix, and allow continuation across allocations. They do not rebuild the extension.

## Hardware and interpretation limits

Ethernet uses `enP6p3s0f1np1` on NUMA node 0, 200 Gb/s, MTU 1500. Velo RDMA uses `rc_mlx5` active-message lanes on the NUMA-local `mlx5_0:1` and `mlx5_4:1` ports, each InfiniBand 400 Gb/s with active MTU 4096. The linked UCX version is 1.22.0. UCX is restricted to `rc_mlx5,ud_mlx5,self`; per-process lane checks reject TCP fallback. These are different fabrics, so RDMA differences cannot be attributed to software alone.

NIC settings, offloads, queue settings, and IRQ affinity were identical across all 16 runs. Ethernet used 63 combined queues, with effective IRQ affinity on CPUs 0–62 and 66. TSO, GSO, GRO, and UDP segmentation were enabled; hardware GRO and LRO were disabled. Full settings are in the environment artifact. Saved `/proc` status and NUMA maps record actual frontend memory placement. All sampled private anonymous pages were resident on the requested NUMA node for both frontends in all 16 runs. These samples were taken after drain; they do not describe every access during the measurement.

This is a fixed-concurrency mock-worker comparison, not a claim about maximum transport capacity or real-model throughput. The faster modes used about 127–130 client process cores out of 144 and logged 28,232–40,014 event-loop warnings per run. Client limits can affect these results. The frontend active-request gauge also differed by mode despite the same requested client concurrency; its observed distributions are retained in the summary. Small request error counts are retained. Client credits cancelled at the drain deadline are counted separately; the pinned AIPerf does not export records for those credits. Latency distributions are conditional on successful exported requests, which limits tail comparisons when cancellations differ.

The final C8192 RDMA preflight exported 709,099 completed records with no errors or drain cancellations. It verified all 2,048 KV sources on both frontends and all 14 processes' selected RDMA lanes. Velo live slots drained to zero. However, the largest worker pool had 601 live slots and no send-admission backpressure: this workload did not reproduce the previously documented concentration of 5,760 streams on one peer. The bounded-admission unit test passed separately; the preflight does not establish that all historical saturation cases are resolved.

## Validation

Final Velo validation: 977 unit tests passed, one ignored, plus six cancellation and 30 multiplexing tests at four test threads. Four idle-endpoint tests had timed out in an earlier unrestricted parallel run; all 40 selected UCX tests then passed serially and the full selected suite passed at four threads. Final Dynamo shared lifecycle validation passed on TCP and hardware RDMA; the default-feature TCP suite also passed. The UCX-enabled release build passed. The earlier adapter validation passed 284 network tests and four Python configuration tests; existing KVBM config, engine, and physical-layer consumers compiled.

Two draft PRs contain the implementation: [Dynamo #15231](https://github.com/ai-dynamo/dynamo/pull/15231) and [Velo #90](https://github.com/ai-dynamo/velo/pull/90).

## Reproduction and artifacts

The harness and workload template are in the [parent directory](../README.md). Raw records, logs, per-run configs, telemetry, hardware snapshots, and source/build manifests remain on Tyche under:

`/lustre/fsw/coreai_comparch_trtllm/jothomson/dynamo-velo-response-20260923`

The first TCP warmup finished its full export but exposed a harness count error: it incorrectly added 1,560 drain cancellations to 398,816 completed records. All 398,816 completed records were present, including one error. The count check was corrected and the completed warmup was not repeated. Its original failure artifacts are retained. It is outside the 16 measured runs.

The first 14 measured runs used allocation `2885802`; the last Velo TCP and QUIC repeats used `2886188`. The continuation retained the same node order, RDMA devices, link speeds, MTU, and frozen binary. Runs completed at 18:18:55 PDT; the full analysis completed at 18:41:43 PDT. Allocation accounting shows `CANCELLED` because the idle allocation keeper jobs were explicitly released after their work finished; this does not indicate a failed benchmark.

Compact reproduction artifacts:

- [Environment, workload, source pins, node settings, and per-run configuration hashes](environment.json).
- [Frozen source, lockfile, feature, and binary hashes](frozen-build.json).
- [Common Dynamo benchmark overlay](benchmark-overlay.patch) and [ZMQ dependency patch](zmq-dependency.patch).
- [Harness revision and file hashes](harness-revision.json), [allocation accounting](allocation-accounting.txt), and [harness instructions](../README.md), including the continuation procedure.

On Tyche, after the recorded configs and raw results are present, run `harness/analyze_campaign.py` and `harness/analyze_packets.py` for each of the 16 measured labels, then `harness/summarize_environment.py` and `harness/report.py`. Run large record analysis on an allocated compute node. The controller performs this sequence after timing runs. Full lockfile copies, dependency audits, raw exports, logs, and telemetry remain under the recorded Tyche root.
