# Autoscaler evaluation: goodput per GPU-second

## How to read these tables

- **One row per autoscaler, one table per topology.** `agg` runs a single pool of aggregated workers; `disagg` runs separate prefill and decode pools. The seven methods, the 14 traces, the SLO and the simulator are identical within a table, so rows are directly comparable.
- **The metric is `goodput_per_gpu`**: SLO-compliant (good) requests served per second, divided by the average number of GPUs the autoscaler had provisioned (starting, active and draining workers all count). It is good requests per GPU-second, i.e. how much useful work one GPU-second bought; higher is better. Multiply by 3600 for good requests per GPU-hour.
- **`all (golden)`**: every cell shows the mean over all 14 traces, then in parentheses the mean over the six Astra GPT-OSS Golden Set traces (hour-long production-like traffic). The other eight are shorter registry traces (the Mooncake anchor plus seven synthetic patterns). Each trace is first averaged over its 3 repetitions, then traces are averaged with equal weight.
- **Goodput/GPU-s** is the headline column and orders the rows. **Mean rank** ranks the methods on each trace by that metric (1 = best, ties share the average rank) and averages the ranks, so it rewards being consistently good rather than winning a few traces by a wide margin. **Win rate** is the share of (trace, opponent) head-to-head comparisons a method wins (a tie counts half); 0.5 means average.
- **Good rate** (share of all requests that met both the TTFT and ITL SLO) and **Avg GPUs** explain *how* a method got its goodput per GPU: a high good rate on a large fleet (static) and a lower good rate on a small fleet (lean autoscalers) can score alike. Neither is a ranking criterion.
- **Fixed fleets are the reference points.** The static entry (4 workers, or 4 prefill + 4 decode) has the highest good rate because it never under-provisions, and the lowest goodput per GPU because it never scales down.
- **Caveat for the RL row.** Five of the six golden traces were training workloads for the RL planner (only random-bursts, flash_crowd and diurnal were held out), so its golden numbers are partly in-sample; the other methods never see the traces in advance.

Per-workload goodput_per_gpu, the rank matrices and the long-form data (with the leaderboard's latency and scaling metrics) are in `evaluation.xlsx` (sheets `<mode> goodput_per_gpu`, `<mode> ranks`, `<mode> data`).

## agg mode

14 workloads x 3 repetitions, SLA `relaxed`. Every column reads `all (golden)`: the mean over all workloads, then over the last 6 workloads in matrix order (the Astra GPT-OSS Golden Set). Goodput/GPU-s is `goodput_per_gpu`: good requests per second divided by the average GPU count, i.e. good requests per GPU-second (x3600 for GPU-hours). Values are averaged over repetitions, then over workloads with equal weight. Mean rank 1 = best per workload (ties share the average rank); pairwise win rate counts ties as half.

| Method | Goodput/GPU-s | Mean rank | Win rate | Good rate | Avg GPUs |
|---|---:|---:|---:|---:|---:|
| cloudai-mpc | 1.400 (0.925) | 2.96 (2.00) | 0.673 (0.833) | 69.7% (88.3%) | 2.18 (1.35) |
| **cloudai-rl** | 1.360 (0.863) | 3.21 (2.67) | 0.631 (0.722) | 68.2% (89.5%) | 1.96 (1.51) |
| jev | 1.360 (0.697) | 3.61 (4.67) | 0.565 (0.389) | 71.5% (94.3%) | 2.08 (2.00) |
| reactive-default | 1.296 (0.783) | 4.11 (3.75) | 0.482 (0.542) | 63.4% (73.4%) | 2.36 (1.36) |
| keda-default | 1.274 (0.793) | 4.21 (3.58) | 0.464 (0.569) | 73.8% (80.9%) | 3.40 (1.56) |
| planner | 1.135 (0.744) | 5.32 (4.33) | 0.280 (0.444) | 58.3% (89.6%) | 1.54 (1.70) |
| static-4 | 0.841 (0.365) | 4.57 (7.00) | 0.405 (0.000) | 93.4% (99.6%) | 4.00 (4.00) |

## disagg mode

14 workloads x 3 repetitions, SLA `relaxed`. Every column reads `all (golden)`: the mean over all workloads, then over the last 6 workloads in matrix order (the Astra GPT-OSS Golden Set). Goodput/GPU-s is `goodput_per_gpu`: good requests per second divided by the average GPU count, i.e. good requests per GPU-second (x3600 for GPU-hours). Values are averaged over repetitions, then over workloads with equal weight. Mean rank 1 = best per workload (ties share the average rank); pairwise win rate counts ties as half.

| Method | Goodput/GPU-s | Mean rank | Win rate | Good rate | Avg GPUs |
|---|---:|---:|---:|---:|---:|
| cloudai-mpc | 0.951 (0.626) | 3.79 (4.17) | 0.536 (0.472) | 83.1% (97.6%) | 3.49 (2.21) |
| **cloudai-rl** | 0.932 (0.647) | 3.43 (3.17) | 0.595 (0.639) | 76.8% (94.0%) | 2.85 (2.04) |
| jev | 0.931 (0.438) | 3.96 (6.00) | 0.506 (0.167) | 79.8% (98.7%) | 3.08 (3.32) |
| planner | 0.924 (0.657) | 3.04 (2.17) | 0.661 (0.806) | 71.4% (96.8%) | 2.39 (2.08) |
| reactive-default | 0.920 (0.658) | 3.00 (1.92) | 0.667 (0.847) | 79.2% (97.8%) | 3.13 (2.16) |
| keda-default | 0.806 (0.627) | 4.93 (3.58) | 0.345 (0.569) | 83.2% (96.8%) | 5.11 (2.21) |
| static-4p4d | 0.436 (0.185) | 5.86 (7.00) | 0.190 (0.000) | 95.5% (99.6%) | 8.00 (8.00) |

## CloudAI RL planner by mode

Mean goodput per GPU-second, `all (golden)`; the last column is the best non-RL method of the mode.

| Mode | cloudai-rl | Best other method |
|---|---:|---|
| agg | 1.360 (0.863) | cloudai-mpc: 1.400 (0.925) |
| disagg | 0.932 (0.647) | cloudai-mpc: 0.951 (0.626) |

