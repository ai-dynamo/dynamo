### goodput_per_gpu — all workloads (14), SLA `relaxed`, RL entry `cloudai-rl`

| Method | Mean rank | Pairwise win rate |
|---|---:|---:|
| reactive-default | 3.00 | 0.667 |
| planner | 3.04 | 0.661 |
| cloudai-rl | 3.43 | 0.595 |
| cloudai-mpc | 3.79 | 0.536 |
| jev | 3.96 | 0.506 |
| keda-default | 4.93 | 0.345 |
| static-4p4d | 5.86 | 0.190 |

### goodput_per_gpu — golden workloads (6), SLA `relaxed`, RL entry `cloudai-rl`

| Method | Mean rank | Pairwise win rate |
|---|---:|---:|
| reactive-default | 1.92 | 0.847 |
| planner | 2.17 | 0.806 |
| cloudai-rl | 3.17 | 0.639 |
| keda-default | 3.58 | 0.569 |
| cloudai-mpc | 4.17 | 0.472 |
| jev | 6.00 | 0.167 |
| static-4p4d | 7.00 | 0.000 |

Mean rank: 1 = best per workload, ties share the average rank, lower is better. Pairwise win rate: share of (workload, opponent) comparisons won, ties count 0.5, higher is better. Metric values are averaged over repetitions before ranking.
