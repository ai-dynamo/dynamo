### goodput_per_gpu — all workloads (14), SLA `relaxed`, RL entry `cloudai-rl`

| Method | Mean rank | Pairwise win rate |
|---|---:|---:|
| cloudai-mpc | 2.96 | 0.673 |
| cloudai-rl | 3.21 | 0.631 |
| jev | 3.61 | 0.565 |
| reactive-default | 4.11 | 0.482 |
| keda-default | 4.21 | 0.464 |
| static-4 | 4.57 | 0.405 |
| planner | 5.32 | 0.280 |

### goodput_per_gpu — golden workloads (6), SLA `relaxed`, RL entry `cloudai-rl`

| Method | Mean rank | Pairwise win rate |
|---|---:|---:|
| cloudai-mpc | 2.00 | 0.833 |
| cloudai-rl | 2.67 | 0.722 |
| keda-default | 3.58 | 0.569 |
| reactive-default | 3.75 | 0.542 |
| planner | 4.33 | 0.444 |
| jev | 4.67 | 0.389 |
| static-4 | 7.00 | 0.000 |

Mean rank: 1 = best per workload, ties share the average rank, lower is better. Pairwise win rate: share of (workload, opponent) comparisons won, ties count 0.5, higher is better. Metric values are averaged over repetitions before ranking.
