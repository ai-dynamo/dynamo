# DGD container discovery comparison

Completed trials: 8/8. Main group aggregates use 32Gi cases only.

Seconds; group values are mean (min–max).

| Group | n | DGD request → workload Ready | Pod → Ready* | DGD → handler | CRIU | CUDA | GMS span | All published | Gate wait |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| control | 3 | 27.742 (27.355–27.970) | 27.636 (27.262–28.225) | 5.689 (5.497–5.985) | 6.523 (6.103–6.957) | 9.313 (7.715–10.628) | 20.576 (20.415–20.829) | 23.659 (23.293–23.876) | 0.000 (0.000–0.000) |
| runtime | 3 | 24.951 (24.054–25.975) | 24.786 (23.588–26.210) | 2.712 (2.421–3.185) | 4.599 (3.935–5.041) | 8.940 (8.310–9.603) | 19.103 (18.937–19.263) | 22.062 (21.897–22.369) | 3.099 (2.970–3.186) |

| Case / cohort | DGD request → workload Ready | Pod → Ready* | DGD → handler | CRIU | CUDA | GMS span | All published | Gate wait | Runtime lookup proven |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| discovery-dgd-control-1 / qualification_8Gi | 29.405 | 29.568 | 5.967 | 8.156 | 9.634 | 19.737 | 23.056 | 0.000 | False |
| discovery-dgd-runtime-1 / qualification_8Gi | 25.534 | 25.231 | 2.930 | 4.771 | 7.354 | 18.745 | 21.929 | 3.041 | True |
| discovery-dgd-runtime-2 / main_32Gi | 24.054 | 23.588 | 2.529 | 5.041 | 8.310 | 19.108 | 21.897 | 3.139 | True |
| discovery-dgd-control-2 / main_32Gi | 27.970 | 28.225 | 5.985 | 6.957 | 9.596 | 20.485 | 23.876 | 0.000 | False |
| discovery-dgd-control-3 / main_32Gi | 27.355 | 27.419 | 5.585 | 6.509 | 7.715 | 20.829 | 23.809 | 0.000 | False |
| discovery-dgd-runtime-3 / main_32Gi | 24.825 | 24.561 | 2.421 | 3.935 | 9.603 | 19.263 | 21.922 | 3.186 | True |
| discovery-dgd-control-4 / main_32Gi | 27.900 | 27.262 | 5.497 | 6.103 | 10.628 | 20.415 | 23.293 | 0.000 | False |
| discovery-dgd-runtime-4 / main_32Gi | 25.975 | 26.210 | 3.185 | 4.821 | 8.908 | 18.937 | 22.369 | 2.970 | True |

| Pair / cohort | Order | Δ request → workload Ready | Δ handler entry | Δ GMS span | Δ gate wait |
|---|---|---:|---:|---:|---:|
| 1 / qualification_8Gi | A B | -3.872 | -3.037 | -0.992 | +3.041 |
| 2 / main_32Gi | B A | -3.915 | -3.457 | -1.377 | +3.139 |
| 3 / main_32Gi | A B | -2.530 | -3.164 | -1.566 | +3.186 |
| 4 / main_32Gi | A B | -1.925 | -2.312 | -1.478 | +2.970 |

Event offsets from DGD creation request; Pod first-observed time includes API-watch latency.

| Case | Pod first observed | Pod created* | First GMS | Wake gate entered | Wake gate passed | DGD Ready observed |
|---|---:|---:|---:|---:|---:|---:|
| discovery-dgd-control-1 | 0.727 | -0.163 | 3.319 | 26.740 | 26.740 | 29.418 |
| discovery-dgd-runtime-1 | 0.748 | 0.303 | 3.184 | 18.976 | 22.017 | 24.845 |
| discovery-dgd-runtime-2 | 0.699 | 0.466 | 2.789 | 18.842 | 21.981 | 23.641 |
| discovery-dgd-control-2 | 0.709 | -0.256 | 3.391 | 25.314 | 25.314 | 27.758 |
| discovery-dgd-control-3 | 0.782 | -0.064 | 2.980 | 24.689 | 24.689 | 26.904 |
| discovery-dgd-runtime-3 | 0.742 | 0.264 | 2.659 | 18.825 | 22.011 | 24.772 |
| discovery-dgd-control-4 | 0.766 | 0.638 | 2.878 | 25.285 | 25.285 | 27.526 |
| discovery-dgd-runtime-4 | 0.771 | -0.235 | 3.432 | 19.491 | 22.461 | 25.350 |

Every included case validates the DGD ownership chain, two inference prompts, eight rank publications, 112 unique O_DIRECT payload files, and resident agent/PageBroker readiness with starts before t0.

- t0 is the client DGD creation request; the main endpoint is observed workload Pod Ready after coherent generation, distinct from the DGD Ready condition.
- Pod creationTimestamp and container startedAt have one-second precision. Pod creation offsets may be negative by rounding; Pod-to-Ready has that uncertainty.
- Watch receipt times include remote observation latency and cross-host clock uncertainty.
- Agent restore handler entry is per-request dispatch, not daemon startup.
- CRIU/CUDA durations are agent phase measurements; CUDA includes PageBroker work.
- Main-series adjacent pairs follow B A / A B / A B. Deltas are runtime minus control; negative values mean faster with identity discovery.
- The first A/B pair used the operator default 8Gi /dev/shm and is qualification only. Main-series groups use explicit 32Gi, classified from the rendered DGD and actual Pod mount.
- This small same-node series uses preallocated DRA, cached images and PVC O_DIRECT; NFS server cache state is uncontrolled.
