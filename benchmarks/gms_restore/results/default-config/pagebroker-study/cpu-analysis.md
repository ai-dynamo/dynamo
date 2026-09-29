# PageBroker CPU and transfer-ring analysis

The measured PageBroker cgroups show no quota throttling and low CPU scheduling pressure. The evidence does not identify a CPU-limit bottleneck or a repeatable end-to-end benefit from increasing the limit to 64.

| Case | PB request / limit | Ready (s) | Payload window (s) | Effective GiB/s | PB CPU seconds | PB throttled seconds | PB CPU some-pressure (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| pagebroker-isolated64-1 | 2 / 64 | — (isolated) | 13.868 | 32.305 | 41.666 | 0.000000 | 64.265 |
| pagebroker-native16-1 | 2 / 16 | 22.650 | 18.386 | 24.366 | 47.908 | 0.000000 | 60.698 |
| pagebroker-native16-2 | 2 / 16 | 22.732 | 18.084 | 24.773 | 47.013 | 0.000000 | 62.961 |
| pagebroker-native64-1 | 2 / 64 | 25.931 | 17.414 | 25.726 | 46.905 | 0.000000 | 68.007 |
| pagebroker-native64-2 | 2 / 64 | 22.603 | 17.592 | 25.465 | 47.572 | 0.000000 | 52.441 |

## CPU-limit cohorts

| PageBroker CPU limit | Trials | Mean Ready (s) | Ready range (s) | Mean payload window (s) | Mean CPU seconds |
|---|---:|---:|---:|---:|---:|
| 16 | 2 | 22.691 | 22.650–22.732 | 18.235 | 47.460 |
| 64 | 2 | 24.267 | 22.603–25.931 | 17.503 | 47.238 |

## pagebroker-isolated64-1

PageBroker consumed 41.666 CPU-seconds (9.747 user, 31.919 system) over the 42.242s sample interval: 0.986 cores on average across that interval. Its cgroup quota was 64 cores and scheduling weight 79. CPU pressure occupied 0.152% of that interval; 0 of 250 accounted periods were throttled.

The Snapshot agent added 0.099 CPU-seconds with 0.000000s throttling. No engine Pod was created in this isolated load.

Isolated ring acquisition waits were negligible: 2.849 microseconds mean/rank and 5.055 microseconds maximum/rank. There were no native residual transfers competing for the per-GPU pinned ring. These acquisition timings do not establish meaningful shared-ring contention.

Across eight equal 56GiB ranks, mean active read+copy time was 13.382s and mean CUDA event-wait time was 1.155s. Mean summed storage-request service time was 249.050s/rank; its overlapping requests make this unsuitable for addition to wall time.

## pagebroker-native16-1

PageBroker consumed 47.908 CPU-seconds (11.024 user, 36.884 system) over the 63.782s sample interval: 0.751 cores on average across that interval. Its cgroup quota was 16 cores and scheduling weight 79. CPU pressure occupied 0.095% of that interval; 0 of 346 accounted periods were throttled.

The Snapshot agent added 0.666 CPU-seconds with 0.000000s throttling. The engine main container had 0.000000s cumulative throttling across its 57.237s observed lifetime, including post-restore inference and collection.

There is measurable shared-ring contention: GMS waits total 6.726s across eight ranks (0.841s mean/rank, 1.123s maximum/rank), while the 8 native residual transfers wait 4.352s in total (0.544s mean, 0.941s maximum). These are waits for the common per-GPU pinned ring, independently of CPU quota or CPU scheduling. They prove serialization in the shared transfer path, not that all of that sum extends the critical path.

Across eight equal 56GiB ranks, mean active read+copy time was 16.522s and mean CUDA event-wait time was 1.178s. Mean summed storage-request service time was 318.153s/rank; its overlapping requests make this unsuitable for addition to wall time.

## pagebroker-native16-2

PageBroker consumed 47.013 CPU-seconds (11.102 user, 35.911 system) over the 64.627s sample interval: 0.727 cores on average across that interval. Its cgroup quota was 16 cores and scheduling weight 79. CPU pressure occupied 0.097% of that interval; 0 of 338 accounted periods were throttled.

The Snapshot agent added 0.706 CPU-seconds with 0.000000s throttling. The engine main container had 0.000000s cumulative throttling across its 58.522s observed lifetime, including post-restore inference and collection.

There is measurable shared-ring contention: GMS waits total 6.304s across eight ranks (0.788s mean/rank, 0.950s maximum/rank), while the 8 native residual transfers wait 6.198s in total (0.775s mean, 1.318s maximum). These are waits for the common per-GPU pinned ring, independently of CPU quota or CPU scheduling. They prove serialization in the shared transfer path, not that all of that sum extends the critical path.

Across eight equal 56GiB ranks, mean active read+copy time was 16.595s and mean CUDA event-wait time was 1.183s. Mean summed storage-request service time was 353.228s/rank; its overlapping requests make this unsuitable for addition to wall time.

## pagebroker-native64-1

PageBroker consumed 46.905 CPU-seconds (11.009 user, 35.896 system) over the 67.137s sample interval: 0.699 cores on average across that interval. Its cgroup quota was 64 cores and scheduling weight 79. CPU pressure occupied 0.101% of that interval; 0 of 354 accounted periods were throttled.

The Snapshot agent added 0.814 CPU-seconds with 0.000000s throttling. The engine main container had 0.000000s cumulative throttling across its 55.383s observed lifetime, including post-restore inference and collection.

There is measurable shared-ring contention: GMS waits total 3.485s across eight ranks (0.436s mean/rank, 1.013s maximum/rank), while the 8 native residual transfers wait 2.116s in total (0.264s mean, 1.055s maximum). These are waits for the common per-GPU pinned ring, independently of CPU quota or CPU scheduling. They prove serialization in the shared transfer path, not that all of that sum extends the critical path.

Across eight equal 56GiB ranks, mean active read+copy time was 15.949s and mean CUDA event-wait time was 1.182s. Mean summed storage-request service time was 317.716s/rank; its overlapping requests make this unsuitable for addition to wall time.

## pagebroker-native64-2

PageBroker consumed 47.572 CPU-seconds (10.987 user, 36.584 system) over the 64.487s sample interval: 0.738 cores on average across that interval. Its cgroup quota was 64 cores and scheduling weight 79. CPU pressure occupied 0.081% of that interval; 0 of 348 accounted periods were throttled.

The Snapshot agent added 0.699 CPU-seconds with 0.000000s throttling. The engine main container had 0.000000s cumulative throttling across its 55.746s observed lifetime, including post-restore inference and collection.

There is measurable shared-ring contention: GMS waits total 5.634s across eight ranks (0.704s mean/rank, 1.109s maximum/rank), while the 8 native residual transfers wait 4.047s in total (0.506s mean, 1.015s maximum). These are waits for the common per-GPU pinned ring, independently of CPU quota or CPU scheduling. They prove serialization in the shared transfer path, not that all of that sum extends the critical path.

Across eight equal 56GiB ranks, mean active read+copy time was 16.332s and mean CUDA event-wait time was 1.176s. Mean summed storage-request service time was 343.815s/rank; its overlapping requests make this unsuitable for addition to wall time.

## Equal-payload isolated comparison: pagebroker-native64-1

pagebroker-isolated64-1 and pagebroker-native64-1 used the same PageBroker Pod, CPU resources, destination GPUs and captured artifact digests. Both transferred exactly 448GiB in 112 shards. The concurrent run's payload window was 3.547s longer (25.57%), with 20.37% lower effective payload throughput.

The concurrent engine restore additionally transferred 41.234375GiB of GPU residual state through PageBroker. Its CRIU CPU-image I/O is additional but unmeasured by these native GPU reports. The isolated run created no engine Pod and logged no native residual transfers during its interval.

This single matched pair supports transfer contention: ring waits disappear in isolation, active read+copy time and summed I/O service latency fall, while CUDA event waits change little and CPU throttling stays zero. The exact wall-time difference is an observation from n=1 per mode, not a repeatable causal estimate; storage and orchestration variance remain possible contributors.

## Equal-payload isolated comparison: pagebroker-native64-2

pagebroker-isolated64-1 and pagebroker-native64-2 used the same PageBroker Pod, CPU resources, destination GPUs and captured artifact digests. Both transferred exactly 448GiB in 112 shards. The concurrent run's payload window was 3.725s longer (26.86%), with 21.17% lower effective payload throughput.

The concurrent engine restore additionally transferred 41.234375GiB of GPU residual state through PageBroker. Its CRIU CPU-image I/O is additional but unmeasured by these native GPU reports. The isolated run created no engine Pod and logged no native residual transfers during its interval.

This single matched pair supports transfer contention: ring waits disappear in isolation, active read+copy time and summed I/O service latency fall, while CUDA event waits change little and CPU throttling stays zero. The exact wall-time difference is an observation from n=1 per mode, not a repeatable causal estimate; storage and orchestration variance remain possible contributors.

## Interpretation limits

- Before/after CPU sampling includes idle lead-in and evidence collection after Ready; mean cores is not restore-only utilization or peak demand.
- CPU quota throttling and scheduler CPU pressure differ from waiting for the PageBroker per-GPU transfer ring.
- CPU request controls relative scheduling weight under competition, not a hard core cap.
- Parent cgroup counters and node-wide runnable/CPU occupancy were not collected; low leaf CPU pressure supports but does not prove absence of all host competition.
- Storage request service time sums overlapping in-flight I/O latency and is neither CPU time nor a serial wall-time component. CUDA wait sums host event synchronization waits.
- Rank and native ring waits overlap across GPUs; their sums cannot be added directly to end-to-end Ready time.
- Effective payload throughput divides 448GiB by earliest read to final transfer-routine completion, including intervening ring waits; it is not raw storage or PCIe bandwidth.

Keep the 16-CPU limit for now: there is no measured throttling at 16 CPUs and no repeatable Ready-time benefit from the 64-CPU limit. The 64-CPU payload window was shorter, but native/GMS ring overlap also changed; these small sequential cohorts cannot isolate a CPU-limit effect. A request/weight experiment becomes useful if scheduler pressure rises during a busy-node workload.
