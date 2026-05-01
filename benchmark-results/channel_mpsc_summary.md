# Runtime MPSC Channel Benchmark Results

## Summary

- Added `dynamo_runtime::channels::mpsc`, a facade that uses Tokio channels by default and flume async channels with the `flume-channels` feature.
- Routed selected runtime transport channel paths through the facade so they can be built against either backend.
- Added the `channel_mpsc_perf` Criterion microbenchmark for bounded SPSC, bounded MPSC, and unbounded SPSC channel traffic.
- Recorded long-run Tokio and flume measurements with the original capacities plus 2x and 4x higher caps.

## Validation

```bash
cargo fmt --check
cargo check -p dynamo-runtime --benches
cargo check -p dynamo-runtime --features flume-channels --benches
cargo test -p dynamo-runtime channels
cargo test -p dynamo-runtime --features flume-channels channels
```

All validation commands passed.

## Benchmark Results

Date: 2026-05-01 UTC
Base commit: `563cd37d7b9904901f668833dc2ea704d38f9e35`

Commands:

```bash
cargo bench -p dynamo-runtime --bench channel_mpsc_perf -- --noplot
cargo bench -p dynamo-runtime --features flume-channels --bench channel_mpsc_perf -- --noplot
```

Long-run settings:

- Criterion warmup: 10 seconds per benchmark case
- Criterion measurement target: 30 seconds per benchmark case
- Bounded SPSC capacities: `1, 2, 4, 64, 128, 256, 1024, 2048, 4096`
- Bounded MPSC capacities: `64, 128, 256, 1024, 2048, 4096`

Raw logs:

- `benchmark-results/channel_mpsc_tokio_long.log`
- `benchmark-results/channel_mpsc_flume_long.log`
- Earlier short-run logs are kept in the same directory for comparison.

Criterion point estimates from the long run:

| Benchmark | Tokio time | Flume time | Flume speedup | Tokio throughput | Flume throughput |
| --- | ---: | ---: | ---: | ---: | ---: |
| Bounded SPSC, cap 1 | 335.12 ms | 197.73 ms | 1.69x | 48.890 Kelem/s | 82.860 Kelem/s |
| Bounded SPSC, cap 2 | 170.33 ms | 149.24 ms | 1.14x | 96.188 Kelem/s | 109.78 Kelem/s |
| Bounded SPSC, cap 4 | 85.007 ms | 169.76 ms | 0.50x | 192.74 Kelem/s | 96.512 Kelem/s |
| Bounded SPSC, cap 64 | 5.3372 ms | 9.2099 ms | 0.58x | 3.0698 Melem/s | 1.7789 Melem/s |
| Bounded SPSC, cap 128 | 1.7618 ms | 3.7887 ms | 0.47x | 9.2995 Melem/s | 4.3244 Melem/s |
| Bounded SPSC, cap 256 | 1.4689 ms | 2.4872 ms | 0.59x | 11.154 Melem/s | 6.5872 Melem/s |
| Bounded SPSC, cap 1024 | 1.4030 ms | 1.2269 ms | 1.14x | 11.678 Melem/s | 13.354 Melem/s |
| Bounded SPSC, cap 2048 | 1.4025 ms | 1.0167 ms | 1.38x | 11.682 Melem/s | 16.116 Melem/s |
| Bounded SPSC, cap 4096 | 1.3573 ms | 951.00 us | 1.43x | 12.071 Melem/s | 17.228 Melem/s |
| Bounded MPSC, cap 64 | 1.7388 ms | 1.1456 ms | 1.52x | 9.4228 Melem/s | 14.301 Melem/s |
| Bounded MPSC, cap 128 | 1.7310 ms | 1.0066 ms | 1.72x | 9.4651 Melem/s | 16.276 Melem/s |
| Bounded MPSC, cap 256 | 1.6653 ms | 1.0172 ms | 1.64x | 9.8384 Melem/s | 16.107 Melem/s |
| Bounded MPSC, cap 1024 | 1.8078 ms | 1.4042 ms | 1.29x | 9.0630 Melem/s | 11.668 Melem/s |
| Bounded MPSC, cap 2048 | 1.7146 ms | 1.4645 ms | 1.17x | 9.5558 Melem/s | 11.187 Melem/s |
| Bounded MPSC, cap 4096 | 1.8779 ms | 936.83 us | 2.00x | 8.7246 Melem/s | 17.489 Melem/s |
| Unbounded SPSC | 1.2724 ms | 776.21 us | 1.64x | 12.876 Melem/s | 21.108 Melem/s |

Long-run takeaway: flume wins every bounded MPSC case and unbounded SPSC. For bounded SPSC, flume wins at capacity `1`, `2`, and `1024+`, while Tokio wins at capacity `4` through `256`.
