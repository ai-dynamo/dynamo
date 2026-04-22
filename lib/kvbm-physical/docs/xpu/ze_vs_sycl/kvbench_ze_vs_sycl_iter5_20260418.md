# kvbench — Level Zero vs SYCL Side-by-Side Comparison

> **Run:** `20260418_171130` | **Sources:** `kvbench_ze_iter5.log` + `kvbench_sycl_iter5.log` | **Iteration:** 5 of 5
> **Hardware:** Intel Arc B580, 160 EUs, 2900 MHz, PCIe Gen4 x8

Compares raw kvbench throughput between Level Zero (ZE) and SYCL backends
for all direction × pattern × backend × host combinations.

Ratio = SYCL / ZE (>1.00 means SYCL faster).

## Production-Size Comparison: N=256, tpb=64

| Dir | Pattern | Backend | Host | SYCL/ZE |
|-----|---------|---------|------|--------:|
| D2D | fc_to_fc | memcpy | - | 1.62x |
| D2D | fc_to_fc | vectorized | - | 0.99x |
| D2D | lw_to_fc | vectorized | - | 1.00x |
| D2D | lw_to_fc | memcpy | - | 1.08x |
| D2D | fc_to_lw | vectorized | - | 1.00x |
| D2D | fc_to_lw | memcpy | - | 1.08x |
| H2D | fc_to_fc | memcpy | pinned | 1.00x |
| H2D | fc_to_fc | vectorized | pinned | 1.00x |
| H2D | fc_to_fc | memcpy | system | 1.00x |
| H2D | lw_to_fc | vectorized | pinned | 1.00x |
| H2D | lw_to_fc | memcpy | pinned | 1.00x |
| H2D | lw_to_fc | memcpy | system | 0.98x |
| H2D | fc_to_lw | vectorized | pinned | 1.00x |
| H2D | fc_to_lw | memcpy | pinned | 1.00x |
| H2D | fc_to_lw | memcpy | system | 0.96x |
| D2H | fc_to_fc | memcpy | pinned | 1.00x |
| D2H | fc_to_fc | vectorized | pinned | 1.02x |
| D2H | fc_to_fc | memcpy | system | 1.00x |
| D2H | lw_to_fc | vectorized | pinned | 1.00x |
| D2H | lw_to_fc | memcpy | pinned | 1.00x |
| D2H | lw_to_fc | memcpy | system | 0.95x |
| D2H | fc_to_lw | vectorized | pinned | 1.07x |
| D2H | fc_to_lw | memcpy | pinned | 1.00x |
| D2H | fc_to_lw | memcpy | system | 0.98x |

## Full-Sweep Average (All 27 N×tpb Points)

| Dir | Pattern | Backend | Host | SYCL/ZE |
|-----|---------|---------|------|--------:|
| D2D | fc_to_fc | memcpy | - | 1.56x |
| D2D | fc_to_fc | vectorized | - | 1.00x |
| D2D | lw_to_fc | vectorized | - | 0.98x |
| D2D | lw_to_fc | memcpy | - | 0.97x |
| D2D | fc_to_lw | vectorized | - | 0.97x |
| D2D | fc_to_lw | memcpy | - | 0.97x |
| H2D | fc_to_fc | memcpy | pinned | 1.00x |
| H2D | fc_to_fc | vectorized | pinned | 1.00x |
| H2D | fc_to_fc | memcpy | system | 1.00x |
| H2D | lw_to_fc | vectorized | pinned | 1.00x |
| H2D | lw_to_fc | memcpy | pinned | 1.00x |
| H2D | lw_to_fc | memcpy | system | 0.99x |
| H2D | fc_to_lw | vectorized | pinned | 1.00x |
| H2D | fc_to_lw | memcpy | pinned | 1.00x |
| H2D | fc_to_lw | memcpy | system | 1.00x |
| D2H | fc_to_fc | memcpy | pinned | 1.00x |
| D2H | fc_to_fc | vectorized | pinned | 1.00x |
| D2H | fc_to_fc | memcpy | system | 0.99x |
| D2H | lw_to_fc | vectorized | pinned | 1.00x |
| D2H | lw_to_fc | memcpy | pinned | 1.00x |
| D2H | lw_to_fc | memcpy | system | 0.96x |
| D2H | fc_to_lw | vectorized | pinned | 1.00x |
| D2H | fc_to_lw | memcpy | pinned | 1.00x |
| D2H | fc_to_lw | memcpy | system | 0.95x |

## Full N × tpb Side-by-Side Grids

Each cell: `ZE | SYCL | ratio` (ratio = SYCL/ZE)

### D2D / fc_to_fc / memcpy

| N | Ratio(16) | Ratio(32) | Ratio(64) |
|--:|----------:|----------:|----------:|
| 1 | 1.75x | 1.93x | 1.10x |
| 2 | 1.50x | 1.12x | 1.60x |
| 4 | 1.19x | 1.58x | 1.61x |
| 8 | 1.60x | 1.58x | 1.61x |
| 16 | 1.61x | 1.59x | 1.61x |
| 32 | 1.62x | 1.60x | 1.62x |
| 64 | 1.64x | 1.60x | 1.62x |
| 128 | 1.69x | 1.60x | 1.62x |
| 256 | 1.68x | 1.60x | 1.62x |

**SYCL/ZE:** peak 1.93x, avg 1.56x

### D2D / fc_to_fc / vectorized

| N | Ratio(16) | Ratio(32) | Ratio(64) |
|--:|----------:|----------:|----------:|
| 1 | 1.04x | 1.18x | 1.01x |
| 2 | 1.04x | 1.01x | 1.01x |
| 4 | 1.00x | 0.99x | 1.00x |
| 8 | 0.99x | 1.00x | 1.01x |
| 16 | 1.00x | 1.00x | 1.00x |
| 32 | 1.00x | 1.00x | 1.00x |
| 64 | 1.00x | 1.00x | 1.00x |
| 128 | 1.01x | 1.00x | 0.99x |
| 256 | 1.00x | 1.00x | 0.99x |

**SYCL/ZE:** peak 1.00x, avg 1.00x

### D2D / lw_to_fc / vectorized

| N | Ratio(16) | Ratio(32) | Ratio(64) |
|--:|----------:|----------:|----------:|
| 1 | 0.80x | 0.92x | 0.97x |
| 2 | 0.95x | 0.97x | 0.97x |
| 4 | 0.97x | 0.98x | 0.98x |
| 8 | 0.98x | 1.00x | 0.99x |
| 16 | 0.98x | 0.99x | 1.00x |
| 32 | 0.99x | 1.00x | 0.99x |
| 64 | 1.00x | 1.00x | 1.00x |
| 128 | 1.00x | 1.00x | 1.00x |
| 256 | 1.00x | 1.00x | 1.00x |

**SYCL/ZE:** peak 0.95x, avg 0.98x

### D2D / fc_to_lw / vectorized

| N | Ratio(16) | Ratio(32) | Ratio(64) |
|--:|----------:|----------:|----------:|
| 1 | 0.80x | 0.91x | 0.96x |
| 2 | 0.90x | 0.97x | 0.97x |
| 4 | 0.97x | 1.01x | 0.95x |
| 8 | 0.98x | 1.02x | 1.00x |
| 16 | 0.99x | 1.00x | 1.00x |
| 32 | 0.98x | 1.01x | 0.98x |
| 64 | 0.99x | 0.99x | 1.00x |
| 128 | 1.00x | 1.02x | 1.00x |
| 256 | 1.01x | 1.00x | 1.00x |

**SYCL/ZE:** peak 0.90x, avg 0.97x

### H2D / fc_to_fc / memcpy, pinned

| N | Ratio(16) | Ratio(32) | Ratio(64) |
|--:|----------:|----------:|----------:|
| 1 | 1.00x | 1.00x | 1.00x |
| 2 | 1.00x | 1.00x | 1.00x |
| 4 | 1.00x | 1.00x | 1.00x |
| 8 | 1.00x | 1.00x | 1.00x |
| 16 | 1.00x | 1.00x | 1.00x |
| 32 | 1.00x | 1.00x | 1.00x |
| 64 | 1.00x | 1.00x | 1.00x |
| 128 | 1.00x | 1.00x | 1.00x |
| 256 | 1.00x | 1.00x | 1.00x |

**SYCL/ZE:** peak 1.00x, avg 1.00x

### H2D / lw_to_fc / vectorized, pinned

| N | Ratio(16) | Ratio(32) | Ratio(64) |
|--:|----------:|----------:|----------:|
| 1 | 0.99x | 1.00x | 1.00x |
| 2 | 1.00x | 1.00x | 1.00x |
| 4 | 1.00x | 1.00x | 1.00x |
| 8 | 1.00x | 1.00x | 1.00x |
| 16 | 1.00x | 1.00x | 1.00x |
| 32 | 1.00x | 0.99x | 1.00x |
| 64 | 0.99x | 1.00x | 1.00x |
| 128 | 1.00x | 1.00x | 1.00x |
| 256 | 1.00x | 1.00x | 1.00x |

**SYCL/ZE:** peak 1.00x, avg 1.00x

### H2D / fc_to_lw / vectorized, pinned

| N | Ratio(16) | Ratio(32) | Ratio(64) |
|--:|----------:|----------:|----------:|
| 1 | 0.99x | 1.00x | 1.00x |
| 2 | 1.00x | 1.00x | 1.00x |
| 4 | 1.00x | 1.00x | 1.00x |
| 8 | 1.00x | 1.00x | 1.00x |
| 16 | 1.00x | 1.00x | 1.00x |
| 32 | 1.00x | 0.99x | 1.00x |
| 64 | 0.99x | 1.00x | 1.00x |
| 128 | 1.00x | 1.00x | 1.00x |
| 256 | 1.00x | 1.00x | 1.00x |

**SYCL/ZE:** peak 1.00x, avg 1.00x

### D2H / fc_to_fc / memcpy, pinned

| N | Ratio(16) | Ratio(32) | Ratio(64) |
|--:|----------:|----------:|----------:|
| 1 | 1.00x | 1.00x | 1.00x |
| 2 | 1.00x | 1.00x | 1.00x |
| 4 | 1.00x | 1.00x | 1.00x |
| 8 | 1.00x | 1.00x | 1.00x |
| 16 | 1.00x | 1.00x | 1.00x |
| 32 | 1.00x | 1.00x | 1.00x |
| 64 | 1.00x | 1.00x | 1.00x |
| 128 | 1.00x | 1.00x | 1.00x |
| 256 | 1.00x | 1.00x | 1.00x |

**SYCL/ZE:** peak 1.00x, avg 1.00x

### D2H / lw_to_fc / vectorized, pinned

| N | Ratio(16) | Ratio(32) | Ratio(64) |
|--:|----------:|----------:|----------:|
| 1 | 1.00x | 1.00x | 1.00x |
| 2 | 1.00x | 1.00x | 1.00x |
| 4 | 1.00x | 1.00x | 1.00x |
| 8 | 1.00x | 1.00x | 1.00x |
| 16 | 1.00x | 1.00x | 0.99x |
| 32 | 1.00x | 1.00x | 1.00x |
| 64 | 0.99x | 1.00x | 1.00x |
| 128 | 1.00x | 1.00x | 1.00x |
| 256 | 1.00x | 1.00x | 1.00x |

**SYCL/ZE:** peak 1.00x, avg 1.00x

### D2H / fc_to_lw / vectorized, pinned

| N | Ratio(16) | Ratio(32) | Ratio(64) |
|--:|----------:|----------:|----------:|
| 1 | 0.99x | 1.00x | 1.00x |
| 2 | 1.00x | 1.00x | 1.00x |
| 4 | 1.00x | 1.00x | 1.00x |
| 8 | 1.00x | 1.00x | 1.00x |
| 16 | 1.00x | 1.00x | 0.99x |
| 32 | 1.00x | 1.00x | 0.98x |
| 64 | 1.02x | 0.98x | 1.00x |
| 128 | 0.98x | 1.00x | 1.00x |
| 256 | 1.00x | 1.00x | 1.07x |

**SYCL/ZE:** peak 1.00x, avg 1.00x
