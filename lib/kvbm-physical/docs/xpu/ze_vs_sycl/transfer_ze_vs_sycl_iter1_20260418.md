# TransferManager — Level Zero vs SYCL Side-by-Side Comparison

> **Run:** `20260418_171130` | **Sources:** `transfer_ze_iter1.log` + `transfer_sycl_iter1.log` | **Iteration:** 1 of 5
> **Hardware:** Intel Arc B580, 160 EUs, 2900 MHz, PCIe Gen4 x8

Compares TransferManager throughput between Level Zero (ZE) and SYCL backends
for all direction × pattern combinations.

Ratio = SYCL / ZE (>1.00 means SYCL faster).

## Production-Size Comparison: N=256, tpb=64

| Dir | Pattern | Host | SYCL/ZE |
|-----|---------|------|--------:|
| D2D | fc_to_fc | - | 1.60x |
| D2D | lw_to_fc | - | 0.99x |
| D2D | fc_to_lw | - | 1.00x |
| H2D | fc_to_fc | pinned | 1.00x |
| H2D | lw_to_fc | pinned | 1.00x |
| H2D | fc_to_lw | pinned | 1.00x |
| D2H | fc_to_fc | pinned | 1.00x |
| D2H | lw_to_fc | pinned | 1.00x |
| D2H | fc_to_lw | pinned | 0.99x |

## Full-Sweep Average (All 27 N×tpb Points)

| Dir | Pattern | Host | SYCL/ZE |
|-----|---------|------|--------:|
| D2D | fc_to_fc | - | 1.32x |
| D2D | lw_to_fc | - | 1.05x |
| D2D | fc_to_lw | - | 1.03x |
| H2D | fc_to_fc | pinned | 1.01x |
| H2D | lw_to_fc | pinned | 1.01x |
| H2D | fc_to_lw | pinned | 1.00x |
| D2H | fc_to_fc | pinned | 1.01x |
| D2H | lw_to_fc | pinned | 1.01x |
| D2H | fc_to_lw | pinned | 1.00x |

## Full N × tpb Side-by-Side Grids

Each cell: `ZE | SYCL | ratio`

### D2D / fc_to_fc

| N | Ratio(16) | Ratio(32) | Ratio(64) |
|--:|----------:|----------:|----------:|
| 1 | 0.99x | 0.88x | 1.03x |
| 2 | 0.98x | 1.07x | 1.05x |
| 4 | 0.97x | 1.05x | 1.04x |
| 8 | 0.95x | 1.18x | 1.22x |
| 16 | 0.93x | 1.07x | 1.28x |
| 32 | 0.96x | 0.94x | 1.42x |
| 64 | 1.14x | 1.17x | 1.66x |
| 128 | 1.42x | 1.52x | 1.55x |
| 256 | 1.51x | 1.55x | 1.60x |

**SYCL/ZE:** peak 1.60x, avg 1.32x

### D2D / lw_to_fc

| N | Ratio(16) | Ratio(32) | Ratio(64) |
|--:|----------:|----------:|----------:|
| 1 | 0.98x | 0.98x | 1.08x |
| 2 | 0.98x | 1.07x | 1.08x |
| 4 | 0.98x | 1.08x | 1.10x |
| 8 | 0.96x | 1.11x | 1.23x |
| 16 | 0.98x | 1.11x | 1.12x |
| 32 | 1.00x | 1.01x | 1.04x |
| 64 | 1.15x | 1.19x | 1.02x |
| 128 | 1.00x | 1.03x | 1.07x |
| 256 | 0.94x | 1.07x | 0.99x |

**SYCL/ZE:** peak 1.02x, avg 1.05x

### D2D / fc_to_lw

| N | Ratio(16) | Ratio(32) | Ratio(64) |
|--:|----------:|----------:|----------:|
| 1 | 0.98x | 1.06x | 1.07x |
| 2 | 0.98x | 1.07x | 1.07x |
| 4 | 0.98x | 1.05x | 1.10x |
| 8 | 0.96x | 1.11x | 1.22x |
| 16 | 0.97x | 1.10x | 1.11x |
| 32 | 1.00x | 1.01x | 1.04x |
| 64 | 1.06x | 1.19x | 1.05x |
| 128 | 0.99x | 0.82x | 1.08x |
| 256 | 0.94x | 1.05x | 1.00x |

**SYCL/ZE:** peak 1.01x, avg 1.03x

### H2D / fc_to_fc (host=pinned)

| N | Ratio(16) | Ratio(32) | Ratio(64) |
|--:|----------:|----------:|----------:|
| 1 | 0.99x | 0.90x | 1.06x |
| 2 | 0.99x | 1.08x | 1.10x |
| 4 | 0.97x | 1.08x | 1.08x |
| 8 | 0.97x | 1.08x | 0.98x |
| 16 | 0.97x | 0.98x | 0.99x |
| 32 | 0.98x | 0.99x | 1.01x |
| 64 | 1.02x | 0.99x | 1.01x |
| 128 | 1.00x | 1.00x | 1.01x |
| 256 | 1.00x | 1.00x | 1.00x |

**SYCL/ZE:** peak 1.01x, avg 1.01x

### H2D / lw_to_fc (host=pinned)

| N | Ratio(16) | Ratio(32) | Ratio(64) |
|--:|----------:|----------:|----------:|
| 1 | 0.97x | 0.89x | 1.07x |
| 2 | 0.97x | 1.08x | 1.11x |
| 4 | 0.93x | 1.09x | 1.08x |
| 8 | 1.00x | 1.08x | 0.99x |
| 16 | 0.98x | 1.00x | 0.99x |
| 32 | 0.99x | 0.99x | 0.99x |
| 64 | 1.02x | 0.99x | 1.00x |
| 128 | 0.99x | 0.99x | 1.00x |
| 256 | 1.01x | 1.00x | 1.00x |

**SYCL/ZE:** peak 1.00x, avg 1.01x

### H2D / fc_to_lw (host=pinned)

| N | Ratio(16) | Ratio(32) | Ratio(64) |
|--:|----------:|----------:|----------:|
| 1 | 0.97x | 0.89x | 1.06x |
| 2 | 0.97x | 1.07x | 1.00x |
| 4 | 0.94x | 1.10x | 1.08x |
| 8 | 0.99x | 1.09x | 1.01x |
| 16 | 0.98x | 0.99x | 0.99x |
| 32 | 0.99x | 0.99x | 0.99x |
| 64 | 1.00x | 0.99x | 1.00x |
| 128 | 1.00x | 1.00x | 1.00x |
| 256 | 1.01x | 1.00x | 1.00x |

**SYCL/ZE:** peak 1.00x, avg 1.00x

### D2H / fc_to_fc (host=pinned)

| N | Ratio(16) | Ratio(32) | Ratio(64) |
|--:|----------:|----------:|----------:|
| 1 | 0.99x | 0.88x | 1.08x |
| 2 | 0.98x | 1.09x | 0.97x |
| 4 | 0.97x | 1.10x | 1.05x |
| 8 | 0.96x | 0.93x | 1.06x |
| 16 | 0.98x | 1.07x | 1.03x |
| 32 | 0.97x | 1.03x | 1.01x |
| 64 | 1.02x | 1.00x | 1.01x |
| 128 | 1.02x | 1.00x | 1.00x |
| 256 | 1.00x | 1.00x | 1.00x |

**SYCL/ZE:** peak 1.00x, avg 1.01x

### D2H / lw_to_fc (host=pinned)

| N | Ratio(16) | Ratio(32) | Ratio(64) |
|--:|----------:|----------:|----------:|
| 1 | 0.97x | 0.90x | 1.07x |
| 2 | 0.98x | 1.04x | 1.00x |
| 4 | 0.98x | 1.07x | 1.08x |
| 8 | 0.98x | 1.08x | 1.00x |
| 16 | 1.00x | 1.01x | 1.00x |
| 32 | 0.99x | 1.01x | 1.00x |
| 64 | 1.01x | 1.00x | 1.01x |
| 128 | 1.00x | 1.00x | 1.00x |
| 256 | 1.00x | 1.00x | 1.00x |

**SYCL/ZE:** peak 1.00x, avg 1.01x

### D2H / fc_to_lw (host=pinned)

| N | Ratio(16) | Ratio(32) | Ratio(64) |
|--:|----------:|----------:|----------:|
| 1 | 0.93x | 0.90x | 1.07x |
| 2 | 0.96x | 1.06x | 1.00x |
| 4 | 0.98x | 1.06x | 1.07x |
| 8 | 0.98x | 1.09x | 1.00x |
| 16 | 1.00x | 1.01x | 1.00x |
| 32 | 0.99x | 1.00x | 1.00x |
| 64 | 0.99x | 1.02x | 1.01x |
| 128 | 1.00x | 0.99x | 1.01x |
| 256 | 0.94x | 0.99x | 0.99x |

**SYCL/ZE:** peak 1.00x, avg 1.00x
