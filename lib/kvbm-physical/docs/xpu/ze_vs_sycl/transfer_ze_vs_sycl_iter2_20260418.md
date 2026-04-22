# TransferManager — Level Zero vs SYCL Side-by-Side Comparison

> **Run:** `20260418_171130` | **Sources:** `transfer_ze_iter2.log` + `transfer_sycl_iter2.log` | **Iteration:** 2 of 5
> **Hardware:** Intel Arc B580, 160 EUs, 2900 MHz, PCIe Gen4 x8

Compares TransferManager throughput between Level Zero (ZE) and SYCL backends
for all direction × pattern combinations.

Ratio = SYCL / ZE (>1.00 means SYCL faster).

## Production-Size Comparison: N=256, tpb=64

| Dir | Pattern | Host | SYCL/ZE |
|-----|---------|------|--------:|
| D2D | fc_to_fc | - | 1.56x |
| D2D | lw_to_fc | - | 0.98x |
| D2D | fc_to_lw | - | 0.97x |
| H2D | fc_to_fc | pinned | 1.00x |
| H2D | lw_to_fc | pinned | 1.00x |
| H2D | fc_to_lw | pinned | 1.00x |
| D2H | fc_to_fc | pinned | 1.00x |
| D2H | lw_to_fc | pinned | 1.00x |
| D2H | fc_to_lw | pinned | 1.00x |

## Full-Sweep Average (All 27 N×tpb Points)

| Dir | Pattern | Host | SYCL/ZE |
|-----|---------|------|--------:|
| D2D | fc_to_fc | - | 1.37x |
| D2D | lw_to_fc | - | 1.00x |
| D2D | fc_to_lw | - | 1.00x |
| H2D | fc_to_fc | pinned | 1.00x |
| H2D | lw_to_fc | pinned | 0.99x |
| H2D | fc_to_lw | pinned | 0.99x |
| D2H | fc_to_fc | pinned | 1.00x |
| D2H | lw_to_fc | pinned | 1.00x |
| D2H | fc_to_lw | pinned | 1.00x |

## Full N × tpb Side-by-Side Grids

Each cell: `ZE | SYCL | ratio`

### D2D / fc_to_fc

| N | Ratio(16) | Ratio(32) | Ratio(64) |
|--:|----------:|----------:|----------:|
| 1 | 1.01x | 1.12x | 1.15x |
| 2 | 0.99x | 0.98x | 1.15x |
| 4 | 0.97x | 0.87x | 0.85x |
| 8 | 1.11x | 0.95x | 0.72x |
| 16 | 0.88x | 0.94x | 1.70x |
| 32 | 0.86x | 1.94x | 1.29x |
| 64 | 1.87x | 1.47x | 1.53x |
| 128 | 1.41x | 1.61x | 1.50x |
| 256 | 1.38x | 1.45x | 1.56x |

**SYCL/ZE:** peak 1.56x, avg 1.37x

### D2D / lw_to_fc

| N | Ratio(16) | Ratio(32) | Ratio(64) |
|--:|----------:|----------:|----------:|
| 1 | 0.99x | 1.13x | 1.12x |
| 2 | 0.99x | 0.99x | 1.12x |
| 4 | 1.00x | 0.98x | 0.86x |
| 8 | 1.11x | 0.98x | 0.85x |
| 16 | 0.99x | 0.99x | 0.84x |
| 32 | 0.87x | 1.00x | 0.90x |
| 64 | 0.98x | 1.03x | 1.14x |
| 128 | 1.00x | 1.29x | 1.01x |
| 256 | 0.94x | 1.13x | 0.98x |

**SYCL/ZE:** peak 0.97x, avg 1.00x

### D2D / fc_to_lw

| N | Ratio(16) | Ratio(32) | Ratio(64) |
|--:|----------:|----------:|----------:|
| 1 | 0.95x | 1.12x | 1.13x |
| 2 | 1.00x | 0.99x | 1.15x |
| 4 | 1.00x | 0.98x | 0.86x |
| 8 | 1.11x | 0.97x | 0.85x |
| 16 | 0.98x | 0.98x | 0.86x |
| 32 | 0.89x | 1.00x | 0.89x |
| 64 | 0.97x | 1.03x | 1.17x |
| 128 | 0.99x | 1.01x | 1.08x |
| 256 | 0.95x | 1.19x | 0.97x |

**SYCL/ZE:** peak 1.03x, avg 1.00x

### H2D / fc_to_fc (host=pinned)

| N | Ratio(16) | Ratio(32) | Ratio(64) |
|--:|----------:|----------:|----------:|
| 1 | 0.97x | 1.14x | 0.98x |
| 2 | 0.96x | 0.99x | 1.00x |
| 4 | 0.97x | 0.97x | 0.98x |
| 8 | 0.97x | 0.98x | 0.99x |
| 16 | 0.98x | 0.99x | 1.00x |
| 32 | 0.98x | 1.03x | 1.00x |
| 64 | 1.03x | 1.01x | 1.00x |
| 128 | 1.00x | 1.00x | 1.00x |
| 256 | 1.01x | 1.00x | 1.00x |

**SYCL/ZE:** peak 1.00x, avg 1.00x

### H2D / lw_to_fc (host=pinned)

| N | Ratio(16) | Ratio(32) | Ratio(64) |
|--:|----------:|----------:|----------:|
| 1 | 0.98x | 1.12x | 0.99x |
| 2 | 0.97x | 0.99x | 0.99x |
| 4 | 0.99x | 0.95x | 0.97x |
| 8 | 0.98x | 0.98x | 1.00x |
| 16 | 0.98x | 0.99x | 0.99x |
| 32 | 0.99x | 1.00x | 1.00x |
| 64 | 0.98x | 1.00x | 1.00x |
| 128 | 1.00x | 1.00x | 1.00x |
| 256 | 1.00x | 1.00x | 1.00x |

**SYCL/ZE:** peak 1.00x, avg 0.99x

### H2D / fc_to_lw (host=pinned)

| N | Ratio(16) | Ratio(32) | Ratio(64) |
|--:|----------:|----------:|----------:|
| 1 | 0.99x | 1.12x | 0.98x |
| 2 | 0.98x | 0.98x | 1.00x |
| 4 | 0.96x | 0.95x | 0.98x |
| 8 | 0.98x | 0.96x | 1.00x |
| 16 | 0.98x | 0.99x | 0.99x |
| 32 | 0.99x | 1.00x | 1.00x |
| 64 | 0.99x | 1.00x | 1.00x |
| 128 | 1.00x | 1.00x | 1.00x |
| 256 | 0.99x | 1.00x | 1.00x |

**SYCL/ZE:** peak 1.00x, avg 0.99x

### D2H / fc_to_fc (host=pinned)

| N | Ratio(16) | Ratio(32) | Ratio(64) |
|--:|----------:|----------:|----------:|
| 1 | 0.97x | 1.13x | 0.98x |
| 2 | 0.99x | 1.00x | 1.00x |
| 4 | 0.98x | 0.97x | 0.99x |
| 8 | 0.99x | 0.97x | 0.99x |
| 16 | 0.98x | 0.98x | 1.00x |
| 32 | 0.98x | 0.99x | 1.00x |
| 64 | 0.98x | 0.99x | 1.00x |
| 128 | 1.01x | 1.00x | 1.00x |
| 256 | 1.00x | 1.00x | 1.00x |

**SYCL/ZE:** peak 1.00x, avg 1.00x

### D2H / lw_to_fc (host=pinned)

| N | Ratio(16) | Ratio(32) | Ratio(64) |
|--:|----------:|----------:|----------:|
| 1 | 0.98x | 1.13x | 0.99x |
| 2 | 0.98x | 0.99x | 0.99x |
| 4 | 0.99x | 0.96x | 0.99x |
| 8 | 0.99x | 0.99x | 0.99x |
| 16 | 1.00x | 0.99x | 0.98x |
| 32 | 0.99x | 1.00x | 1.00x |
| 64 | 0.99x | 1.00x | 1.01x |
| 128 | 1.00x | 0.99x | 1.00x |
| 256 | 1.00x | 1.00x | 1.00x |

**SYCL/ZE:** peak 1.00x, avg 1.00x

### D2H / fc_to_lw (host=pinned)

| N | Ratio(16) | Ratio(32) | Ratio(64) |
|--:|----------:|----------:|----------:|
| 1 | 0.98x | 1.14x | 0.99x |
| 2 | 0.98x | 0.99x | 1.00x |
| 4 | 0.99x | 0.97x | 0.98x |
| 8 | 0.98x | 0.98x | 0.98x |
| 16 | 0.99x | 0.99x | 1.01x |
| 32 | 0.99x | 1.00x | 0.99x |
| 64 | 0.99x | 1.00x | 0.99x |
| 128 | 1.00x | 0.99x | 1.00x |
| 256 | 1.00x | 1.00x | 1.00x |

**SYCL/ZE:** peak 0.99x, avg 1.00x
