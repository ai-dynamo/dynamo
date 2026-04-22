# TransferManager — Level Zero vs SYCL Side-by-Side Comparison

> **Run:** `20260419_134055` | **Sources:** `transfer_ze_iter2.log` + `transfer_sycl_iter2.log` | **Iteration:** 2 of 5
> **Hardware:** Intel Arc B580, 160 EUs, 2900 MHz, PCIe Gen4 x8

Ratio = SYCL / ZE (>1.00 means SYCL faster).

## Production-Size Comparison: N=256, tpb=64

| Dir | Pattern | Host | SYCL/ZE |
|-----|---------|------|--------:|
| D2D | fc_to_fc | - | 1.58x |
| D2D | lw_to_fc | - | 1.00x |
| D2D | fc_to_lw | - | 1.00x |
| H2D | fc_to_fc | pinned | 1.00x |
| H2D | lw_to_fc | pinned | 1.00x |
| H2D | fc_to_lw | pinned | 1.00x |
| D2H | fc_to_fc | pinned | 1.00x |
| D2H | lw_to_fc | pinned | 1.00x |
| D2H | fc_to_lw | pinned | 1.00x |

## Full-Sweep Average (All 27 N×tpb Points)

| Dir | Pattern | Host | SYCL/ZE |
|-----|---------|------|--------:|
| D2D | fc_to_fc | - | 1.36x |
| D2D | lw_to_fc | - | 0.99x |
| D2D | fc_to_lw | - | 0.99x |
| H2D | fc_to_fc | pinned | 0.96x |
| H2D | lw_to_fc | pinned | 0.97x |
| H2D | fc_to_lw | pinned | 0.97x |
| D2H | fc_to_fc | pinned | 0.98x |
| D2H | lw_to_fc | pinned | 0.98x |
| D2H | fc_to_lw | pinned | 0.97x |

## Full N × tpb Side-by-Side Grids

Each cell: `ZE | SYCL | ratio`

### D2D / fc_to_fc

| N | Ratio(16) | Ratio(32) | Ratio(64) |
|--:|----------:|----------:|----------:|
| 1 | 1.02x | 1.11x | 0.92x |
| 2 | 0.97x | 0.87x | 0.87x |
| 4 | 1.14x | 0.85x | 0.83x |
| 8 | 1.13x | 0.85x | 0.98x |
| 16 | 1.09x | 0.82x | 1.70x |
| 32 | 1.08x | 1.71x | 1.27x |
| 64 | 1.91x | 1.48x | 1.44x |
| 128 | 1.03x | 1.48x | 1.53x |
| 256 | 1.49x | 1.56x | 1.58x |

**SYCL/ZE:** peak 1.58x, avg 1.36x

### D2D / lw_to_fc

| N | Ratio(16) | Ratio(32) | Ratio(64) |
|--:|----------:|----------:|----------:|
| 1 | 0.97x | 1.00x | 0.90x |
| 2 | 1.00x | 0.89x | 0.89x |
| 4 | 1.15x | 0.90x | 0.85x |
| 8 | 1.12x | 0.87x | 0.98x |
| 16 | 1.12x | 0.86x | 0.86x |
| 32 | 1.10x | 0.87x | 0.85x |
| 64 | 1.11x | 0.99x | 1.12x |
| 128 | 1.04x | 1.18x | 0.99x |
| 256 | 0.92x | 1.00x | 1.00x |

**SYCL/ZE:** peak 0.92x, avg 0.99x

### D2D / fc_to_lw

| N | Ratio(16) | Ratio(32) | Ratio(64) |
|--:|----------:|----------:|----------:|
| 1 | 0.97x | 0.99x | 0.91x |
| 2 | 1.02x | 0.89x | 0.89x |
| 4 | 1.15x | 0.91x | 0.87x |
| 8 | 1.13x | 0.85x | 0.98x |
| 16 | 1.13x | 0.87x | 0.87x |
| 32 | 1.11x | 0.85x | 0.87x |
| 64 | 1.11x | 1.00x | 1.12x |
| 128 | 1.03x | 1.19x | 0.99x |
| 256 | 0.92x | 1.01x | 1.00x |

**SYCL/ZE:** peak 0.92x, avg 0.99x

### H2D / fc_to_fc (host=pinned)

| N | Ratio(16) | Ratio(32) | Ratio(64) |
|--:|----------:|----------:|----------:|
| 1 | 1.00x | 0.76x | 0.86x |
| 2 | 0.98x | 0.89x | 0.88x |
| 4 | 0.97x | 0.87x | 0.89x |
| 8 | 0.99x | 0.90x | 0.98x |
| 16 | 0.98x | 0.99x | 1.00x |
| 32 | 0.98x | 1.00x | 0.99x |
| 64 | 1.02x | 1.01x | 1.00x |
| 128 | 1.00x | 1.00x | 1.00x |
| 256 | 1.00x | 1.00x | 1.00x |

**SYCL/ZE:** peak 1.00x, avg 0.96x

### H2D / lw_to_fc (host=pinned)

| N | Ratio(16) | Ratio(32) | Ratio(64) |
|--:|----------:|----------:|----------:|
| 1 | 0.99x | 0.77x | 1.00x |
| 2 | 1.00x | 0.89x | 0.88x |
| 4 | 0.95x | 0.90x | 0.90x |
| 8 | 1.00x | 0.91x | 0.99x |
| 16 | 1.00x | 1.00x | 0.99x |
| 32 | 1.00x | 1.00x | 1.00x |
| 64 | 1.00x | 1.00x | 1.01x |
| 128 | 1.00x | 1.00x | 1.00x |
| 256 | 1.00x | 1.00x | 1.00x |

**SYCL/ZE:** peak 1.00x, avg 0.97x

### H2D / fc_to_lw (host=pinned)

| N | Ratio(16) | Ratio(32) | Ratio(64) |
|--:|----------:|----------:|----------:|
| 1 | 0.99x | 0.77x | 1.04x |
| 2 | 0.99x | 0.90x | 0.88x |
| 4 | 0.98x | 0.90x | 0.90x |
| 8 | 1.00x | 0.92x | 1.00x |
| 16 | 0.99x | 0.99x | 1.00x |
| 32 | 1.00x | 1.00x | 1.00x |
| 64 | 1.00x | 0.99x | 1.00x |
| 128 | 0.99x | 1.00x | 1.00x |
| 256 | 1.00x | 1.00x | 1.00x |

**SYCL/ZE:** peak 1.00x, avg 0.97x

### D2H / fc_to_fc (host=pinned)

| N | Ratio(16) | Ratio(32) | Ratio(64) |
|--:|----------:|----------:|----------:|
| 1 | 0.98x | 0.76x | 1.01x |
| 2 | 1.00x | 0.90x | 0.89x |
| 4 | 0.98x | 0.89x | 1.06x |
| 8 | 0.97x | 1.08x | 0.93x |
| 16 | 0.98x | 1.00x | 1.01x |
| 32 | 0.99x | 0.99x | 1.00x |
| 64 | 0.98x | 0.99x | 1.01x |
| 128 | 1.01x | 1.00x | 1.00x |
| 256 | 1.00x | 1.00x | 1.00x |

**SYCL/ZE:** peak 1.00x, avg 0.98x

### D2H / lw_to_fc (host=pinned)

| N | Ratio(16) | Ratio(32) | Ratio(64) |
|--:|----------:|----------:|----------:|
| 1 | 0.99x | 0.89x | 0.98x |
| 2 | 0.99x | 0.86x | 0.91x |
| 4 | 1.00x | 0.93x | 0.90x |
| 8 | 0.99x | 0.92x | 0.98x |
| 16 | 0.99x | 0.97x | 0.98x |
| 32 | 0.99x | 0.98x | 1.00x |
| 64 | 1.01x | 1.01x | 1.00x |
| 128 | 0.99x | 1.01x | 1.00x |
| 256 | 1.00x | 1.00x | 1.00x |

**SYCL/ZE:** peak 1.00x, avg 0.98x

### D2H / fc_to_lw (host=pinned)

| N | Ratio(16) | Ratio(32) | Ratio(64) |
|--:|----------:|----------:|----------:|
| 1 | 0.99x | 0.87x | 0.97x |
| 2 | 0.98x | 0.87x | 0.91x |
| 4 | 0.86x | 0.93x | 0.90x |
| 8 | 0.98x | 0.92x | 0.98x |
| 16 | 1.00x | 0.97x | 1.01x |
| 32 | 0.99x | 1.02x | 0.99x |
| 64 | 1.00x | 0.99x | 1.01x |
| 128 | 1.01x | 1.00x | 1.01x |
| 256 | 1.00x | 1.00x | 1.00x |

**SYCL/ZE:** peak 0.98x, avg 0.97x
