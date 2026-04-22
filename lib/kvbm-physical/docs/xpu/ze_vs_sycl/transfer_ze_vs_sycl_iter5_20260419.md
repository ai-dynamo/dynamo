# TransferManager — Level Zero vs SYCL Side-by-Side Comparison

> **Run:** `20260419_134055` | **Sources:** `transfer_ze_iter5.log` + `transfer_sycl_iter5.log` | **Iteration:** 5 of 5
> **Hardware:** Intel Arc B580, 160 EUs, 2900 MHz, PCIe Gen4 x8

Ratio = SYCL / ZE (>1.00 means SYCL faster).

## Production-Size Comparison: N=256, tpb=64

| Dir | Pattern | Host | SYCL/ZE |
|-----|---------|------|--------:|
| D2D | fc_to_fc | - | 1.56x |
| D2D | lw_to_fc | - | 1.01x |
| D2D | fc_to_lw | - | 1.02x |
| H2D | fc_to_fc | pinned | 0.99x |
| H2D | lw_to_fc | pinned | 1.00x |
| H2D | fc_to_lw | pinned | 1.00x |
| D2H | fc_to_fc | pinned | 1.00x |
| D2H | lw_to_fc | pinned | 1.00x |
| D2H | fc_to_lw | pinned | 1.00x |

## Full-Sweep Average (All 27 N×tpb Points)

| Dir | Pattern | Host | SYCL/ZE |
|-----|---------|------|--------:|
| D2D | fc_to_fc | - | 1.38x |
| D2D | lw_to_fc | - | 1.02x |
| D2D | fc_to_lw | - | 1.01x |
| H2D | fc_to_fc | pinned | 0.99x |
| H2D | lw_to_fc | pinned | 0.99x |
| H2D | fc_to_lw | pinned | 0.99x |
| D2H | fc_to_fc | pinned | 1.00x |
| D2H | lw_to_fc | pinned | 0.99x |
| D2H | fc_to_lw | pinned | 0.99x |

## Full N × tpb Side-by-Side Grids

Each cell: `ZE | SYCL | ratio`

### D2D / fc_to_fc

| N | Ratio(16) | Ratio(32) | Ratio(64) |
|--:|----------:|----------:|----------:|
| 1 | 0.88x | 0.85x | 1.24x |
| 2 | 0.86x | 0.84x | 1.27x |
| 4 | 0.97x | 0.83x | 1.04x |
| 8 | 0.94x | 1.07x | 1.09x |
| 16 | 0.94x | 1.06x | 1.26x |
| 32 | 0.97x | 1.21x | 1.15x |
| 64 | 1.95x | 1.46x | 1.67x |
| 128 | 1.42x | 1.74x | 1.50x |
| 256 | 1.54x | 1.55x | 1.56x |

**SYCL/ZE:** peak 1.56x, avg 1.38x

### D2D / lw_to_fc

| N | Ratio(16) | Ratio(32) | Ratio(64) |
|--:|----------:|----------:|----------:|
| 1 | 0.88x | 0.87x | 1.22x |
| 2 | 0.87x | 0.86x | 1.17x |
| 4 | 0.96x | 0.85x | 1.11x |
| 8 | 0.99x | 1.10x | 1.11x |
| 16 | 0.99x | 1.07x | 1.10x |
| 32 | 1.01x | 1.12x | 1.11x |
| 64 | 1.00x | 1.01x | 1.03x |
| 128 | 0.99x | 0.84x | 1.07x |
| 256 | 0.98x | 1.05x | 1.01x |

**SYCL/ZE:** peak 0.99x, avg 1.02x

### D2D / fc_to_lw

| N | Ratio(16) | Ratio(32) | Ratio(64) |
|--:|----------:|----------:|----------:|
| 1 | 0.89x | 0.88x | 1.20x |
| 2 | 0.89x | 0.86x | 1.19x |
| 4 | 0.97x | 0.85x | 1.11x |
| 8 | 0.98x | 1.11x | 1.10x |
| 16 | 0.99x | 1.08x | 1.10x |
| 32 | 1.03x | 1.13x | 1.16x |
| 64 | 0.98x | 1.00x | 1.04x |
| 128 | 0.99x | 0.81x | 1.07x |
| 256 | 0.99x | 0.91x | 1.02x |

**SYCL/ZE:** peak 0.99x, avg 1.01x

### H2D / fc_to_fc (host=pinned)

| N | Ratio(16) | Ratio(32) | Ratio(64) |
|--:|----------:|----------:|----------:|
| 1 | 0.98x | 0.84x | 1.06x |
| 2 | 0.84x | 0.94x | 1.08x |
| 4 | 0.95x | 0.97x | 1.07x |
| 8 | 0.97x | 0.98x | 0.98x |
| 16 | 0.98x | 0.98x | 0.99x |
| 32 | 0.97x | 1.03x | 1.00x |
| 64 | 1.01x | 1.02x | 1.00x |
| 128 | 1.00x | 1.00x | 1.00x |
| 256 | 1.00x | 1.00x | 0.99x |

**SYCL/ZE:** peak 1.00x, avg 0.99x

### H2D / lw_to_fc (host=pinned)

| N | Ratio(16) | Ratio(32) | Ratio(64) |
|--:|----------:|----------:|----------:|
| 1 | 0.98x | 0.84x | 1.04x |
| 2 | 0.85x | 0.97x | 1.06x |
| 4 | 0.98x | 0.98x | 1.05x |
| 8 | 0.96x | 0.99x | 1.00x |
| 16 | 0.99x | 1.00x | 0.99x |
| 32 | 0.99x | 0.99x | 0.99x |
| 64 | 1.01x | 0.99x | 1.00x |
| 128 | 1.00x | 1.00x | 1.00x |
| 256 | 1.01x | 1.00x | 1.00x |

**SYCL/ZE:** peak 1.00x, avg 0.99x

### H2D / fc_to_lw (host=pinned)

| N | Ratio(16) | Ratio(32) | Ratio(64) |
|--:|----------:|----------:|----------:|
| 1 | 0.85x | 0.84x | 1.04x |
| 2 | 0.85x | 0.97x | 1.06x |
| 4 | 0.98x | 0.98x | 1.06x |
| 8 | 0.98x | 0.99x | 1.00x |
| 16 | 1.00x | 0.99x | 0.99x |
| 32 | 1.00x | 0.99x | 0.99x |
| 64 | 1.04x | 1.00x | 1.00x |
| 128 | 1.00x | 1.00x | 1.00x |
| 256 | 1.00x | 1.00x | 1.00x |

**SYCL/ZE:** peak 1.00x, avg 0.99x

### D2H / fc_to_fc (host=pinned)

| N | Ratio(16) | Ratio(32) | Ratio(64) |
|--:|----------:|----------:|----------:|
| 1 | 0.85x | 0.85x | 1.06x |
| 2 | 0.84x | 0.94x | 1.08x |
| 4 | 0.95x | 0.97x | 1.05x |
| 8 | 0.98x | 0.98x | 1.08x |
| 16 | 0.98x | 1.06x | 1.03x |
| 32 | 0.97x | 1.03x | 1.01x |
| 64 | 0.98x | 1.00x | 1.00x |
| 128 | 1.00x | 1.00x | 1.00x |
| 256 | 1.00x | 1.00x | 1.00x |

**SYCL/ZE:** peak 1.00x, avg 1.00x

### D2H / lw_to_fc (host=pinned)

| N | Ratio(16) | Ratio(32) | Ratio(64) |
|--:|----------:|----------:|----------:|
| 1 | 0.84x | 0.85x | 1.04x |
| 2 | 0.85x | 0.97x | 1.07x |
| 4 | 0.98x | 0.99x | 1.08x |
| 8 | 0.98x | 0.99x | 1.00x |
| 16 | 0.99x | 1.00x | 1.00x |
| 32 | 0.99x | 1.01x | 1.00x |
| 64 | 1.01x | 1.00x | 1.00x |
| 128 | 1.00x | 0.99x | 1.00x |
| 256 | 1.01x | 1.00x | 1.00x |

**SYCL/ZE:** peak 1.00x, avg 0.99x

### D2H / fc_to_lw (host=pinned)

| N | Ratio(16) | Ratio(32) | Ratio(64) |
|--:|----------:|----------:|----------:|
| 1 | 0.84x | 0.85x | 1.07x |
| 2 | 0.85x | 0.97x | 1.07x |
| 4 | 0.98x | 0.99x | 1.07x |
| 8 | 0.99x | 1.00x | 1.00x |
| 16 | 1.00x | 1.01x | 0.97x |
| 32 | 0.98x | 0.98x | 1.00x |
| 64 | 1.00x | 1.00x | 0.98x |
| 128 | 1.00x | 1.00x | 0.99x |
| 256 | 0.99x | 0.99x | 1.00x |

**SYCL/ZE:** peak 0.99x, avg 0.99x
