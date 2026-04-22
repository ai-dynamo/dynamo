# TransferManager — Level Zero vs SYCL Side-by-Side Comparison

> **Run:** `20260418_171130` | **Sources:** `transfer_ze_iter5.log` + `transfer_sycl_iter5.log` | **Iteration:** 5 of 5
> **Hardware:** Intel Arc B580, 160 EUs, 2900 MHz, PCIe Gen4 x8

Compares TransferManager throughput between Level Zero (ZE) and SYCL backends
for all direction × pattern combinations.

Ratio = SYCL / ZE (>1.00 means SYCL faster).

## Production-Size Comparison: N=256, tpb=64

| Dir | Pattern | Host | SYCL/ZE |
|-----|---------|------|--------:|
| D2D | fc_to_fc | - | 1.58x |
| D2D | lw_to_fc | - | 1.01x |
| D2D | fc_to_lw | - | 1.01x |
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
| D2D | lw_to_fc | - | 1.04x |
| D2D | fc_to_lw | - | 1.01x |
| H2D | fc_to_fc | pinned | 0.98x |
| H2D | lw_to_fc | pinned | 0.98x |
| H2D | fc_to_lw | pinned | 0.99x |
| D2H | fc_to_fc | pinned | 0.99x |
| D2H | lw_to_fc | pinned | 0.99x |
| D2H | fc_to_lw | pinned | 0.99x |

## Full N × tpb Side-by-Side Grids

Each cell: `ZE | SYCL | ratio`

### D2D / fc_to_fc

| N | Ratio(16) | Ratio(32) | Ratio(64) |
|--:|----------:|----------:|----------:|
| 1 | 0.87x | 0.89x | 1.15x |
| 2 | 0.86x | 0.86x | 0.98x |
| 4 | 0.97x | 0.83x | 0.96x |
| 8 | 0.95x | 0.93x | 1.04x |
| 16 | 0.93x | 0.95x | 1.25x |
| 32 | 1.08x | 2.02x | 1.15x |
| 64 | 1.95x | 1.20x | 1.72x |
| 128 | 1.41x | 1.49x | 1.50x |
| 256 | 1.65x | 1.44x | 1.58x |

**SYCL/ZE:** peak 1.58x, avg 1.37x

### D2D / lw_to_fc

| N | Ratio(16) | Ratio(32) | Ratio(64) |
|--:|----------:|----------:|----------:|
| 1 | 0.90x | 0.88x | 1.14x |
| 2 | 0.89x | 0.84x | 0.98x |
| 4 | 0.98x | 0.84x | 0.95x |
| 8 | 0.97x | 0.96x | 0.96x |
| 16 | 0.99x | 0.98x | 1.19x |
| 32 | 1.14x | 0.99x | 1.16x |
| 64 | 1.00x | 1.19x | 0.91x |
| 128 | 1.03x | 1.05x | 1.01x |
| 256 | 1.07x | 1.14x | 1.01x |

**SYCL/ZE:** peak 1.08x, avg 1.04x

### D2D / fc_to_lw

| N | Ratio(16) | Ratio(32) | Ratio(64) |
|--:|----------:|----------:|----------:|
| 1 | 0.90x | 0.89x | 1.15x |
| 2 | 0.90x | 0.84x | 0.98x |
| 4 | 0.98x | 0.85x | 0.96x |
| 8 | 0.96x | 0.95x | 0.95x |
| 16 | 0.98x | 0.98x | 1.11x |
| 32 | 1.12x | 0.98x | 1.17x |
| 64 | 1.00x | 1.17x | 1.07x |
| 128 | 1.03x | 0.81x | 1.01x |
| 256 | 1.01x | 1.00x | 1.01x |

**SYCL/ZE:** peak 1.01x, avg 1.01x

### H2D / fc_to_fc (host=pinned)

| N | Ratio(16) | Ratio(32) | Ratio(64) |
|--:|----------:|----------:|----------:|
| 1 | 0.98x | 0.85x | 0.97x |
| 2 | 0.84x | 0.95x | 1.00x |
| 4 | 0.97x | 0.97x | 0.98x |
| 8 | 0.98x | 0.99x | 0.99x |
| 16 | 0.98x | 0.99x | 0.99x |
| 32 | 0.98x | 1.03x | 0.99x |
| 64 | 1.02x | 0.99x | 1.00x |
| 128 | 1.00x | 0.99x | 1.00x |
| 256 | 1.00x | 1.00x | 1.00x |

**SYCL/ZE:** peak 1.00x, avg 0.98x

### H2D / lw_to_fc (host=pinned)

| N | Ratio(16) | Ratio(32) | Ratio(64) |
|--:|----------:|----------:|----------:|
| 1 | 1.01x | 0.83x | 0.99x |
| 2 | 0.84x | 0.98x | 1.00x |
| 4 | 1.00x | 0.97x | 0.99x |
| 8 | 0.97x | 0.99x | 0.99x |
| 16 | 0.99x | 1.00x | 1.00x |
| 32 | 1.00x | 0.99x | 0.99x |
| 64 | 1.01x | 1.00x | 0.99x |
| 128 | 1.00x | 1.00x | 1.00x |
| 256 | 0.99x | 1.00x | 1.00x |

**SYCL/ZE:** peak 1.00x, avg 0.98x

### H2D / fc_to_lw (host=pinned)

| N | Ratio(16) | Ratio(32) | Ratio(64) |
|--:|----------:|----------:|----------:|
| 1 | 0.99x | 0.84x | 0.95x |
| 2 | 0.85x | 0.98x | 0.99x |
| 4 | 1.01x | 0.98x | 0.99x |
| 8 | 0.99x | 0.99x | 1.00x |
| 16 | 0.99x | 1.00x | 0.99x |
| 32 | 1.00x | 1.00x | 0.99x |
| 64 | 1.04x | 1.00x | 1.00x |
| 128 | 1.00x | 0.99x | 1.00x |
| 256 | 1.00x | 1.00x | 1.00x |

**SYCL/ZE:** peak 1.00x, avg 0.99x

### D2H / fc_to_fc (host=pinned)

| N | Ratio(16) | Ratio(32) | Ratio(64) |
|--:|----------:|----------:|----------:|
| 1 | 1.00x | 0.84x | 0.96x |
| 2 | 0.83x | 0.96x | 1.00x |
| 4 | 0.99x | 0.97x | 1.06x |
| 8 | 0.98x | 0.99x | 0.99x |
| 16 | 0.98x | 1.06x | 1.04x |
| 32 | 0.99x | 0.98x | 1.01x |
| 64 | 1.01x | 0.99x | 1.01x |
| 128 | 1.01x | 1.00x | 1.00x |
| 256 | 1.00x | 1.00x | 1.00x |

**SYCL/ZE:** peak 1.00x, avg 0.99x

### D2H / lw_to_fc (host=pinned)

| N | Ratio(16) | Ratio(32) | Ratio(64) |
|--:|----------:|----------:|----------:|
| 1 | 0.99x | 0.84x | 0.99x |
| 2 | 0.84x | 0.98x | 1.00x |
| 4 | 1.00x | 0.99x | 1.07x |
| 8 | 0.98x | 1.00x | 0.99x |
| 16 | 0.99x | 1.01x | 1.00x |
| 32 | 1.01x | 1.00x | 1.01x |
| 64 | 1.00x | 1.01x | 1.00x |
| 128 | 1.00x | 1.00x | 1.00x |
| 256 | 1.00x | 1.00x | 1.00x |

**SYCL/ZE:** peak 1.00x, avg 0.99x

### D2H / fc_to_lw (host=pinned)

| N | Ratio(16) | Ratio(32) | Ratio(64) |
|--:|----------:|----------:|----------:|
| 1 | 0.99x | 0.84x | 0.99x |
| 2 | 0.85x | 1.07x | 1.00x |
| 4 | 0.96x | 0.99x | 1.08x |
| 8 | 0.99x | 0.99x | 1.00x |
| 16 | 0.99x | 1.00x | 1.01x |
| 32 | 1.00x | 1.03x | 1.01x |
| 64 | 1.00x | 1.02x | 0.99x |
| 128 | 1.01x | 1.01x | 1.01x |
| 256 | 1.00x | 0.99x | 1.00x |

**SYCL/ZE:** peak 1.00x, avg 0.99x
