# TransferManager — Level Zero vs SYCL Side-by-Side Comparison

> **Run:** `20260418_171130` | **Sources:** `transfer_ze_iter4.log` + `transfer_sycl_iter4.log` | **Iteration:** 4 of 5
> **Hardware:** Intel Arc B580, 160 EUs, 2900 MHz, PCIe Gen4 x8

Compares TransferManager throughput between Level Zero (ZE) and SYCL backends
for all direction × pattern combinations.

Ratio = SYCL / ZE (>1.00 means SYCL faster).

## Production-Size Comparison: N=256, tpb=64

| Dir | Pattern | Host | SYCL/ZE |
|-----|---------|------|--------:|
| D2D | fc_to_fc | - | 1.56x |
| D2D | lw_to_fc | - | 1.01x |
| D2D | fc_to_lw | - | 1.00x |
| H2D | fc_to_fc | pinned | 1.00x |
| H2D | lw_to_fc | pinned | 1.00x |
| H2D | fc_to_lw | pinned | 1.00x |
| D2H | fc_to_fc | pinned | 1.00x |
| D2H | lw_to_fc | pinned | 1.00x |
| D2H | fc_to_lw | pinned | 1.03x |

## Full-Sweep Average (All 27 N×tpb Points)

| Dir | Pattern | Host | SYCL/ZE |
|-----|---------|------|--------:|
| D2D | fc_to_fc | - | 1.39x |
| D2D | lw_to_fc | - | 1.02x |
| D2D | fc_to_lw | - | 1.00x |
| H2D | fc_to_fc | pinned | 0.99x |
| H2D | lw_to_fc | pinned | 0.99x |
| H2D | fc_to_lw | pinned | 1.00x |
| D2H | fc_to_fc | pinned | 0.99x |
| D2H | lw_to_fc | pinned | 1.00x |
| D2H | fc_to_lw | pinned | 1.00x |

## Full N × tpb Side-by-Side Grids

Each cell: `ZE | SYCL | ratio`

### D2D / fc_to_fc

| N | Ratio(16) | Ratio(32) | Ratio(64) |
|--:|----------:|----------:|----------:|
| 1 | 1.00x | 0.99x | 0.98x |
| 2 | 0.97x | 0.98x | 1.16x |
| 4 | 0.99x | 0.95x | 0.82x |
| 8 | 0.83x | 0.94x | 0.96x |
| 16 | 0.90x | 0.94x | 1.95x |
| 32 | 0.95x | 1.98x | 1.41x |
| 64 | 1.92x | 1.47x | 1.57x |
| 128 | 1.12x | 1.62x | 1.53x |
| 256 | 1.51x | 1.44x | 1.56x |

**SYCL/ZE:** peak 1.56x, avg 1.39x

### D2D / lw_to_fc

| N | Ratio(16) | Ratio(32) | Ratio(64) |
|--:|----------:|----------:|----------:|
| 1 | 0.99x | 0.98x | 0.99x |
| 2 | 0.99x | 0.98x | 1.14x |
| 4 | 0.99x | 0.98x | 1.00x |
| 8 | 0.86x | 0.98x | 0.99x |
| 16 | 0.89x | 0.97x | 1.01x |
| 32 | 0.98x | 0.98x | 0.95x |
| 64 | 1.03x | 1.01x | 1.00x |
| 128 | 1.15x | 1.27x | 0.99x |
| 256 | 1.01x | 1.03x | 1.01x |

**SYCL/ZE:** peak 1.01x, avg 1.02x

### D2D / fc_to_lw

| N | Ratio(16) | Ratio(32) | Ratio(64) |
|--:|----------:|----------:|----------:|
| 1 | 0.98x | 0.99x | 0.99x |
| 2 | 0.98x | 0.97x | 1.16x |
| 4 | 0.98x | 0.98x | 1.00x |
| 8 | 0.85x | 0.97x | 0.98x |
| 16 | 0.87x | 0.97x | 0.98x |
| 32 | 0.98x | 0.97x | 0.95x |
| 64 | 1.03x | 1.00x | 1.00x |
| 128 | 1.13x | 0.99x | 0.99x |
| 256 | 1.01x | 1.01x | 1.00x |

**SYCL/ZE:** peak 1.01x, avg 1.00x

### H2D / fc_to_fc (host=pinned)

| N | Ratio(16) | Ratio(32) | Ratio(64) |
|--:|----------:|----------:|----------:|
| 1 | 0.98x | 0.99x | 0.98x |
| 2 | 0.99x | 0.97x | 1.00x |
| 4 | 0.90x | 0.98x | 0.98x |
| 8 | 0.97x | 0.99x | 0.98x |
| 16 | 0.98x | 0.99x | 0.99x |
| 32 | 0.98x | 0.99x | 0.99x |
| 64 | 1.02x | 1.01x | 1.01x |
| 128 | 1.00x | 0.99x | 1.00x |
| 256 | 1.00x | 1.00x | 1.00x |

**SYCL/ZE:** peak 1.00x, avg 0.99x

### H2D / lw_to_fc (host=pinned)

| N | Ratio(16) | Ratio(32) | Ratio(64) |
|--:|----------:|----------:|----------:|
| 1 | 1.00x | 0.98x | 1.00x |
| 2 | 0.99x | 0.97x | 1.01x |
| 4 | 0.90x | 0.99x | 0.98x |
| 8 | 1.07x | 1.00x | 0.99x |
| 16 | 0.98x | 1.00x | 0.99x |
| 32 | 1.00x | 1.00x | 0.99x |
| 64 | 1.00x | 1.00x | 1.00x |
| 128 | 0.99x | 1.00x | 1.00x |
| 256 | 1.00x | 1.00x | 1.00x |

**SYCL/ZE:** peak 1.00x, avg 0.99x

### H2D / fc_to_lw (host=pinned)

| N | Ratio(16) | Ratio(32) | Ratio(64) |
|--:|----------:|----------:|----------:|
| 1 | 0.99x | 0.99x | 0.99x |
| 2 | 1.00x | 0.96x | 1.01x |
| 4 | 0.98x | 1.00x | 0.98x |
| 8 | 1.09x | 1.00x | 0.98x |
| 16 | 0.98x | 0.99x | 0.99x |
| 32 | 1.00x | 1.00x | 0.99x |
| 64 | 1.02x | 1.00x | 1.00x |
| 128 | 1.01x | 1.00x | 0.99x |
| 256 | 1.00x | 1.00x | 1.00x |

**SYCL/ZE:** peak 1.00x, avg 1.00x

### D2H / fc_to_fc (host=pinned)

| N | Ratio(16) | Ratio(32) | Ratio(64) |
|--:|----------:|----------:|----------:|
| 1 | 0.98x | 0.98x | 0.97x |
| 2 | 0.99x | 0.99x | 1.00x |
| 4 | 0.98x | 0.99x | 0.99x |
| 8 | 0.98x | 0.99x | 1.00x |
| 16 | 0.91x | 0.98x | 1.03x |
| 32 | 0.97x | 0.99x | 1.01x |
| 64 | 0.98x | 0.99x | 1.01x |
| 128 | 1.01x | 1.00x | 1.00x |
| 256 | 1.00x | 1.00x | 1.00x |

**SYCL/ZE:** peak 1.00x, avg 0.99x

### D2H / lw_to_fc (host=pinned)

| N | Ratio(16) | Ratio(32) | Ratio(64) |
|--:|----------:|----------:|----------:|
| 1 | 1.00x | 0.98x | 0.98x |
| 2 | 0.98x | 1.00x | 0.99x |
| 4 | 0.99x | 1.00x | 0.99x |
| 8 | 1.05x | 0.98x | 0.99x |
| 16 | 1.07x | 0.99x | 0.99x |
| 32 | 0.99x | 0.99x | 1.00x |
| 64 | 1.00x | 1.00x | 1.00x |
| 128 | 1.00x | 1.00x | 1.00x |
| 256 | 1.00x | 1.00x | 1.00x |

**SYCL/ZE:** peak 1.00x, avg 1.00x

### D2H / fc_to_lw (host=pinned)

| N | Ratio(16) | Ratio(32) | Ratio(64) |
|--:|----------:|----------:|----------:|
| 1 | 0.98x | 0.98x | 0.98x |
| 2 | 1.01x | 1.00x | 0.98x |
| 4 | 0.99x | 0.99x | 0.98x |
| 8 | 1.05x | 0.99x | 0.99x |
| 16 | 1.00x | 0.99x | 1.00x |
| 32 | 0.99x | 0.99x | 1.01x |
| 64 | 1.00x | 1.00x | 0.99x |
| 128 | 0.99x | 1.00x | 0.94x |
| 256 | 0.99x | 1.00x | 1.03x |

**SYCL/ZE:** peak 0.99x, avg 1.00x
