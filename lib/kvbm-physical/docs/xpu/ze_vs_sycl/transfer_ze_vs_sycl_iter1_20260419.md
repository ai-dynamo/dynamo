# TransferManager — Level Zero vs SYCL Side-by-Side Comparison

> **Run:** `20260419_134055` | **Sources:** `transfer_ze_iter1.log` + `transfer_sycl_iter1.log` | **Iteration:** 1 of 5
> **Hardware:** Intel Arc B580, 160 EUs, 2900 MHz, PCIe Gen4 x8

Ratio = SYCL / ZE (>1.00 means SYCL faster).

## Production-Size Comparison: N=256, tpb=64

| Dir | Pattern | Host | SYCL/ZE |
|-----|---------|------|--------:|
| D2D | fc_to_fc | - | 1.57x |
| D2D | lw_to_fc | - | 1.01x |
| D2D | fc_to_lw | - | 0.99x |
| H2D | fc_to_fc | pinned | 1.00x |
| H2D | lw_to_fc | pinned | 1.00x |
| H2D | fc_to_lw | pinned | 1.00x |
| D2H | fc_to_fc | pinned | 1.00x |
| D2H | lw_to_fc | pinned | 1.00x |
| D2H | fc_to_lw | pinned | 1.01x |

## Full-Sweep Average (All 27 N×tpb Points)

| Dir | Pattern | Host | SYCL/ZE |
|-----|---------|------|--------:|
| D2D | fc_to_fc | - | 1.44x |
| D2D | lw_to_fc | - | 1.01x |
| D2D | fc_to_lw | - | 1.00x |
| H2D | fc_to_fc | pinned | 0.99x |
| H2D | lw_to_fc | pinned | 1.00x |
| H2D | fc_to_lw | pinned | 1.00x |
| D2H | fc_to_fc | pinned | 0.99x |
| D2H | lw_to_fc | pinned | 1.00x |
| D2H | fc_to_lw | pinned | 1.00x |

## Full N × tpb Side-by-Side Grids

Each cell: `ZE | SYCL | ratio`

### D2D / fc_to_fc

| N | Ratio(16) | Ratio(32) | Ratio(64) |
|--:|----------:|----------:|----------:|
| 1 | 0.99x | 1.13x | 0.99x |
| 2 | 0.96x | 0.97x | 1.16x |
| 4 | 1.01x | 0.95x | 0.97x |
| 8 | 0.98x | 0.95x | 0.95x |
| 16 | 1.11x | 0.95x | 1.95x |
| 32 | 1.08x | 2.04x | 1.27x |
| 64 | 1.94x | 1.50x | 1.46x |
| 128 | 1.44x | 1.66x | 1.68x |
| 256 | 1.55x | 1.44x | 1.57x |

**SYCL/ZE:** peak 1.58x, avg 1.44x

### D2D / lw_to_fc

| N | Ratio(16) | Ratio(32) | Ratio(64) |
|--:|----------:|----------:|----------:|
| 1 | 0.99x | 1.10x | 1.00x |
| 2 | 1.00x | 0.98x | 1.14x |
| 4 | 1.00x | 0.97x | 1.00x |
| 8 | 0.98x | 0.97x | 0.99x |
| 16 | 1.14x | 0.99x | 1.00x |
| 32 | 1.10x | 1.00x | 0.89x |
| 64 | 1.00x | 1.03x | 1.12x |
| 128 | 1.00x | 1.02x | 0.93x |
| 256 | 1.01x | 1.01x | 1.01x |

**SYCL/ZE:** peak 1.01x, avg 1.01x

### D2D / fc_to_lw

| N | Ratio(16) | Ratio(32) | Ratio(64) |
|--:|----------:|----------:|----------:|
| 1 | 1.00x | 1.09x | 0.99x |
| 2 | 1.01x | 0.99x | 1.15x |
| 4 | 1.00x | 0.96x | 1.14x |
| 8 | 0.99x | 0.97x | 0.99x |
| 16 | 1.14x | 0.99x | 0.99x |
| 32 | 1.10x | 0.98x | 0.89x |
| 64 | 1.00x | 1.01x | 1.11x |
| 128 | 0.99x | 1.01x | 0.93x |
| 256 | 1.02x | 1.00x | 0.99x |

**SYCL/ZE:** peak 1.02x, avg 1.00x

### H2D / fc_to_fc (host=pinned)

| N | Ratio(16) | Ratio(32) | Ratio(64) |
|--:|----------:|----------:|----------:|
| 1 | 0.99x | 1.15x | 0.97x |
| 2 | 0.98x | 0.99x | 0.98x |
| 4 | 0.97x | 0.98x | 0.98x |
| 8 | 0.98x | 0.98x | 0.99x |
| 16 | 0.98x | 0.99x | 0.99x |
| 32 | 0.97x | 0.99x | 0.99x |
| 64 | 1.02x | 0.99x | 1.00x |
| 128 | 1.00x | 1.00x | 1.00x |
| 256 | 0.99x | 1.00x | 1.00x |

**SYCL/ZE:** peak 1.00x, avg 0.99x

### H2D / lw_to_fc (host=pinned)

| N | Ratio(16) | Ratio(32) | Ratio(64) |
|--:|----------:|----------:|----------:|
| 1 | 0.98x | 1.15x | 0.99x |
| 2 | 0.98x | 0.99x | 0.98x |
| 4 | 0.99x | 0.98x | 0.98x |
| 8 | 0.98x | 0.98x | 0.99x |
| 16 | 0.98x | 0.99x | 0.99x |
| 32 | 0.99x | 1.00x | 1.00x |
| 64 | 1.01x | 1.00x | 1.00x |
| 128 | 1.00x | 1.00x | 1.00x |
| 256 | 1.00x | 1.00x | 1.00x |

**SYCL/ZE:** peak 1.00x, avg 1.00x

### H2D / fc_to_lw (host=pinned)

| N | Ratio(16) | Ratio(32) | Ratio(64) |
|--:|----------:|----------:|----------:|
| 1 | 0.98x | 1.12x | 0.99x |
| 2 | 0.98x | 1.00x | 0.97x |
| 4 | 1.00x | 0.98x | 0.98x |
| 8 | 0.98x | 0.98x | 0.99x |
| 16 | 0.98x | 1.00x | 0.99x |
| 32 | 1.00x | 1.00x | 1.00x |
| 64 | 1.01x | 1.00x | 1.00x |
| 128 | 1.00x | 1.00x | 1.00x |
| 256 | 1.00x | 1.00x | 1.00x |

**SYCL/ZE:** peak 1.00x, avg 1.00x

### D2H / fc_to_fc (host=pinned)

| N | Ratio(16) | Ratio(32) | Ratio(64) |
|--:|----------:|----------:|----------:|
| 1 | 0.98x | 1.15x | 0.98x |
| 2 | 0.82x | 0.97x | 0.98x |
| 4 | 0.97x | 0.98x | 0.99x |
| 8 | 0.98x | 0.99x | 1.00x |
| 16 | 0.97x | 0.99x | 1.03x |
| 32 | 0.98x | 0.99x | 1.02x |
| 64 | 0.98x | 0.99x | 1.00x |
| 128 | 1.00x | 1.00x | 1.00x |
| 256 | 1.00x | 1.00x | 1.00x |

**SYCL/ZE:** peak 1.00x, avg 0.99x

### D2H / lw_to_fc (host=pinned)

| N | Ratio(16) | Ratio(32) | Ratio(64) |
|--:|----------:|----------:|----------:|
| 1 | 0.98x | 1.12x | 1.00x |
| 2 | 1.00x | 0.98x | 0.98x |
| 4 | 0.98x | 0.98x | 0.99x |
| 8 | 0.98x | 0.99x | 0.99x |
| 16 | 0.99x | 0.98x | 0.99x |
| 32 | 1.00x | 0.99x | 1.00x |
| 64 | 0.99x | 1.00x | 1.00x |
| 128 | 1.00x | 1.00x | 1.00x |
| 256 | 1.00x | 1.00x | 1.00x |

**SYCL/ZE:** peak 1.00x, avg 1.00x

### D2H / fc_to_lw (host=pinned)

| N | Ratio(16) | Ratio(32) | Ratio(64) |
|--:|----------:|----------:|----------:|
| 1 | 0.98x | 1.12x | 0.99x |
| 2 | 0.99x | 0.99x | 0.98x |
| 4 | 1.00x | 0.98x | 0.99x |
| 8 | 0.98x | 0.99x | 0.99x |
| 16 | 1.00x | 1.00x | 0.99x |
| 32 | 0.99x | 1.00x | 1.00x |
| 64 | 0.99x | 1.00x | 1.00x |
| 128 | 1.00x | 0.99x | 0.99x |
| 256 | 1.01x | 1.00x | 1.01x |

**SYCL/ZE:** peak 0.99x, avg 1.00x
