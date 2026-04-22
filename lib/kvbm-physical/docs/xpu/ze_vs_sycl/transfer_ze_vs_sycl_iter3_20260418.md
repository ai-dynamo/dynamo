# TransferManager — Level Zero vs SYCL Side-by-Side Comparison

> **Run:** `20260418_171130` | **Sources:** `transfer_ze_iter3.log` + `transfer_sycl_iter3.log` | **Iteration:** 3 of 5
> **Hardware:** Intel Arc B580, 160 EUs, 2900 MHz, PCIe Gen4 x8

Compares TransferManager throughput between Level Zero (ZE) and SYCL backends
for all direction × pattern combinations.

Ratio = SYCL / ZE (>1.00 means SYCL faster).

## Production-Size Comparison: N=256, tpb=64

| Dir | Pattern | Host | SYCL/ZE |
|-----|---------|------|--------:|
| D2D | fc_to_fc | - | 1.56x |
| D2D | lw_to_fc | - | 1.00x |
| D2D | fc_to_lw | - | 1.00x |
| H2D | fc_to_fc | pinned | 1.00x |
| H2D | lw_to_fc | pinned | 1.00x |
| H2D | fc_to_lw | pinned | 1.00x |
| D2H | fc_to_fc | pinned | 1.00x |
| D2H | lw_to_fc | pinned | 1.00x |
| D2H | fc_to_lw | pinned | 1.05x |

## Full-Sweep Average (All 27 N×tpb Points)

| Dir | Pattern | Host | SYCL/ZE |
|-----|---------|------|--------:|
| D2D | fc_to_fc | - | 1.47x |
| D2D | lw_to_fc | - | 1.06x |
| D2D | fc_to_lw | - | 1.05x |
| H2D | fc_to_fc | pinned | 0.99x |
| H2D | lw_to_fc | pinned | 0.99x |
| H2D | fc_to_lw | pinned | 1.00x |
| D2H | fc_to_fc | pinned | 1.00x |
| D2H | lw_to_fc | pinned | 1.01x |
| D2H | fc_to_lw | pinned | 1.00x |

## Full N × tpb Side-by-Side Grids

Each cell: `ZE | SYCL | ratio`

### D2D / fc_to_fc

| N | Ratio(16) | Ratio(32) | Ratio(64) |
|--:|----------:|----------:|----------:|
| 1 | 1.01x | 1.01x | 1.00x |
| 2 | 1.02x | 0.99x | 1.16x |
| 4 | 1.14x | 1.13x | 1.14x |
| 8 | 1.12x | 1.14x | 1.14x |
| 16 | 0.96x | 1.09x | 1.96x |
| 32 | 0.96x | 1.96x | 1.41x |
| 64 | 1.97x | 1.46x | 1.69x |
| 128 | 1.40x | 1.54x | 1.48x |
| 256 | 1.64x | 1.45x | 1.56x |

**SYCL/ZE:** peak 1.56x, avg 1.47x

### D2D / lw_to_fc

| N | Ratio(16) | Ratio(32) | Ratio(64) |
|--:|----------:|----------:|----------:|
| 1 | 0.98x | 1.01x | 0.99x |
| 2 | 1.00x | 0.99x | 1.15x |
| 4 | 1.12x | 1.14x | 1.14x |
| 8 | 1.12x | 1.12x | 1.15x |
| 16 | 0.99x | 1.14x | 1.01x |
| 32 | 0.99x | 1.14x | 1.00x |
| 64 | 1.02x | 1.02x | 1.00x |
| 128 | 1.00x | 1.23x | 1.06x |
| 256 | 1.08x | 1.04x | 1.00x |

**SYCL/ZE:** peak 1.08x, avg 1.06x

### D2D / fc_to_lw

| N | Ratio(16) | Ratio(32) | Ratio(64) |
|--:|----------:|----------:|----------:|
| 1 | 0.99x | 1.00x | 0.99x |
| 2 | 1.01x | 0.99x | 1.15x |
| 4 | 1.13x | 1.14x | 1.13x |
| 8 | 1.12x | 1.12x | 1.13x |
| 16 | 0.99x | 1.12x | 1.01x |
| 32 | 0.99x | 1.11x | 0.99x |
| 64 | 1.01x | 1.00x | 0.99x |
| 128 | 1.00x | 1.24x | 1.06x |
| 256 | 1.05x | 1.03x | 1.00x |

**SYCL/ZE:** peak 1.05x, avg 1.05x

### H2D / fc_to_fc (host=pinned)

| N | Ratio(16) | Ratio(32) | Ratio(64) |
|--:|----------:|----------:|----------:|
| 1 | 0.98x | 0.84x | 0.98x |
| 2 | 0.97x | 0.97x | 0.98x |
| 4 | 0.97x | 0.98x | 0.99x |
| 8 | 0.99x | 0.98x | 0.98x |
| 16 | 0.98x | 0.99x | 0.99x |
| 32 | 0.98x | 1.03x | 0.99x |
| 64 | 1.02x | 1.01x | 1.01x |
| 128 | 1.00x | 0.99x | 1.00x |
| 256 | 1.00x | 1.00x | 1.00x |

**SYCL/ZE:** peak 1.00x, avg 0.99x

### H2D / lw_to_fc (host=pinned)

| N | Ratio(16) | Ratio(32) | Ratio(64) |
|--:|----------:|----------:|----------:|
| 1 | 0.98x | 0.84x | 0.99x |
| 2 | 0.98x | 1.00x | 0.98x |
| 4 | 1.08x | 0.98x | 0.99x |
| 8 | 1.00x | 0.99x | 1.00x |
| 16 | 1.00x | 0.99x | 1.00x |
| 32 | 1.00x | 0.99x | 0.99x |
| 64 | 1.00x | 1.00x | 1.00x |
| 128 | 1.00x | 1.00x | 1.00x |
| 256 | 0.99x | 0.99x | 1.00x |

**SYCL/ZE:** peak 1.00x, avg 0.99x

### H2D / fc_to_lw (host=pinned)

| N | Ratio(16) | Ratio(32) | Ratio(64) |
|--:|----------:|----------:|----------:|
| 1 | 0.99x | 0.99x | 0.99x |
| 2 | 0.99x | 0.99x | 0.98x |
| 4 | 1.08x | 0.98x | 0.99x |
| 8 | 1.00x | 1.07x | 1.00x |
| 16 | 0.99x | 0.99x | 0.99x |
| 32 | 1.00x | 1.01x | 1.00x |
| 64 | 1.00x | 1.00x | 1.00x |
| 128 | 1.00x | 1.00x | 1.00x |
| 256 | 0.99x | 1.00x | 1.00x |

**SYCL/ZE:** peak 1.00x, avg 1.00x

### D2H / fc_to_fc (host=pinned)

| N | Ratio(16) | Ratio(32) | Ratio(64) |
|--:|----------:|----------:|----------:|
| 1 | 0.99x | 0.97x | 0.98x |
| 2 | 0.98x | 0.98x | 0.98x |
| 4 | 1.09x | 0.98x | 0.98x |
| 8 | 0.99x | 0.93x | 1.08x |
| 16 | 0.99x | 0.99x | 1.00x |
| 32 | 0.97x | 0.99x | 1.02x |
| 64 | 0.98x | 0.99x | 1.00x |
| 128 | 1.01x | 0.99x | 1.00x |
| 256 | 1.00x | 1.00x | 1.00x |

**SYCL/ZE:** peak 1.00x, avg 1.00x

### D2H / lw_to_fc (host=pinned)

| N | Ratio(16) | Ratio(32) | Ratio(64) |
|--:|----------:|----------:|----------:|
| 1 | 0.98x | 0.98x | 0.98x |
| 2 | 0.99x | 0.99x | 1.06x |
| 4 | 1.07x | 0.98x | 1.06x |
| 8 | 1.00x | 1.07x | 1.00x |
| 16 | 1.01x | 1.00x | 0.99x |
| 32 | 1.00x | 0.99x | 1.00x |
| 64 | 1.00x | 1.00x | 1.00x |
| 128 | 0.99x | 1.00x | 1.00x |
| 256 | 1.00x | 1.00x | 1.00x |

**SYCL/ZE:** peak 1.00x, avg 1.01x

### D2H / fc_to_lw (host=pinned)

| N | Ratio(16) | Ratio(32) | Ratio(64) |
|--:|----------:|----------:|----------:|
| 1 | 0.98x | 0.98x | 0.99x |
| 2 | 0.99x | 0.99x | 1.05x |
| 4 | 1.09x | 0.98x | 0.99x |
| 8 | 0.99x | 1.07x | 1.00x |
| 16 | 1.00x | 0.99x | 1.00x |
| 32 | 1.00x | 1.00x | 1.00x |
| 64 | 1.00x | 0.98x | 0.99x |
| 128 | 0.98x | 0.99x | 1.01x |
| 256 | 0.98x | 1.04x | 1.05x |

**SYCL/ZE:** peak 1.00x, avg 1.00x
