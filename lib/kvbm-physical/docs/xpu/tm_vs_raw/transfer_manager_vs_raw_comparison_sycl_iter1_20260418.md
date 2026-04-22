# TransferManager vs Raw Level Zero — Side-by-Side Comparison

> **Run:** `20260418_171130` | **Sources:** `kvbench_sycl_iter1.log` + `transfer_sycl_iter1.log` | **Backend:** SYCL | **Iteration:** 1 of 5

Compares `bench_xpu_transfer` (TransferManager API) against `kvbench_xpu`
(raw Level Zero API) for all 9 direction × pattern combinations.

The kvbench column uses the backend that the TransferManager selects internally:
- fc_to_fc → memcpy (BCS)
- lw_to_fc / fc_to_lw → vectorized kernel (CCS)

Host memory = pinned for H2D/D2H.

## Production-Size Comparison: N=256, tpb=64

At N=256, tpb=64 (1.28 GB total per transfer), fixed per-transfer overhead
is fully amortized:

| Direction | Pattern | TM Engine | TM/Raw |
|-----------|---------|-----------|-------:|
| D2D | fc_to_fc | memcpy | 0.93x |
| D2D | lw_to_fc | vectorized | 0.85x |
| D2D | fc_to_lw | vectorized | 0.87x |
| H2D | fc_to_fc | memcpy | 1.00x |
| H2D | lw_to_fc | vectorized | 1.00x |
| H2D | fc_to_lw | vectorized | 1.00x |
| D2H | fc_to_fc | memcpy | 1.00x |
| D2H | lw_to_fc | vectorized | 1.00x |
| D2H | fc_to_lw | vectorized | 0.99x |

**H2D/D2H: 0.99x–1.00x** — zero measurable TransferManager overhead.
**D2D: 0.85x–0.93x** — residual overhead from TM dispatch vs sub-ms raw transfers.

## Full N × tpb Side-by-Side Grids

Each cell: `TM_value | kv_value | ratio`

### D2D / fc_to_fc (kvbench: memcpy)

| N | Ratio(16) | Ratio(32) | Ratio(64) |
|--:|----------:|----------:|----------:|
| 1 | 0.01x | 0.01x | 0.04x |
| 2 | 0.02x | 0.05x | 0.09x |
| 4 | 0.05x | 0.09x | 0.17x |
| 8 | 0.09x | 0.17x | 0.33x |
| 16 | 0.18x | 0.33x | 0.65x |
| 32 | 0.34x | 0.54x | 0.63x |
| 64 | 0.60x | 0.65x | 0.84x |
| 128 | 0.65x | 0.88x | 0.85x |
| 256 | 0.83x | 0.86x | 0.93x |

**Ratio:** peak 0.60x, avg 0.39x

### D2D / lw_to_fc (kvbench: vectorized)

| N | Ratio(16) | Ratio(32) | Ratio(64) |
|--:|----------:|----------:|----------:|
| 1 | 0.01x | 0.02x | 0.05x |
| 2 | 0.02x | 0.05x | 0.10x |
| 4 | 0.05x | 0.10x | 0.20x |
| 8 | 0.09x | 0.19x | 0.39x |
| 16 | 0.18x | 0.35x | 0.75x |
| 32 | 0.33x | 0.61x | 0.73x |
| 64 | 0.64x | 0.66x | 0.73x |
| 128 | 0.65x | 0.83x | 0.85x |
| 256 | 0.80x | 0.89x | 0.85x |

**Ratio:** peak 0.70x, avg 0.41x

### D2D / fc_to_lw (kvbench: vectorized)

| N | Ratio(16) | Ratio(32) | Ratio(64) |
|--:|----------:|----------:|----------:|
| 1 | 0.01x | 0.02x | 0.05x |
| 2 | 0.02x | 0.05x | 0.10x |
| 4 | 0.05x | 0.10x | 0.19x |
| 8 | 0.09x | 0.19x | 0.38x |
| 16 | 0.18x | 0.35x | 0.75x |
| 32 | 0.35x | 0.61x | 0.74x |
| 64 | 0.65x | 0.68x | 0.74x |
| 128 | 0.65x | 0.66x | 0.88x |
| 256 | 0.81x | 0.90x | 0.87x |

**Ratio:** peak 0.70x, avg 0.41x

### H2D / fc_to_fc (kvbench: memcpy, pinned)

| N | Ratio(16) | Ratio(32) | Ratio(64) |
|--:|----------:|----------:|----------:|
| 1 | 0.36x | 0.55x | 0.64x |
| 2 | 0.71x | 0.64x | 0.90x |
| 4 | 0.65x | 0.89x | 0.93x |
| 8 | 0.89x | 0.93x | 0.94x |
| 16 | 0.91x | 0.94x | 0.96x |
| 32 | 0.93x | 0.96x | 0.98x |
| 64 | 0.98x | 0.98x | 1.00x |
| 128 | 0.99x | 0.99x | 1.01x |
| 256 | 1.00x | 1.00x | 1.00x |

**Ratio:** peak 1.00x, avg 0.88x

### H2D / lw_to_fc (kvbench: vectorized, pinned)

| N | Ratio(16) | Ratio(32) | Ratio(64) |
|--:|----------:|----------:|----------:|
| 1 | 0.34x | 0.53x | 0.62x |
| 2 | 0.67x | 0.63x | 0.86x |
| 4 | 0.60x | 0.86x | 0.90x |
| 8 | 0.86x | 0.91x | 0.94x |
| 16 | 0.89x | 0.95x | 0.96x |
| 32 | 0.93x | 0.96x | 0.98x |
| 64 | 0.96x | 0.99x | 0.99x |
| 128 | 0.97x | 0.99x | 1.00x |
| 256 | 0.98x | 0.99x | 1.00x |

**Ratio:** peak 1.00x, avg 0.86x

### H2D / fc_to_lw (kvbench: vectorized, pinned)

| N | Ratio(16) | Ratio(32) | Ratio(64) |
|--:|----------:|----------:|----------:|
| 1 | 0.34x | 0.53x | 0.62x |
| 2 | 0.67x | 0.63x | 0.78x |
| 4 | 0.60x | 0.86x | 0.90x |
| 8 | 0.86x | 0.91x | 0.95x |
| 16 | 0.89x | 0.94x | 0.95x |
| 32 | 0.93x | 0.95x | 0.98x |
| 64 | 0.95x | 0.98x | 0.99x |
| 128 | 0.97x | 0.99x | 1.00x |
| 256 | 0.99x | 1.00x | 1.00x |

**Ratio:** peak 1.00x, avg 0.86x

### D2H / fc_to_fc (kvbench: memcpy, pinned)

| N | Ratio(16) | Ratio(32) | Ratio(64) |
|--:|----------:|----------:|----------:|
| 1 | 0.33x | 0.51x | 0.62x |
| 2 | 0.66x | 0.62x | 0.76x |
| 4 | 0.61x | 0.83x | 0.87x |
| 8 | 0.82x | 0.87x | 0.96x |
| 16 | 0.86x | 0.97x | 0.98x |
| 32 | 0.95x | 0.98x | 0.99x |
| 64 | 0.97x | 0.99x | 1.00x |
| 128 | 1.00x | 0.99x | 1.00x |
| 256 | 0.99x | 1.00x | 1.00x |

**Ratio:** peak 1.00x, avg 0.86x

### D2H / lw_to_fc (kvbench: vectorized, pinned)

| N | Ratio(16) | Ratio(32) | Ratio(64) |
|--:|----------:|----------:|----------:|
| 1 | 0.40x | 0.63x | 0.74x |
| 2 | 0.79x | 0.73x | 0.74x |
| 4 | 0.74x | 0.79x | 0.93x |
| 8 | 0.78x | 0.94x | 0.96x |
| 16 | 0.93x | 0.97x | 0.98x |
| 32 | 0.96x | 0.98x | 0.99x |
| 64 | 0.97x | 0.98x | 1.00x |
| 128 | 0.98x | 0.99x | 1.00x |
| 256 | 0.99x | 0.99x | 1.00x |

**Ratio:** peak 1.00x, avg 0.88x

### D2H / fc_to_lw (kvbench: vectorized, pinned)

| N | Ratio(16) | Ratio(32) | Ratio(64) |
|--:|----------:|----------:|----------:|
| 1 | 0.39x | 0.63x | 0.74x |
| 2 | 0.79x | 0.74x | 0.74x |
| 4 | 0.75x | 0.79x | 0.93x |
| 8 | 0.78x | 0.94x | 0.97x |
| 16 | 0.93x | 0.96x | 0.98x |
| 32 | 0.96x | 0.96x | 0.99x |
| 64 | 0.94x | 1.01x | 0.98x |
| 128 | 0.98x | 0.98x | 1.01x |
| 256 | 0.92x | 0.99x | 0.99x |

**Ratio:** peak 0.96x, avg 0.88x
