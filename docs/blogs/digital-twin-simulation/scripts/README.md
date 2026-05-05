# Reproducing the Mocker / AIC / Hardware Plots

This folder has the data and script behind the fidelity plots in §2.1 of the
blog post.

## Files

- `data.csv` — long-format table: one row per `(source, concurrency)`, with
  `tps_gpu`, `tps_user`, `tpot_ms`, `ttft_ms`. Sources are `hardware`,
  `mocker`, `aic`.
- `plot.py` — reads `data.csv`, writes `hw_mocker_aic_pareto.png` and
  `hw_mocker_aic_4panel.png` to `../images/`, prints the MAPE table.

## Running

```bash
pip install matplotlib
python plot.py
```

Optional flags:

```bash
python plot.py --keep-c4              # include c=4 (excluded by default — see below)
python plot.py --out-dir ./somewhere  # write PNGs elsewhere
```

## Setup

- Model: MiniMax-M2.5 FP8
- Hardware: NVIDIA B200 SXM, TP=4
- Workload: ISL = OSL = 1024
- Concurrencies: 4, 8, 16, 32, 64
- Hardware engine: vLLM 0.17
- AIC + mocker (mocker uses AIC's perf model): vLLM v0.14

## Why c=4 is excluded by default

The c=4 hardware TTFT measurement (~121 ms) is well above the trend line set
by c={8, 16, 32, 64} and looks like a cold-start / first-batch artifact. The
default excludes it from both plotting and MAPE so the comparison reflects
steady-state behavior. `--keep-c4` puts it back if you want to inspect it.
