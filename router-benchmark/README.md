# Router benchmark history

This directory stores **benchmark results over time** for the Router Benchmark CI so you can visualize trends.

## Data

- **`benchmark_history.json`** – Time-series of benchmark runs (appended automatically by CI on `main`). Each entry includes run metadata (timestamp, branch, commit, run URL) and the full result payload (TTFT/ITL percentiles per prefix ratio, config).
- **`index.html`** – A small dashboard that plots **TTFT p50** and **ITL p50** vs. time, with one series per prefix ratio.

## How to view the dashboard

1. **GitHub Pages (recommended)**  
   If your repo has GitHub Pages enabled (e.g. “Deploy from branch”, branch `main`, folder `/ (root)` or `docs`), open:
   - `https://<org>.github.io/<repo>/benchmarks/router/history/`  
   The page will load `benchmark_history.json` from the same origin.

2. **Raw JSON from GitHub**  
   The dashboard can also load the JSON from the raw GitHub URL. Open `index.html` from your local clone (e.g. `file:///.../benchmarks/router/history/index.html`) or from the GitHub blob view; the script will try to infer the repo and use `https://raw.githubusercontent.com/<org>/<repo>/main/benchmarks/router/history/benchmark_history.json`.  
   If you open the HTML from a non-GitHub origin, point the script at the raw URL or serve both files from the same directory.

3. **Local**  
   Run a static server from the repo root and open `/benchmarks/router/history/`:
   - `python3 -m http.server 8000` then visit `http://localhost:8000/benchmarks/router/history/`

## How data is recorded

- The [Router Benchmark CI](https://github.com/ai-dynamo/dynamo/blob/main/.github/workflows/router-benchmark-ci.yml) runs on push/PR (and manual) for the benchmark paths.
- A follow-up job **records** each successful run: it appends the result to `benchmark_history.json` and pushes a commit (with `[skip ci]` to avoid loops).
  - **Push to `main`** or **workflow_dispatch**: history is pushed to `main`.
  - **Pull request**: history is pushed to the **PR branch** so you can test the dashboard (e.g. open the HTML from your branch or from GitHub Pages for that branch).
- History is capped at 200 entries (see `.github/scripts/append_benchmark_history.py`).

## Schema (one history entry)

- `run_id`, `timestamp`, `ref`, `branch`, `platform_arch`, `commit`, `run_url`
- `results` – same shape as `router_benchmark_results.json` (from `prefix_ratio_benchmark.py`):
  - `prefix_ratios`, `ttft_p25_values`, `ttft_p50_values`, `ttft_p75_values`, `itl_p25_values`, `itl_p50_values`, `itl_p75_values`, `config`
