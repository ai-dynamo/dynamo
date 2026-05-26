# Figure Sources

The benchmark SVGs are rendered from the compiled TSVs in `data/`. The
original benchmark TSVs needed to rebuild them are checked in under
`data/raw_root/results/`.

Regenerate the compiled data from the checked-in raw TSVs with:

```bash
node docs/blogs/dynamo-snapshot/figures/compile_benchmark_data.mjs
```

To rebuild from an external benchmark workspace instead, point `BENCH_DIR` at
the directory that contains `results/`:

```bash
BENCH_DIR=/path/to/benchmark-workspace node docs/blogs/dynamo-snapshot/figures/compile_benchmark_data.mjs
```

Render the benchmark SVGs with:

```bash
node docs/blogs/dynamo-snapshot/figures/render_benchmark_figures.mjs
```

The cold-start benchmark intentionally excludes the base container startup gap
from Kubernetes `container.startedAt` to the first Python/vLLM worker log. The
plotted boundary starts at that first worker log.

| Figure | Source data |
| --- | --- |
| `cold_start_bench.svg` | `data/benchmark_segments.tsv` rows where `figure=cold_start_bench`; compiled from `data/raw_root/results/cold-dgd-full-d6dn5-offlinecache-20260501T002339Z/cold_dgd_vllm_intervals.tsv`. |
| `regular_restore.svg` | `data/benchmark_segments.tsv` rows where `figure=regular_restore`; compiled from `data/raw_root/results/compare-targetready-samenode-s2877-20260430T015100Z-vs-080837Z/regular_restore_gantt.tsv` plus the cold-start rows above. |
| `regular_restore_criudev.svg` | `data/benchmark_segments.tsv` rows where `figure=regular_restore_criudev`; compiled from `data/raw_root/results/regular-slow-criu-blog-tx5tk-20260503T222310Z/regular_restore_gantt.tsv` plus the cold-start rows above. |
| `gms_pvc_restore_bench.svg` | `data/benchmark_segments.tsv` rows where `figure=gms_pvc_restore_bench`; compiled from `data/raw_root/results/gms-defapi-preload2-full-s2877-05061745/gms_restore_gantt.tsv`. |
| `gms_sharded_ssd_restore_bench.svg` | `data/benchmark_segments.tsv` rows where `figure=gms_sharded_ssd_restore_bench`; compiled from `data/raw_root/results/gms-local-ssd-pinned-manual-trigger-api-full-s2877-20260503T173107Z/gms_restore_gantt.tsv`. |
| `checkpoint_restore_state_panels.svg` | Hand-authored schematic SVG; the checked-in SVG is the source. |
| `k8s_checkpoint_restore_lifecycle.svg` | Hand-authored schematic SVG; the checked-in SVG is the source. |
| `worker_agent_quiesce_resume_sequence.svg` | Hand-authored schematic SVG; the checked-in SVG is the source. |
| `kv_cache_unmap_release.svg` | Hand-authored schematic SVG; the checked-in SVG is the source. |
| `preadv_serial_before.svg` | Hand-authored schematic SVG; the checked-in SVG is the source. |
| `aio_pipeline_after.svg` | Hand-authored schematic SVG; the checked-in SVG is the source. |
| `gms_checkpoint_restore_flow.svg` | Hand-authored schematic SVG; the checked-in SVG is the source. |
| `gms_combined_dataflow.svg` | Hand-authored schematic SVG; the checked-in SVG is the source. |
| `hero.svg` | Hand-authored schematic SVG; the checked-in SVG is the source. |
| `scoreboard.svg` | Hand-authored schematic SVG; the checked-in SVG is the source. Numbers come from the bench TSVs above (cold-start TSV + the four restore TSVs). |
| `restore_compare.svg` | Hand-authored schematic SVG; the checked-in SVG is the source. Numbers come from the cold-start TSV and the two `regular_restore_*` TSVs. |

`data/source_files.tsv` is the machine-readable source-file manifest for the
benchmark figures.
