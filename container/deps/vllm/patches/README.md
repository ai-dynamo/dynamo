# vLLM source patches

Patches applied directly against the installed `vllm` package
(`${SITE_PACKAGES}/vllm`) inside the vllm runtime image, on top of the
upstream `vllm/vllm-openai` (or `vllm/vllm-openai-cpu`) wheel pinned by
`vllm.<device>.runtime_image_tag` in `container/context.yaml`. Applied by the
`{% if device != "xpu" %}` block in `container/templates/vllm_runtime.Dockerfile`
via `patch -p1`. Not applied on xpu, which tracks a separate vLLM fork/ref.

Each patch is a plain `git diff` taken from a local `ai-dynamo`-internal
vLLM checkout pinned to the same tag as `runtime_image_tag`. To regenerate
or re-validate one:

```bash
cd <local vllm checkout, at the tag in context.yaml>
git diff > container/deps/vllm/patches/000N-<name>.patch
git apply --check -p1 container/deps/vllm/patches/000N-<name>.patch  # sanity check
```

Whenever `runtime_image_tag` is bumped, re-run `git apply --check -p1` for
every patch here against the new vLLM tag before merging the bump. Drop or
rebase any patch that conflicts, and check whether upstream has since merged
the equivalent change (in which case delete the patch).

## Patches

- `0001-offload-g2-observability-metrics.patch` — adds vLLM offload (G2/CPU
  KV-cache) observability metrics used for the host-offload performance
  investigation in `dynamo/disagg/offloading/offload_g2_probe_recap.md`:
  - `CPUOffloadingManager.lookup()` outcome counters: `vllm:kv_offload_lookup_queries`,
    `..._hits`, `..._misses`, `..._not_ready`.
  - Scheduler-side store pin-duration histogram
    `vllm:kv_offload_store_pin_duration_seconds` (time from a store job's
    creation in `_build_store_jobs` to `complete_store`), split into
    `vllm:kv_offload_store_pin_queue_duration_seconds` (waiting before the
    write began) and `vllm:kv_offload_store_pin_write_duration_seconds`
    (the worker's own CUDA-event-measured copy time, forwarded from
    `OffloadingConnectorWorker.get_finished()` — no new synchronization or
    per-job timestamp added on the worker's hot path), plus in-flight store
    gauges `vllm:kv_offload_inflight_store_jobs` /
    `vllm:kv_offload_inflight_store_blocks`.
  - Pinned/validated against vLLM `v0.24.0` (matches
    `vllm.cuda13.0.runtime_image_tag` / `vllm.cpu.runtime_image_tag` in
    `container/context.yaml` at the time this patch was written).
