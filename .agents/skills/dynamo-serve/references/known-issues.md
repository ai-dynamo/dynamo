# Local-Dev Known Issues

Stable issue patterns specific to running `python3 -m dynamo.<backend>`
on a workstation. Strict 6-element shape per.

---

### HF token not visible to the worker process

**Symptom:** Worker starts but logs `401 Unauthorized` on the gated model's URL; `/v1/models` returns empty.

**Root cause:** `HF_TOKEN` env var is set in the user's interactive shell but not in the shell the worker process is running under (different terminal, different login shell, missing export).

**Affected:** All local-dev runs of gated HuggingFace models (Llama family, some Qwen variants).

**Fix:** Either `export HF_TOKEN=<token>` in the same terminal before launching the worker, or use `huggingface-cli login` (which writes a persistent token at `~/.cache/huggingface/token`).

**Verify:** `python3 -c "from huggingface_hub import HfApi; print(HfApi().whoami())"` returns your account info without error.

Source: (K8s variant).

---

### Port 8000 already in use

**Symptom:** Worker fails to start with `OSError: [Errno 48] Address already in use`.

**Root cause:** Port 8000 is bound by another process — often a previous worker that didn't shut down cleanly, or a sibling service.

**Affected:** Any local-dev run on a host where port 8000 is occupied.

**Fix:** Either kill the prior process (`lsof -ti:8000 | xargs kill`) or pick a different port (`--port 8001`). When iterating between two workers (disagg), assign distinct ports.

**Verify:** `lsof -i:<port>` returns nothing after the fix.

Source: standard POSIX socket behavior.

---

### GPU memory exhausted on shared workstation

**Symptom:** Worker crashes during weight load with `CUDA out of memory`. `nvidia-smi` shows another process holding most of the GPU's memory.

**Root cause:** The GPU is shared and another process (a previous worker, a notebook, a fine-tuning run) is holding the memory.

**Affected:** Shared dev boxes. Common in CI environments without GPU isolation.

**Fix:** Either (a) free the GPU (`nvidia-smi --query-compute-apps=pid,used_memory --format=csv` then `kill` the offending PID), (b) constrain the worker via `--gpu-memory-utilization 0.5` (vLLM) or `--mem-fraction-static 0.5` (SGLang), or (c) pick a different GPU via `CUDA_VISIBLE_DEVICES=<idx>`.

**Verify:** `nvidia-smi --query-gpu=memory.free --format=csv,noheader` reports enough free memory for the model.

Source: standard CUDA OOM behavior.

---

### Disagg prefill worker exits with NIXL deprecation

**Symptom:** Prefill worker exits immediately with `ValueError: --connector is deprecated and the default is no longer nixl`.

**Root cause:** Per. Dynamo 1.0.0+ requires `--kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}'` explicitly. The v0.9.x recipe of passing only `--disaggregation-mode prefill` was removed.

**Affected:** Dynamo 1.0.0+ disagg runs.

**Fix:** Per. Add the explicit transfer config to the prefill invocation.

**Verify:** Prefill worker stays running and accepts requests from the decode worker.

Source:.

---

### Workstation works, K8s deploy doesn't

**Symptom:** A model runs fine locally but the same model in a recipe-based DGD fails or behaves differently.

**Root cause:** Local pip-installed `ai-dynamo[<backend>]` pulls the PyPI version of NIXL (`nixl[cu12]==1.1.0` per). The Dynamo container ships a different NIXL built from source (per `container/context.yaml`'s `nixl_ref`). Per the 1.2.0 line all containers pin `nixl_ref: 0.10.1`. The local-vs-container NIXL drift can cause behavior differences in disagg KV transfer.

**Affected:** Disagg deployments. Aggregated mode rarely hits this.

**Fix:** Either (a) accept that local-dev is approximate and trust the recipe's published numbers, (b) replicate the container's NIXL version locally by installing the same `nixl[cu12]` pin the container builds, or (c) skip local-dev for disagg and iterate in a small K8s cluster instead.

**Verify:** `pip show nixl` and compare to the container's `nixl_ref` value in `container/context.yaml` on the target release branch.

Source:,.

---

### Worker logs "weights downloading" indefinitely

**Symptom:** Worker stays in startup; logs show repeated "downloading shard X/N" lines without completion. `nvidia-smi` shows no GPU activity yet.

**Root cause:** HF download is rate-limited or stalled (network, HF Hub backpressure). Large models (>10 GB) can take 30+ minutes on a slow link.

**Affected:** First-time loads of large models on slow networks.

**Fix:** Either (a) pre-download with `huggingface-cli download <model> --local-dir <path>` and pass `--model <path>` to the worker, or (b) use a faster network mirror, or (c) wait it out. `~/.cache/huggingface/hub/` accumulates the shards; subsequent runs reuse them.

**Verify:** `du -sh ~/.cache/huggingface/hub/models--<owner>--<name>` shows the full model size on disk.

Source: standard HF Hub behavior.
