# Nemotron 3.5 Super VL (NVFP4) — TensorRT-LLM

Aggregated serving recipe for `nvidia/nemotron_35_super_conservative_fp8kv` on
**1x B200 (SM100)**, TensorRT-LLM backend.

## Model

NVFP4 weights + FP8 KV (modelopt 0.42.0, group_size 16), 79 GiB on disk.
Architecture `NemotronH_Omni_Reasoning_V3`: 88 layers — 40 linear-attention
(Mamba2) + 40 MoE (512 experts, top-22) + 8 full-attention — hidden 4096,
vocab 131072, max_position 262144, RADIO ViT-H/16 vision tower. Vision-only
despite the "Omni" name (`sound_config` is null).

It is a **reasoning model**: it emits chain-of-thought, so a small `max_tokens`
truncates before the answer. Budget tokens accordingly and set a reasoning
parser if you want the thinking split out.

Only the NVFP4 exports are supported here. The BF16 export
(`nemotron-3.5-super-pre-ea-vl-08282026`, 248.9 GiB) uses the legacy key layout
and needs tp>=2.

## Variant

| Variant | GPUs | Parallelism | Notes |
|---|---|---|---|
| `trtllm/agg-b200` | 1x B200 | tp=1 | aggregated, single worker |

## Container

Set the worker image in `trtllm/agg-b200/deploy.yaml`. It must be built from a
TensorRT-LLM that carries two fixes not yet in a released tag:

- **NVFP4 vision weight-key remap** (upstream PR #18526, merged 2026-09-03). The
  NVFP4 exports ship the canonical layout while TRT-LLM's native loader wanted
  the legacy one; without it the model fails to load with
  `Missing key(s) in state_dict: "0.weight"...`.
- **MTP forward fix** (not yet upstream). The VL wrapper drops `spec_metadata`
  and `resource_manager` when calling `self.llm.forward()`, so one-model
  speculative decoding fails with
  `conv_state_indices must have shape (batch_size)` or `KeyError: 88`.

The first is required for the model to load at all. The second is required for
the MTP configuration this recipe ships; comment out `speculative_config` in
`agg.yaml` to run without it. Drop the requirement once a runtime image ships
that already contains them.

The image tag must parse as a semantic version, or the operator rejects the
deployment and asks for `runtimeVersionOverride`.

## Deploy

```bash
kubectl apply -f model-cache/model-cache.yaml
kubectl apply -f model-cache/model-download.yaml   # waits for the checkpoint
kubectl apply -f trtllm/agg-b200/deploy.yaml
```

Set the worker and frontend `image:` to your built image first; the manifest
ships a `REPLACE_WITH_BUILT_IMAGE` placeholder rather than a stale digest.

## Configuration

`max_batch_size 4`, `max_num_tokens 8192`, `max_seq_len 4096`, and
`kv_cache_free_gpu_memory_fraction 0.5` are the values the model was validated
at. They are conservative — the KV and Mamba state budget for this hybrid
architecture has not been swept.

`max_input_len` is set explicitly. TensorRT-LLM defaults it to **1024**, which
silently truncates long prompts; raise it further for long-context workloads.

## Limitations

- **Image and video paths are untested.** All validation to date is text-only,
  despite this being a VL model.
- **MTP trades tail latency for per-user speed.** It is enabled with
  `max_draft_len: 1`, the one MTP layer this checkpoint ships. Measured on 2x
  B200 at c=64, it raises output tokens/sec/user from 30.6 to 79.5 but leaves
  aggregate throughput unchanged (1,447 -> 1,449 tok/s) and worsens TTFT p99
  from 21.3 s to 30.3 s, because at that concurrency the draft compute competes
  with other requests rather than filling idle capacity. Comment out
  `speculative_config` in `agg.yaml` if tail latency matters more than per-user
  token rate. It does not conflict with prefix reuse: cache read is 96.86% with
  MTP both on and off.
- B200 only. No GB200 or H200 variant has been validated.
