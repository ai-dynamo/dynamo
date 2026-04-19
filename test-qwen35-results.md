# dynamo.vllm Qwen3.5 Test Results

**Date:** 2026-03-25/26
**Image:** `dynamo:latest-vllm-local-dev-03-25`
**Host GPUs:** GPU 0 (A400 4GB, unused), GPU 1 (RTX 6000 Ada 49GB), GPU 2 (RTX PRO 6000 Blackwell 98GB)

## Key Findings

1. **Qwen3.5 is multimodal** — all variants (2B, 27B, 35B-A3B-FP8) handle vision inputs natively
2. **AGG and MM Routing work across all sizes** — confirmed for 2B, 27B, and 35B-A3B-FP8
3. **P/D Disagg blocked by vLLM hybrid KV cache** — Qwen3.5's hybrid architecture (attention + Mamba/GDN layers) is incompatible with `--kv-transfer-config` which disables the hybrid KV cache manager
4. **E_PD/E_P_D blocked by outdated `transformers`** — standalone encode worker's `AutoModel.from_pretrained()` doesn't recognize `qwen3_5`

---

## Qwen/Qwen3.5-2B

Logs: `dynamo/logs/test-qwen35/`

| Topology | Status | Notes |
|----------|--------|-------|
| **AGG** | **PASS** | Text + multimodal both work |
| **MM Routing** | **PASS** | Text + multimodal through KV-aware router |
| **E_PD** | **FAIL** | `transformers` KeyError: `qwen3_5` |
| **E_P_D** | **FAIL** | Same |

---

## Qwen/Qwen3.5-27B

Logs: `dynamo/logs/test-qwen35-27b/`

| Topology | Status | Notes |
|----------|--------|-------|
| **AGG** | **PASS** | Text + multimodal. 51.1 GiB on 98GB Blackwell |
| **MM Routing** | **PASS** | Text + multimodal. First request ~92s cold start |
| **P/D Disagg** | **FAIL (OOM)** | 27B bf16 (54GB) doesn't fit on 49GB Ada |
| **E_PD** | **FAIL** | `transformers` KeyError: `qwen3_5` |
| **E_P_D** | **FAIL** | Same (confirmed from 2B, not re-run) |

---

## Qwen/Qwen3.5-35B-A3B-FP8 (MoE)

Logs: `dynamo/logs/test-qwen35-35b/`

| Topology | Status | Notes |
|----------|--------|-------|
| **AGG** | **PASS** | Text + multimodal. 34.23 GiB FP8 on 98GB Blackwell |
| **MM Routing** | **PASS** | Text + multimodal. Model served as `__internal` name but requests succeeded |
| **P/D Disagg** | **FAIL** | vLLM error: "Hybrid KV cache manager is disabled but failed to convert KV cache specs to one unified type." Qwen3.5 MoE's hybrid arch (attention + Mamba) incompatible with `--kv-transfer-config` |
| **E_PD** | **FAIL** | `transformers` KeyError: `qwen3_5` (expected) |
| **E_P_D** | **FAIL** | Same (expected) |

---

## Root Causes

### 1. E_PD / E_P_D: `transformers` doesn't support `qwen3_5`
- **Where:** `components/src/dynamo/vllm/multimodal_utils/model.py` → `AutoModel.from_pretrained()`
- **Fix:** Upgrade transformers, or use vLLM's native encoder path with Qwen3.5 added to `SupportedModels`

### 2. P/D Disagg: Hybrid KV cache incompatibility
- **Where:** vLLM `kv_cache_utils.py:1172` — hybrid KV cache manager disabled by `--kv-transfer-config`
- **Root cause:** Qwen3.5 has hybrid attention layers (standard attention + GatedDeltaNet/Mamba), requiring the hybrid KV cache manager. But `--kv-transfer-config` (required for P/D disagg with NixlConnector) forces it off.
- **Fix:** vLLM needs to support hybrid KV cache + KV transfer together, or the NixlConnector needs to handle heterogeneous KV cache specs

### 3. P/D Disagg OOM (27B only): Hardware limitation
- **Where:** 27B bf16 needs ~54GB, GPU 1 only has 49GB
- **Fix:** Use FP8 quantized variant or larger GPUs

---

## Summary Matrix

| Topology | 2B | 27B | 35B-A3B-FP8 |
|----------|-----|------|-------------|
| AGG (text + multimodal) | ✅ | ✅ | ✅ |
| MM Routing (text + multimodal) | ✅ | ✅ | ✅ |
| P/D Disagg | not tested | ❌ OOM | ❌ hybrid KV cache |
| E_PD | ❌ transformers | ❌ transformers | ❌ transformers |
| E_P_D | ❌ transformers | ❌ transformers | ❌ transformers |

**Bottom line:** `dynamo.vllm` supports Qwen3.5 for **AGG** and **MM Routing** topologies. Disaggregated topologies (P/D, E_PD, E_P_D) have blockers that need upstream fixes in vLLM and/or transformers.
