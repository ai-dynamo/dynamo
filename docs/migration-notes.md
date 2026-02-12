# Migration Notes

This document tracks breaking changes and migration instructions for Dynamo.

## vLLM Endpoint Default Change (Upcoming)

**Breaking Change:** vLLM aggregated backend workers now generate model-specific endpoints by default.

### What Changed

**Before:** All vLLM aggregated backend workers used `dynamo.backend.generate`

**After:** Aggregated backend workers get unique endpoints like `dynamo.backend.generate_qwen_qwen2_5-7b_a1b2c3d4`

**Note:** Disaggregated workers (prefill, decode, encoder, decoder, processor) continue using hardcoded `"generate"` endpoints to preserve internal wiring and planner compatibility.

### Impact

- Running multiple vLLM instances with different models on separate servers now works correctly
- Requests for model "Qwen/Qwen2.5-7B-Instruct" will only route to workers serving that specific model
- Fixes issue #3665 (frontend routing to wrong backend)

### Migration

**If you rely on static endpoint names:**

Use the new `--endpoint` flag to specify an explicit endpoint:

```bash
python -m dynamo.vllm --model Qwen/Qwen2.5-7B-Instruct \
  --endpoint dyn://dynamo.backend.generate
```

**If you use multiple models:**

No action needed - endpoints are now automatically unique per model.

### Technical Details

- Only the main backend worker (aggregated/model-serving path) gets a unique endpoint
- Other worker types keep the hardcoded `"generate"` endpoint to preserve internal wiring:
  - **Disaggregated workers**: prefill, decode (planner compatibility required)
  - **Multimodal workers**: processor, encoder, decoder, encode-prefill
  - **Pipeline workers**: omni
- Endpoint format: `generate_{slug}_{hash8}` where:
  - `slug`: First 40 chars of slugified model name (lowercase, alphanumeric/-/_)
  - `hash8`: First 8 chars of sha256 hash of original model name
- Example: `"Qwen/Qwen2.5-7B-Instruct"` → `"generate_qwen_qwen2_5-7b-instruct_0311af13"`

### Known Limitations

- **Disaggregated mode**: Prefill and decode workers still use `"generate"` endpoint. Running multiple disaggregated models on the same cluster may still experience routing issues. This requires planner refactoring (tracked separately).
- **Workaround**: Use `--endpoint` flag to manually assign unique endpoints for disaggregated workers.

### Future Work

- Refactor planner to support model-specific endpoints for disaggregated workers
- Apply same fix to SGLang and TrtLLM backends (tracked in separate PRs)
