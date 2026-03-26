# Intentional Hardcoded Strings in Refactored Code

This document explains the hardcoded strings that remain in the refactored code and why they are intentional (not bugs).

## 1. `"Frontend"` - Common Component Name

**Location:** Multiple files
- `tests/fault_tolerance/deploy/scenarios.py` (lines 81, 111, 218)
- `tests/fault_tolerance/deploy/legacy_parse_results.py` (lines 227, 336)
- `tests/fault_tolerance/deploy/test_deployment.py` (indirectly via scenarios)

**Why it's intentional:**
- `"Frontend"` is a **common component** shared across all backends (vLLM, SGLang, TRTLLM)
- There is **no component name class** for Frontend in `dynamo.planner.defaults`
- Frontend is not backend-specific, so it doesn't belong to any backend's ComponentName class
- This is the actual Kubernetes service name used consistently across all deployments

**Example usage:**
```python
# scenarios.py line 81
WORKER_READY_PATTERNS: Dict[str, Pattern] = {
    "Frontend": re.compile(r"added model"),  # Common to all backends
    # ...
}

# scenarios.py line 218
spec["Frontend"].replicas = replicas  # Common to all backends
```

**Should this be refactored?**
- **No** - Frontend is a shared component with no backend-specific variant
- If refactored, would need to create a `FrontendComponentName` class, but this seems unnecessary since it's a single constant value

---

## 2. `"TRTLLMWorker"` - Aggregated Deployment Name

**Location:**
- `tests/fault_tolerance/deploy/scenarios.py` (line 72)
- `tests/fault_tolerance/deploy/test_deployment.py` (line 383)
- `tests/fault_tolerance/deploy/legacy_parse_results.py` (lines 240, 341)

**Why it's intentional:**
- This is explicitly documented in the code with a comment: `# Aggregated uses different name (not in defaults yet)`
- TRTLLM aggregated deployments use `"TRTLLMWorker"` instead of `"TRTLLMDecodeWorker"`
- This name is **not yet defined** in `TrtllmComponentName` class in `dynamo.planner.defaults`
- The code handles this special case explicitly

**Example usage:**
```python
# scenarios.py line 72
WORKER_MAP = {
    "trtllm": {
        "decode": TrtllmComponentName.decode_worker_k8s_name,  # "TRTLLMDecodeWorker"
        "decode_agg": "TRTLLMWorker",  # Aggregated uses different name (not in defaults yet)
        "prefill": TrtllmComponentName.prefill_worker_k8s_name,
    },
}

# test_deployment.py line 383
if "agg" in scenario.deployment.name and "disagg" not in scenario.deployment.name:
    model = scenario.deployment["TRTLLMWorker"].model  # Special case for aggregated
else:
    model = scenario.deployment[TrtllmComponentName.decode_worker_k8s_name].model
```

**Should this be refactored?**
- **Future improvement** - Once `TrtllmComponentName` is extended to include `decode_agg_worker_k8s_name` or similar, this could be refactored
- For now, it's correctly handled as a special case with clear documentation

---

## 3. ~~`"decode"`~~ - SGLang Component Name (REFACTORED)

**Status:** ✅ **This has been refactored in the PR**

**Previous location:**
- `tests/fault_tolerance/deploy/test_deployment.py` (line 375)

**Refactoring:**
- Changed from hardcoded `"decode"` to `SGLangComponentName.decode_worker_k8s_name`
- This completes the refactoring for consistency with the PR objective
- The value is still `"decode"` (SGLang uses short names for Kubernetes limits), but now references the source of truth

**Current usage:**
```python
# test_deployment.py (after refactoring)
elif scenario.backend == "sglang":
    model = scenario.deployment[SGLangComponentName.decode_worker_k8s_name].model
```

---

## Summary

| String | Location | Reason | Refactor Needed? |
|--------|----------|--------|-------------------|
| `"Frontend"` | Multiple | Common component, no ComponentName class | No - by design |
| `"TRTLLMWorker"` | Multiple | Aggregated deployment name (not in defaults yet) | Future - when added to defaults |
| ~~`"decode"`~~ | ~~test_deployment.py:375~~ | ~~Actual SGLang component name value~~ | ✅ **Refactored** - now uses `SGLangComponentName.decode_worker_k8s_name` |

## Recommendations

1. **`"Frontend"`** - Keep as-is. It's a shared component name.
2. **`"TRTLLMWorker"`** - Keep as-is for now. Refactor when `TrtllmComponentName` is extended (tracked in issue #5116).
3. ~~**`"decode"`**~~ - ✅ **Completed**: Now uses `SGLangComponentName.decode_worker_k8s_name` for consistency.

## Conclusion

All remaining hardcoded strings are **intentional and documented**. They represent:
- Shared components (`Frontend`)
- Special cases with clear comments (`TRTLLMWorker`)

The refactoring successfully replaced all **unintentional** hardcoded backend-specific deployment names with component name class references. The SGLang `"decode"` string has also been refactored for consistency. ✅
