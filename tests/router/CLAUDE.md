# Router Test Authoring Guidance

- Put reusable router test scenarios and core assertions in `tests/router/common.py`.
- Keep backend-specific e2e files thin. Files such as `test_router_e2e_with_mockers.py`, `test_router_e2e_with_vllm.py`, `test_router_e2e_with_sglang.py`, `test_router_e2e_with_trtllm.py`, and `test_router_e2e_with_unified.py` should mostly configure the backend variant, parameterize meaningful cases, and call shared helpers.
- Use `tests/router/e2e_harness.py` for backend-agnostic e2e orchestration helpers that sit between thin backend wrappers and the core scenarios in `common.py`.
- Prefer parameterized wrappers over copied test bodies when a scenario should run across multiple backends, router modes, request planes, or storage backends.
- Put low-level shared utilities in `tests/router/helper.py`; examples include request sending, readiness polling, runtime setup, indexer waits, and response parsing helpers.
- Keep process lifecycle wrappers in `tests/router/router_process.py` or the existing backend-specific process helpers, rather than embedding subprocess management inside scenario logic.
- New e2e tests should assert behavior through stable surfaces such as response `nvext` fields, router helper APIs, metrics, or structured event dumps. Avoid log parsing unless there is no better observable contract.
