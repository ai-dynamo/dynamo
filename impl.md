# GPU Failover POC: Implementation Tracker

Reference: [plan.md](plan.md) · [decisions.md](decisions.md)

---

## Milestone 1: Failover Lock (Diff 1)

**Branch:** `failover/m1-flock-lock` (off `gms-shadow-experiment-v3`)

**Status:** Complete — all 6 tests pass

### Scope

- `FailoverLock` ABC in `lib/gpu_memory_service/failover_lock/interface.py`
- `FlockFailoverLock` implementation in `lib/gpu_memory_service/failover_lock/flock/lock.py`
- Re-export `__init__.py` files
- pytest test suite

### Files Created

```
lib/gpu_memory_service/failover_lock/
├── __init__.py
├── interface.py
└── flock/
    ├── __init__.py
    └── lock.py

tests/
└── test_failover_lock.py
```

### Test Plan

| # | Test | What it validates |
|---|------|-------------------|
| 1 | `test_acquire_release` | Single lock acquire/release, file contains engine_id, FD closed after release |
| 2 | `test_two_engines_contention` | Engine A holds lock, Engine B blocks, A releases, B acquires |
| 3 | `test_process_death_releases` | Fork child that acquires, SIGKILL child, parent acquires (kernel releases flock) |
| 4 | `test_owner` | `owner()` returns correct engine_id after acquire |
| 5 | `test_cross_process` | Two processes race via multiprocessing, exactly one wins, loser acquires after winner dies |

### Results

```
tests/fault_tolerance/gpu_memory_service/test_failover_lock.py::test_acquire_release PASSED
tests/fault_tolerance/gpu_memory_service/test_failover_lock.py::test_two_engines_contention PASSED
tests/fault_tolerance/gpu_memory_service/test_failover_lock.py::test_process_death_releases PASSED
tests/fault_tolerance/gpu_memory_service/test_failover_lock.py::test_owner PASSED
tests/fault_tolerance/gpu_memory_service/test_failover_lock.py::test_owner_separate_instance PASSED
tests/fault_tolerance/gpu_memory_service/test_failover_lock.py::test_cross_process_race PASSED

6 passed in 0.59s
```

### Notes

- No GPU required for this milestone
- No changes to existing code — purely additive
- Needed to `pip install pytest-asyncio` in the dynamo venv (not pre-installed)
- Run command: `python -m pytest tests/fault_tolerance/gpu_memory_service/test_failover_lock.py -v -o "addopts=" -o "filterwarnings="`
  - The `-o` overrides are needed because pyproject.toml has `--mypy` in addopts (requires `pytest-mypy` not installed) and a `pytest_benchmark` filter warning

---

## K8s Integration Test (Diff 6)

**Branch:** `failover/m6-operator` (SHA: `bd442b1a6`)

**Images:**

```
vLLM runtime:  dynamoci.azurecr.io/ai-dynamo/dynamo:failover-m6-bd442b1-vllm-runtime
Operator:      dynamoci.azurecr.io/ai-dynamo/dynamo:failover-m6-bd442b1-operator
```

### Status

- [ ] Image pushed to ACR
- [ ] DGD with `failover.enabled: true` applied
- [ ] Primary engine comes up and serves inference
- [ ] Shadow engine initializes, loads weights, enters STANDBY
- [ ] Kill primary — shadow acquires flock and wakes
- [ ] Inference resumes on shadow engine

### Test Notes

_Collaboration space for parallel agents — append progress below._
