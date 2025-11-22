# Fault Tolerance Test Architecture

This document explains the relationship between **scenarios**, **deployment specs**, **load**, and **failures** in the fault tolerance testing framework.

## Core Components

### Scenario
Top-level test configuration combining:
- **Deployment Spec**: Kubernetes deployment (workers, replicas, TP/DP config)
- **Load**: Client load generation (clients, requests, tokens, thresholds)
- **Failures**: List of failures to inject (timing, pod/process targets)
- **Model**: Optional model identifier
- **Backend**: Backend type (vllm, sglang, trtllm)
- **Checkers**: Validation checkers for post-test validation

### Deployment Spec
Kubernetes deployment configuration loaded from YAML files. Defines worker types, replica counts (data parallelism), tensor parallel size, and backend-specific arguments.

### Load
Client load configuration: number of clients (default: 10), requests per client (default: 150), input/output token lengths (default: 100), retry settings, client type ("aiperf" or "legacy"), and success threshold (default: 90.0%).

### Failure
Fault injection definition: `time` (seconds relative to previous failure), `pod_name` (target pod/service), `command` ("delete_pod" or process name), `signal` (SIGKILL, SIGINT, etc.), and `replicas` (number to affect, default: 1).

## Execution Flow

### Test Execution Sequence

1. **Deployment**: The deployment spec is deployed to Kubernetes
2. **Client Launch**: Client processes are started (in parallel)
3. **Load Generation**: Clients send requests while failures are injected
4. **Failure Injection**: Failures are injected sequentially (see below)
5. **Teardown**: Deployment is cleaned up
6. **Validation**: Checkers validate test results

### Parallel vs Sequential Execution

#### Can They Be Run in Parallel?

**Clients run in parallel**: Multiple client processes are spawned concurrently using `multiprocessing.Process`. Each client:
- Selects a frontend pod using round-robin
- Sets up port forwarding
- Generates load independently

**Failures are injected sequentially**: The `_inject_failures()` function processes failures one at a time:

```python
def _inject_failures(failures, logger, deployment: ManagedDeployment):
    affected_pods = {}

    for failure in failures:
        time.sleep(failure.time)  # Wait before injecting
        # ... inject failure ...

    return affected_pods
```

**Load and failures run concurrently**: Clients generate load while failures are being injected. The test uses a context manager pattern:

```python
with _clients(...):  # Clients run in background
    # Inject failures while clients are running
    affected_pods = _inject_failures(scenario.failures, logger, deployment)
```

### Multiple Failures Execution

**Failures are executed sequentially**, not in parallel. Each failure has a `time` field that specifies how many seconds to wait **after the previous failure** before injecting the current one.

**Example**:
```python
failures = [
    Failure(time=30, pod_name="VllmDecodeWorker", command="delete_pod"),  # After 30s
    Failure(time=60, pod_name="Frontend", command="delete_pod"),         # After 60s more (90s total)
    Failure(time=30, pod_name="VllmPrefillWorker", command="SIGKILL"),   # After 30s more (120s total)
]
```

**Execution timeline**:
- T=0s: Test starts, clients begin sending requests
- T=30s: First failure injected (delete decode worker pod)
- T=90s: Second failure injected (delete frontend pod)
- T=120s: Third failure injected (kill prefill worker process)

**Note**: The `time` field is relative to the **previous** failure, not absolute time from test start.

## Success Criteria

### Test Success Definition

A scenario is considered successful if:

1. **No exceptions raised**: The test completes without raising unhandled exceptions
2. **Validation checkers pass**: All checkers in `scenario.checkers` pass their validation
3. **Success rate threshold met**: The success rate (successful requests / total requests) meets or exceeds `scenario.load.success_threshold` (default: 90.0%)

### Validation Stages

Validation happens in **two stages**:

#### Stage 1: Scenario Verification
Checkers verify that the test scenario executed correctly:
- **PodDeletionChecker**: Verifies pods were actually deleted (via K8s events)
- **ProcessTerminationChecker**: Verifies processes were terminated
- **ContainerRestartChecker**: Verifies containers restarted after failures

These checkers ensure the failures were actually injected, not just that the code ran.

#### Stage 2: Results Validation
Checkers verify system behavior based on parsed results:
- **SuccessRateChecker**: Verifies success rate meets threshold
- **RecoveryTimeChecker**: Verifies recovery time is within acceptable bounds
- **NoFailuresChecker**: Verifies no unexpected failures occurred

### Success Criteria by Component

| Component | Success Criteria |
|-----------|------------------|
| **Individual Failure** | Not evaluated separately - failures are part of the scenario |
| **Whole Scenario** | All checkers pass AND success rate ≥ threshold AND no exceptions |
| **Test Execution** | No unhandled exceptions during test execution |

### Failure Handling

- **If a checker fails**: An `AssertionError` is raised, causing the test to fail
- **If success rate is below threshold**: The `SuccessRateChecker` raises an `AssertionError`
- **If parsing fails**: The test continues but validation is skipped (warning logged)
- **If validation errors occur** (non-assertion exceptions): The test continues but validation is skipped (warning logged)

### Example Success Flow

```
1. Test starts → Deploy deployment spec
2. Clients launch → Generate load in parallel
3. Failures injected → Sequentially at specified times
4. Test completes → Clients finish, deployment torn down
5. Results parsed → Extract metrics from client logs
6. Validation runs:
   ✓ PodDeletionChecker: Pod was deleted (K8s events confirm)
   ✓ SuccessRateChecker: 95% success rate ≥ 90% threshold
   ✓ RecoveryTimeChecker: Recovery time 45s < 60s limit
7. Test passes ✓
```

## Relationship Summary

```
Scenario
├── Deployment Spec (what to deploy)
│   ├── Worker types and counts
│   ├── Resource requirements
│   └── Backend configuration
│
├── Load (how to generate traffic)
│   ├── Client count and concurrency
│   ├── Request parameters
│   └── Success thresholds
│
└── Failures (what faults to inject)
    ├── Failure 1 (time=30s)
    ├── Failure 2 (time=60s)
    └── Failure 3 (time=30s)
        └── Executed sequentially
```

## Defined Scenarios

Scenarios are automatically generated from the Cartesian product of deployments and failures. The naming convention is: `{deployment_name}-{failure_name}`.

### Deployment Configurations

**Standard Deployments** (for vllm, sglang, trtllm backends):
- **Aggregated (agg)**: `{backend}-agg-tp-{1|2|4}-dp-{1|2}`
- **Disaggregated (disagg)**: `{backend}-disagg-prefill-tp-{1|2|4}-decode-tp-{1|2|4}-dp-{1}`

**MoE Deployments** (vllm only):
- `vllm-moe-agg-tp-1-dp-2`
- `vllm-moe-disagg-tp-1-dp-2`

**Total Standard Deployments**: 3 backends × 2 types × 4 configs = 24 (some disagg configs skipped when TP>1 and DP>1)

### Failure Types

**Common Failures** (all backends):
- `none`: No failure injection (baseline test)
- `frontend`: Terminate frontend process (SIGINT to `dynamo.frontend`)
- `frontend_pod`: Delete frontend pod
- `decode_worker`: Terminate decode worker process (SIGKILL to `dynamo.{backend}`)
- `decode_worker_pod`: Delete decode worker pod
- `prefill_worker`: Terminate prefill worker process (SIGKILL to `dynamo.{backend}`) - disagg only
- `prefill_worker_pod`: Delete prefill worker pod - disagg only

**vLLM-Specific Failures**:
- `vllm_decode_engine_core`: Kill VLLM::EngineCore process in decode worker
- `vllm_prefill_engine_core`: Kill VLLM::EngineCore process in prefill worker - disagg only

**SGLang-Specific Failures**:
- `sglang_decode_scheduler`: Kill sglang::scheduler process in decode worker
- `sglang_decode_detokenizer`: Kill sglang::detokenizer process in decode worker
- `sglang_prefill_scheduler`: Kill sglang::scheduler process in prefill worker - disagg only
- `sglang_prefill_detokenizer`: Kill sglang::detokenizer process in prefill worker - disagg only

**Token Overflow Scenarios**:
- `vllm_agg_token_overflow_2x`
- `vllm_disagg_token_overflow_2x`
- `trtllm_agg_token_overflow_2x`
- `trtllm_disagg_token_overflow_2x`
- `sglang_agg_token_overflow_2x`
- `sglang_disagg_token_overflow_2x`

### Example Scenario Names

- `vllm-agg-tp-1-dp-1-frontend_pod`: vLLM aggregated, TP=1, DP=1, delete frontend pod
- `sglang-disagg-prefill-tp-2-decode-tp-2-dp-1-decode_worker`: SGLang disaggregated, TP=2, delete decode worker pod
- `trtllm-agg-tp-4-dp-1-none`: TRT-LLM aggregated, TP=4, DP=1, no failures (baseline)
- `vllm-moe-agg-tp-1-dp-2-decode_worker_pod`: vLLM MoE aggregated, delete decode worker pod

### Total Scenario Count

Approximately **200+ scenarios** generated from:
- 24 standard deployments × ~7-9 failures per deployment
- 2 MoE deployments × ~7 failures
- 6 token overflow scenarios

## Key Takeaways

1. **Scenarios combine deployment, load, and failures** into a single test configuration
2. **Clients run in parallel**, but **failures are injected sequentially**
3. **Load and failures run concurrently** - failures are injected while clients are generating load
4. **Success is evaluated at the scenario level**, not per failure
5. **Success requires**: No exceptions + All checkers pass + Success rate ≥ threshold
6. **Multiple failures use relative timing** - each failure's `time` is relative to the previous one

