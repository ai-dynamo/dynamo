# Verification of `vllm:nixl_num_pending_sends` metric

End-to-end check for `dynamo.vllm.custom_connectors.nixl_with_pending_metrics`,
run through Dynamo's own runtime (`python -m dynamo.vllm` +
`python -m dynamo.frontend`) — the same launch path the customer's DGD uses.

## Environment

```
Dynamo:  v1.1.0        (matches "Dynamo 1.1.0 vllm runtime on 1.0.1 operator")
vLLM:    0.19.0        (the exact pin in Dynamo 1.1.0 pyproject.toml)
NIXL:    1.1.0
Python:  3.12
GPU:     1× (the script runs prefill + decode on the same GPU at 0.30 mem each)
```

## What's in here

| File | Purpose |
|---|---|
| `dynamo_runtime_test.sh` | Full disagg test — `dynamo.frontend` + decode worker + prefill worker, send request, kill decode mid-transfer, dump prefill `/metrics` |

## How to run

```bash
# Install the matching versions if you don't already have them
uv venv .venv --python 3.12
. .venv/bin/activate
uv pip install "ai-dynamo==1.1.0" "vllm==0.19.0" nixl

# Run from any directory (script resolves its own paths)
PYTHON=$(which python) \
    bash components/src/dynamo/vllm/custom_connectors/verification/dynamo_runtime_test.sh
```

## What the script does

1. Sets `DYN_DISCOVERY_BACKEND=file`, `DYN_REQUEST_PLANE=tcp`,
   `DYN_EVENT_PLANE=zmq` so we don't need NATS or etcd.
2. Launches `python -m dynamo.frontend` on `:8000`.
3. Launches `python -m dynamo.vllm --disaggregation-mode decode` on
   `DYN_SYSTEM_PORT=8081`.
4. Launches `python -m dynamo.vllm --disaggregation-mode prefill` on
   `DYN_SYSTEM_PORT=8082` (separate `VLLM_NIXL_SIDE_CHANNEL_PORT`).
5. Waits until both workers report `vllm:` metrics (i.e., engines loaded).
6. Sends a successful chat completion via the frontend.
7. Sends another request, sleeps 3 seconds, then `SIGKILL`s the decode
   worker mid-transfer.
8. Dumps prefill metrics at +3 s and +13 s after the kill. Expected: the new
   gauge stays at 1 while `vllm:num_requests_running` and
   `vllm:num_requests_waiting` both stay at 0 — the diagnostic gap the
   customer was hitting.

## Last observed result

```
====================================================================
  T0: cold start — through Dynamo runtime
====================================================================
  vllm:num_requests_running                                  0.0
  vllm:num_requests_waiting                                  0.0
  vllm:nixl_num_pending_sends                                0.0   [NEW]

[T1] response: "The sum of 7 and 8 is 15."  ← real disagg roundtrip

====================================================================
  T1: after successful round-trip
====================================================================
  vllm:nixl_num_pending_sends                                1.0   [NEW]

====================================================================
  T2: 3s after decode killed — invisible occupancy, prefill side
====================================================================
  vllm:num_requests_running                                  0.0    ← invisible!
  vllm:num_requests_waiting                                  0.0    ← invisible!
  vllm:nixl_num_pending_sends                                1.0   [NEW]

====================================================================
  T3: 13s after decode killed — strand holds, no traffic
====================================================================
  vllm:nixl_num_pending_sends                                1.0   [NEW] (stable)
```
