# Multinode Liveness Probe Specification

## Goal

When any container in a multinode engine group (leader + worker(s)) fails, all
containers in that group must terminate so GPU memory is released and the shadow
engine can take over.

## Architecture

```
Leader Pod                              Worker Pod
┌────────────────────────────┐         ┌────────────────────────────┐
│ engine-0 (active)          │         │ engine-0 (headless worker) │
│   system port: 8100        │         │   no system port           │
│   /health → engine health  │         │                            │
│   /epoch  → session UUID   │         │   startupProbe:            │
│                            │         │     GET leader:8100/epoch  │
│ engine-1 (shadow)          │         │   livenessProbe:           │
│   system port: 8101        │         │     GET leader:8100/health │
│   /health → engine health  │         │     + epoch match check    │
│   /epoch  → session UUID   │         │                            │
│                            │         │ engine-1 (headless worker) │
│ GMS sidecar                │         │   (same probe pattern)     │
└────────────────────────────┘         └────────────────────────────┘
```

## Components

### 1. Epoch Identity

Each engine leader container generates a unique session ID (UUID) at startup and
exposes it via HTTP.

**Leader container startup command (injected by operator):**
```bash
EPOCH=$(cat /proc/sys/kernel/random/uuid) && \
echo $EPOCH > /shared/engine-${ENGINE_ID}-epoch && \
exec python3 -m dynamo.vllm ...
```

**Leader system status server:**
Add a `/epoch` endpoint that serves the file content:
```
GET /epoch → 200 "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
```

This is a small addition to the Dynamo system status server (~5 lines).

### 2. Worker Startup Probe

The worker container's startupProbe waits for the leader to be ready and stores
the leader's epoch locally for later comparison.

```yaml
startupProbe:
  exec:
    command:
      - sh
      - -c
      - "curl -sf http://$LEADER:${LEADER_SYSTEM_PORT}/epoch > /tmp/my-epoch"
  periodSeconds: 5
  failureThreshold: 120  # 10 minute window for leader to load model
```

**Behavior:**
- Fails while leader is still starting (connection refused or no epoch yet)
- Succeeds once leader has initialized and written its epoch
- Stores the epoch to `/tmp/my-epoch` for liveness comparison
- Does NOT block the container from running — the engine process starts
  immediately and joins torch.distributed while the probe is still failing

### 3. Worker Liveness Probe

After the startup probe passes, the liveness probe continuously checks that:
1. The leader is healthy (`/health` returns 200)
2. The leader is the same session the worker initialized with (epoch matches)

```yaml
livenessProbe:
  exec:
    command:
      - sh
      - -c
      - >-
        curl -sf http://$LEADER:${LEADER_SYSTEM_PORT}/health &&
        curl -sf http://$LEADER:${LEADER_SYSTEM_PORT}/epoch |
        diff -q /tmp/my-epoch -
  periodSeconds: 5
  failureThreshold: 2
```

**Behavior:**
- First `curl`: checks leader health. Fails if leader is dead (connection
  refused) or unhealthy (canary failure set health to 503).
- Second `curl` + `diff`: checks epoch. Fails if leader restarted (new UUID).
- `failureThreshold: 2`: two consecutive failures (10s) before K8s kills the
  worker container.

### 4. Leader Probes

The leader keeps its existing probes unchanged:

```yaml
startupProbe:
  httpGet:
    path: /health
    port: ${SYSTEM_PORT}
  periodSeconds: 5
  failureThreshold: 120
livenessProbe:
  httpGet:
    path: /health
    port: ${SYSTEM_PORT}
  periodSeconds: 5
  failureThreshold: 3
```

### 5. Canary Generation (Optional Enhancement)

A background thread on the leader sends a small inference request (1 token)
every N seconds. If the request fails or times out consecutively, the canary
sets the health endpoint to unhealthy.

This catches failures that don't kill the leader process:
- Worker death (NCCL error on all-reduce)
- GPU hang
- Engine internal corruption

```
Canary thread:
  every 5s:
    result = POST localhost:8100/v1/completions {"prompt":"hi","max_tokens":1} timeout=10s
    if result failed:
      consecutive_failures += 1
    else:
      consecutive_failures = 0
    if consecutive_failures >= 2:
      set_health_status(unhealthy)
```

When the canary sets health to unhealthy:
- Leader's own liveness probe fails → K8s kills leader
- Worker's liveness probe (checks /health) fails → K8s kills worker
- Full cascade in 10-15s

## Failure Scenarios

### Leader crashes while active

```
T=0s    Leader process dies
        /health and /epoch endpoints stop responding
T=5s    Worker liveness probe fires → connection refused → FAIL
T=10s   Worker liveness probe fires again → FAIL (threshold=2 reached)
T=11s   K8s kills worker container → GPU memory freed
T=12s   Flock released (leader died at T=0) → shadow engine wakes
```
Detection: **10-11s**

### Worker crashes while active

```
T=0s    Worker process dies → NCCL rank missing
T=5s    Leader canary generation fires → all-reduce fails → canary fails
T=10s   Leader canary fires again → fails (threshold=2)
        Canary sets /health to unhealthy
T=15s   Leader liveness probe → /health returns 503 → FAIL
T=20s   Leader liveness probe → 503 → FAIL (threshold=2 on probe not needed,
        threshold is on canary side. Leader probe just sees unhealthy.)
T=21s   K8s kills leader container
        Flock released → shadow engine wakes
```
Detection: **15-21s**

### Leader restarts quickly

```
T=0s    Leader crashes
T=2s    K8s restarts leader container → new epoch written
T=5s    Worker liveness probe: /health → 200 (new leader healthy)
        /epoch → new UUID, diff fails → FAIL
T=10s   Second probe → FAIL (threshold=2)
T=11s   K8s kills worker → clean restart
```
Detection: **10-11s** (epoch prevents false "all clear")

### Failure during standby (sleeping)

```
T=0s    Leader crashes while sleeping
        /health was returning 200 (standby branch)
T=5s    Worker liveness probe → connection refused → FAIL
T=10s   FAIL again → K8s kills worker
```
Detection: **10-11s** (same as active — probe checks reachability)

### Failure during startup

```
T=0s    Worker crashes during model loading
        Leader's NCCL hangs
        Worker startupProbe has not passed yet → no liveness enforcement
        K8s restarts worker (container restart, not probe-driven)
T=?s    New worker joins → may succeed or fail depending on NCCL state
        If NCCL group is stale → leader eventually times out → both restart
```
Detection: **Minutes** (NCCL timeout). Startup failures are not covered by probes.

## Operator Implementation

The operator needs to:

1. **Inject epoch generation into leader container command:**
   Prepend `EPOCH=$(cat /proc/sys/kernel/random/uuid) && echo $EPOCH > /shared/engine-${ENGINE_ID}-epoch && exec `
   to the existing container command.

2. **Add `/epoch` route to system status server:**
   Small Dynamo code change — read `/shared/engine-${ENGINE_ID}-epoch` and return
   contents. Or serve it from the same startup script via a minimal HTTP server
   sidecar.

3. **Set worker container probes:**
   For multinode failover workers, inject startupProbe and livenessProbe as shown
   above. The `$LEADER` and `$LEADER_SYSTEM_PORT` variables come from LWS/Grove
   deployer (already available as `$LWS_LEADER_ADDRESS` or constructed from
   Grove env vars).

4. **Implement canary thread (optional):**
   Background thread in the vLLM main loop that runs test inferences and controls
   health status. This is a Dynamo code change in `components/src/dynamo/vllm/main.py`.

## Environment Variables

| Variable | Source | Used By |
|----------|--------|---------|
| `ENGINE_ID` | Operator | Epoch filename, probe port selection |
| `LEADER_SYSTEM_PORT` | Operator (8100 + ENGINE_ID) | Worker probes |
| `LWS_LEADER_ADDRESS` or `GROVE_*` | LWS/Grove controller | Worker probe target hostname |
| `FAILOVER_LOCK_PATH` | Operator | Flock-based failover (unchanged) |

## Limitations

- **Unidirectional during standby:** Worker death while sleeping is not detected
  by the leader until it tries to wake (then NCCL timeout). The canary only runs
  on the active engine.
- **Startup phase:** Failures during model loading / graph capture rely on NCCL
  timeout (~600s) for detection. Probes are not active during startup.
- **Probe latency:** 10-21s detection for active engine failures. Faster detection
  requires the NATS KV scheme (2-3s) described in `multinode-failure-propagation.md`.
