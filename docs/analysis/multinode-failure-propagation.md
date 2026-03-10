# Multinode Failure Propagation

## Problem

In multinode TP failover, each engine is a distributed group spanning multiple pods
(leader pod + worker pod(s)). When any container in the group fails, ALL containers
in that group must terminate so:
1. GPU memory is released for the shadow engine to wake
2. No stale NCCL connections persist
3. The shadow engine gets a clean distributed group

K8s has no native mechanism to cascade failure across pods at the container level.

## Two Approaches

### Approach 1: K8s Liveness Probes with Epoch + Canary

Uses K8s-native probe infrastructure with two additions: an epoch for session
identity, and a canary generation for active engine health.

#### Components

**Epoch identity (startup):**
- Leader container writes a UUID to `/shared/engine-N-epoch` at startup
- Leader's system health endpoint exposes the epoch
- Worker container fetches and stores the epoch at init time
- Worker's liveness probe checks the leader's epoch matches its stored copy

**Canary generation (active engine):**
- Background thread on the leader sends a 1-token generation every N seconds
- Uses a short timeout (10s) to avoid blocking on NCCL hangs
- On consecutive failures (e.g., 2), sets the health endpoint to unhealthy
- This catches worker death, GPU hang, NCCL deadlock, engine errors

**Probe configuration:**

Leader container:
```yaml
startupProbe:
  httpGet:
    path: /health
    port: 8100
  periodSeconds: 5
  failureThreshold: 120  # 10 min for model load + graph capture
livenessProbe:
  httpGet:
    path: /health
    port: 8100
  periodSeconds: 5
  failureThreshold: 3
```

Worker container:
```yaml
startupProbe:
  exec:
    command: ["sh", "-c", "curl -sf http://$LEADER:8100/epoch > /dev/null"]
  periodSeconds: 5
  failureThreshold: 120  # wait for leader to start
livenessProbe:
  exec:
    command: ["sh", "-c", "curl -sf http://$LEADER:8100/epoch | diff -q /tmp/my-epoch -"]
  periodSeconds: 5
  failureThreshold: 2
```

#### Failure Propagation

**Leader crashes (active):**
- Canary was keeping health endpoint alive
- Leader dies → health endpoint gone
- Worker liveness probe: connection refused → fails → K8s kills worker (10-15s)

**Worker crashes (active):**
- Canary generation does all-reduce → NCCL finds rank 1 missing → generation fails
- Canary timeout (10s) catches NCCL hang case
- 2 consecutive canary failures → health endpoint set to unhealthy
- Leader liveness probe fails → K8s kills leader (15-25s)
- Worker already dead, so full group is cleaned up

**Leader restarts faster than probe interval:**
- New leader has new epoch
- Worker's liveness probe compares epoch → mismatch → fails → worker killed
- Both restart fresh with matching epochs

**Failure during startup/standby:**
- No canary running (engine not active)
- Relies on NCCL timeout (~600s) for detection
- Acceptable — standby engines are not serving traffic

#### Detection Latency

| Scenario | Detection Time |
|----------|---------------|
| Leader crashes (active) | 10-15s (probe) |
| Worker crashes (active) | 15-25s (canary timeout + probe) |
| Leader crashes (standby) | 10-15s (probe) |
| Worker crashes (standby) | ~600s (NCCL timeout) |
| Leader crashes (startup) | Minutes (startup probe timeout) |
| Worker crashes (startup) | ~600s (NCCL timeout) |

#### Pros
- No external dependencies beyond K8s
- Epoch prevents restart-race false negatives
- Canary tests full inference path (GPU, NCCL, engine health)
- No Dynamo/vLLM code changes for probes (operator pod spec only)
- Canary requires a small code addition (background thread on leader)

#### Cons
- Unidirectional: worker probes leader, but leader cannot probe headless worker
- Worker death during standby/startup detected only via NCCL timeout (~600s)
- Canary generation adds small inference overhead
- Probe-based detection is 10-25s (acceptable but not sub-second)

---

### Approach 2: NATS KV Heartbeat with TTL

Uses NATS JetStream key-value store with TTL-based expiration for fast,
bidirectional failure detection.

#### Components

**NATS KV bucket:**
```
nats kv add failover-heartbeats --ttl 3s
```

**Container wrapper script (replaces entrypoint for each engine container):**
```bash
#!/bin/bash
KEY="engine-${ENGINE_ID}-${NODE_ROLE}"
PEER_KEY="engine-${ENGINE_ID}-$([ "$NODE_ROLE" = "leader" ] && echo worker || echo leader)"

# 1. Heartbeat publisher (background)
while true; do nats kv put failover "$KEY" alive 2>/dev/null; sleep 1; done &

# 2. Peer watcher — exits when peer key expires (background)
(nats kv watch failover "$PEER_KEY" --raw | grep -q "DEL") &

# 3. Engine process (background)
python3 -m dynamo.vllm "$@" &

# Wait for ANY to exit, then tear down all
wait -n
kill 0
```

#### How It Works

Each engine container runs three processes:
1. **Heartbeat publisher**: puts a key every 1s, resets the 3s TTL
2. **Peer watcher**: subscribes to peer's key, exits on DEL (TTL expiry)
3. **Engine process**: the actual vLLM engine

When any process exits, `wait -n` returns and `kill 0` terminates the container.

- Engine crashes → publisher dies → TTL expires → peer watcher triggers → peer container dies
- Peer dies → peer publisher dies → TTL expires → local watcher triggers → local container dies

#### Failure Propagation

**Leader crashes (any phase):**
- Leader publisher stops → key expires in 2-3s
- Worker watcher gets DEL → `wait -n` returns → `kill 0` → worker container dies
- GPU memory freed in 2-3s

**Worker crashes (any phase):**
- Worker publisher stops → key expires in 2-3s
- Leader watcher gets DEL → `wait -n` returns → `kill 0` → leader container dies
- Both containers dead, GPU memory freed in 2-3s

**During startup:**
- Publisher starts immediately with the container
- Works the same as during active serving — no startup probe gating needed

**During standby:**
- Engine is sleeping but process is alive → publisher keeps running
- Correctly stays alive (sleeping is not a failure)

#### Detection Latency

| Scenario | Detection Time |
|----------|---------------|
| Leader crashes (active) | 2-3s |
| Worker crashes (active) | 2-3s |
| Leader crashes (standby) | 2-3s |
| Worker crashes (standby) | 2-3s |
| Leader crashes (startup) | 2-3s |
| Worker crashes (startup) | 2-3s |

#### Pros
- Bidirectional: both leader and worker detect each other's death
- Uniform 2-3s detection across ALL phases (startup, standby, active)
- No health server needed on headless worker
- No Dynamo/vLLM code changes (wrapper script only)
- No startup ordering issues (watcher ignores missing keys, acts on DEL)
- Heartbeat is independent of engine load (separate background process)

#### Cons
- NATS dependency — if NATS goes down, behavior depends on client:
  - Watcher hangs (safe — no false DEL events)
  - Publisher fails silently (unsafe — TTL expires, false cascade)
  - Need to verify NATS client behavior during outage
- NATS network partition could cause false positives
- Adds `nats` CLI or client to container image
- `kill 0` may not reach vLLM subprocesses if they escaped the process group
  (need `shareProcessNamespace` or process group management)

---

## Combined Approach

The two schemes are complementary and can run simultaneously:

| Concern | NATS KV | K8s Probes + Canary |
|---------|---------|---------------------|
| Fast detection (all phases) | ✓ Primary | Backup |
| Active engine health (GPU/NCCL) | Not tested | ✓ Canary tests inference path |
| Works without NATS | ✗ | ✓ |
| Works without health server on worker | ✓ | ✗ (unidirectional) |
| Startup phase coverage | ✓ | ✗ (probe not active) |

**Recommendation:** Use NATS KV as the fast primary detection for process death.
Add the canary generation on the leader as a deeper health check that catches
GPU/NCCL issues that don't cause process death (hangs, corruption).
Keep K8s liveness probes with epoch as a safety net for when NATS is unavailable.

The canary is valuable regardless of which detection scheme is used — it's the
only mechanism that tests the full inference path through all ranks.

---

## Implementation Complexity

| Component | Approach 1 (Probes) | Approach 2 (NATS KV) | Combined |
|-----------|--------------------|-----------------------|----------|
| Operator pod spec changes | Probe config + epoch script | Wrapper entrypoint | Both |
| Dynamo code changes | Canary thread on leader | None | Canary thread |
| vLLM code changes | None | None | None |
| Infrastructure dependencies | K8s only | NATS JetStream | Both |
| New container image deps | None | nats CLI | nats CLI |
