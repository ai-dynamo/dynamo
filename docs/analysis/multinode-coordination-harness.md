# Multinode Coordination Harness Design

## Problem

In multinode TP failover, each engine is a distributed group spanning multiple
pods (leader + worker(s)). These processes must form a consistent group and if
any member fails, ALL members must terminate promptly so:

1. GPU memory is released for the shadow engine
2. No stale NCCL connections persist
3. A clean re-formation can occur on restart

K8s has no native mechanism to cascade a container failure across pods. We need
a coordination layer that provides group membership, failure detection, and
coordinated teardown.

## Design

The coordination uses etcd leases — each member holds a lease that auto-expires
when the process dies. Members watch each other's keys to detect failures.

### Key Insight

The group is identified by the leader. Workers don't discover each other — they
only know the leader. The leader tracks all workers. Worker-to-worker failure
propagation cascades through the leader: worker dies → leader detects → leader
exits → remaining workers detect leader gone → they exit.

### etcd Keys

```
leaders/{group_id}                              = "{leader_hash}"   (lease-backed)
groups/{group_id}/{leader_hash}/rank-{N}        = "{worker_uuid}"   (lease-backed)
groups/{group_id}/{leader_hash}/start           = "go"              (lease-backed)
```

The leader hash is a UUID generated on each startup — it uniquely identifies a
leader instance. If the leader restarts, the hash changes. Workers use this to
detect that their leader is no longer the same instance they joined.

### Leader Lifecycle

```
1. STARTUP
   - Generate unique hash (UUID)
   - Create etcd lease (TTL=5s)
   - Put leaders/{group_id} = hash (attached to lease)
   - Start lease keepalive in background

2. FORMATION (barrier 1)
   - Watch groups/{group_id}/{hash}/rank-* keys
   - Wait for all expected ranks (1 through nnodes-1) to register
   - Timeout after FORMATION_TIMEOUT seconds → exit
   - Record each worker's UUID for change detection

3. START SIGNAL (barrier 2)
   - Put groups/{group_id}/{hash}/start = "go" (attached to lease)
   - Start the engine process

4. MONITORING
   - Poll worker keys every 1s
   - If any recorded worker key is DELETED → exit (worker died, lease expired)
   - If any recorded worker key VALUE CHANGED → exit (worker restarted, new UUID)
   - If engine process dies → exit (stops lease keepalive, keys auto-expire)

5. TEARDOWN
   - Revoke lease (immediate key deletion, faster than TTL expiry)
   - Kill engine process
   - Exit container
```

### Worker Lifecycle

```
1. STARTUP
   - Generate unique UUID
   - Block until leaders/{group_id} exists — read leader hash
   - Create etcd lease (TTL=5s)
   - Put groups/{group_id}/{hash}/rank-{N} = uuid (attached to lease)
   - Start lease keepalive in background

2. WAIT FOR START (barrier 2)
   - Poll for groups/{group_id}/{hash}/start = "go"
   - ALSO poll leaders/{group_id} — if hash changes or key disappears → exit
   - This prevents deadlock if leader dies before sending "go"

3. ENGINE START
   - Start the engine process (headless mode)

4. MONITORING
   - Poll leaders/{group_id} every 1s
   - If leader key DELETED → exit (leader died, lease expired)
   - If leader key VALUE CHANGED → exit (leader restarted, new hash)
   - If engine process dies → exit (stops lease keepalive, keys auto-expire)

5. TEARDOWN
   - Revoke lease
   - Kill engine process
   - Exit container
```

### Why Two Barriers

**Barrier 1 (formation):** Leader waits for all workers to register. Without this,
the leader would start monitoring before workers join and might send "go" to an
incomplete group.

**Barrier 2 (start signal):** Workers wait for leader's "go" before starting their
engine. Without this, a worker might start `torch.distributed.init_process_group`
and connect to a TCP store that belongs to a leader that has since restarted
(different hash). The "go" signal confirms the leader that the worker acked is
the same one running the TCP store.

## Failure Scenarios

### Worker Dies During Active Serving

```
T=0s    Worker process dies
        Lease keepalive stops
T=3-5s  Lease expires, worker's rank key deleted from etcd
T=3-5s  Leader polling detects rank key gone → exits
T=3-5s  Leader lease expires (keepalive stopped)
T=6-8s  Other workers detect leader key gone → exit
```

Detection: ~3-5s (one TTL). Full cascade: ~6-8s (two TTLs for non-adjacent workers).

### Leader Dies During Active Serving

```
T=0s    Leader process dies
        Lease keepalive stops
T=3-5s  Lease expires, leader key + "go" key deleted from etcd
T=3-5s  All workers polling detect leader key gone → exit
```

Detection: ~3-5s (one TTL).

### Leader Restarts Quickly

```
T=0s    Leader dies
T=0.5s  New leader starts, generates new hash, overwrites leaders/{group_id}
T=0.5s  Workers' next poll: leader key VALUE CHANGED → exit immediately
```

Detection: ~0.5-1.5s (next poll cycle, no TTL wait needed).

### Worker Restarts Quickly

```
T=0s    Worker dies
T=0.5s  New worker starts, reads same leader hash, overwrites rank key with new UUID
T=0.5s  Leader's next poll: rank key VALUE CHANGED → exit immediately
```

Detection: ~0.5-1.5s (next poll cycle).

### Leader Dies Before "Go" Signal

```
T=0s    Leader dies during formation (before sending "go")
T=3-5s  Leader key expires
T=3-5s  Workers in barrier 2: detect leader key gone → exit
```

Workers don't hang forever waiting for "go" because they also watch the leader key
during the wait.

### Worker Never Joins

```
T=0s    Leader waiting in formation for workers
T=Ns    FORMATION_TIMEOUT reached → leader exits
```

Bounded by the formation timeout (configurable, default 120s).

### Restart Storm Prevention

```
T=0s    Worker dies → leader detects → leader exits
T=1s    K8s restarts leader container (new hash)
T=2s    K8s restarts worker container
T=2s    Worker blocks, waiting for leaders/{group_id}
T=3s    New leader publishes new hash
T=3s    Worker reads hash, registers under it
T=5s    Leader sees all ranks → sends "go"
T=6s    Both running in new group
```

No storm because: worker waits for leader (doesn't act on stale state), leader
doesn't monitor until all ranks have joined (formation barrier). The barriers
create a clean synchronization point.

## Why etcd Leases

etcd leases provide the key primitive: a TTL handle where ALL keys attached to
the lease are deleted atomically when the lease expires. This means:

- **No manual TTL refresh loops per key.** One keepalive stream per member,
  regardless of how many keys it holds.
- **Atomic cleanup.** When the leader dies, its hash key AND the "go" key both
  disappear together. No window where "go" exists but the leader doesn't.
- **Already in infrastructure.** Dynamo deployments include etcd for discovery.
  etcdctl is in the runtime image.

## Implementation

The harness is implemented as shell wrapper scripts:
- `harness/harness_leader.sh` — wraps the leader engine command
- `harness/harness_worker.sh` — wraps the worker engine command

Usage:
```bash
# Leader
ENGINE_ID=0 NNODES=2 bash harness_leader.sh python3 -m dynamo.vllm --model ... --gms-mode shadow

# Worker
ENGINE_ID=0 NODE_RANK=1 bash harness_worker.sh python3 -m dynamo.vllm --model ... --headless
```

The operator would inject these wrappers as the container entrypoint, prepending
the harness before the actual engine command.

## Validated Performance

Tested locally with TTL=3s:

| Scenario | Detection | Total Propagation |
|----------|-----------|-------------------|
| Leader dies → worker exits | 2.7s | 3.0s |
| Worker dies → leader exits | 3.2s | 4.0s |
| Leader dies before "go" | — | 3.0s |
| Leader restarts (hash change) | — | 1.5s |
| Worker restarts (UUID change) | — | 1.5s |

Hash/UUID changes are detected faster (1.5s) because the new value is written
immediately rather than waiting for TTL expiry.

## Tuning

- **LEASE_TTL:** Controls detection latency. Lower = faster detection, higher risk
  of false positives from transient etcd connectivity issues. Default: 5s.
- **Poll interval:** Leader and worker poll every 1s. Could use etcd watch for
  event-driven detection instead of polling (sub-second).
- **FORMATION_TIMEOUT:** How long the leader waits for workers. Should be long
  enough for model loading + CUDA graph capture. Default: 120s.

## Future Improvements

- Replace polling with `etcdctl watch` for event-driven detection (<100ms latency).
  Current implementation polls every 1s which adds up to 1s to detection time.
  `etcdctl watch` would fire immediately on key change/deletion, reducing detection
  to just the lease TTL for deaths or near-instant for hash/UUID changes.
- Add canary generation on the leader for deep health checking (GPU hang detection)
- Combine with K8s liveness probes as a backup detection mechanism
- Consider NATS KV as an alternative to etcd for deployments without etcd
