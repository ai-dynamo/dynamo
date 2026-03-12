# Coordinated Restart Test Epochs

## Epoch 1: Harness with one engine (engine-0 only)

**Setup:** Leader pod engine-0 uses `harness_leader.sh` wrapper. Worker pod engine-0 uses
`harness_worker.sh` wrapper. Engine-1 containers are `sleep infinity`.

**Tests:**
- Formation: leader publishes hash, worker registers, go signal sent, engine starts
- Kill leader → worker cascades (~3s)
- Kill worker → leader cascades (~3s)

## Epoch 2: Restart convergence

**Setup:** Same as epoch 1 but with `restartPolicy: Always`.

**Tests:**
- Kill leader → worker detects hash gone → both restart → re-form → engine active again
- Kill worker → leader detects UUID gone → both restart → re-form → engine active again
- Kill leader, it restarts fast → worker detects hash change → both restart → converge

## Epoch 3: Both engines with harness

**Setup:** Both engine-0 and engine-1 use harness wrappers. `restartPolicy: Always`.

**Tests:**
- Concurrent startup: engine-1 OOMs, restarts, harness re-forms with engine-0 already active
- Both reach expected states: engine-0 active (holds flock), engine-1 standby (sleeping)
- Verify no restart epoch drift between leader/worker for either engine

## Epoch 4: Failover

**Setup:** Same as epoch 3.

**Tests:**
- Kill engine-0 leader → harness cascades engine-0 worker → GPU freed → engine-1 wakes → serves
- Verify total failover time

## Epoch 5: Operator integration

**Setup:** Operator injects harness wrapper into container entrypoints.

**Tests:**
- Deploy DGD with `multinode + failover` → pods come up → engines converge → failover works
