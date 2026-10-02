# GMS MPS crash interlock

The crash interlock is an optional, best-effort bridge between a catchable
engine-process crash and GMS's MPS quiescence provider. It reduces the unsafe
window in which a failed CUDA process can still submit work against persistent
shared HBM. It does not turn process death, a heartbeat timeout, or an elapsed
delay into proof of GPU quiescence.

It is disabled by default. Enable it only when the deployment satisfies every
requirement below:

    DYN_GMS_GPU_QUIESCENCE_PROVIDER=gms-mps
    DYN_GMS_GPU_CRASH_INTERLOCK=1
    DYN_GMS_GPU_QUIESCENCE_TIMEOUT_SECS=2.0
    # Optional when exactly one MPS server is visible:
    DYN_GMS_MPS_SERVER_PID=<server-pid>
    # Optional when it is not on PATH:
    DYN_GMS_MPS_CONTROL_BINARY=/path/to/nvidia-cuda-mps-control

Backend-specific variables such as DYN_VLLM_GMS_GPU_CRASH_INTERLOCK override
the common setting.

## Optional process-death fallback for capacity reclamation

The default `DYN_GMS_FAILOVER_RECLAIM_POLICY=gpu-proof` retains quarantine
when MPS fails. A deployment may explicitly accept a weaker guarantee:

    DYN_GMS_FAILOVER_RECLAIM_POLICY=process-death-timeout
    DYN_GMS_FAILOVER_PROCESS_DEATH_GRACE_SECS=2

After the bounded MPS attempt fails, the successor checks the predecessor's
immutable, retired writer-cohort guard, waits the configured interval, and
checks it again. Missing or unretired cohorts and live guard holders never
authorize reclamation. Cancellation during the wait preserves quarantine.
Phase two still reclaims only quarantine stamped by this recovery owner;
it does not free SEALED read-pinned generations or classify current-writer pages.
Serving from safe SEALED/FREE capacity continues while reclamation runs.

This fallback also applies when MPS returns CUDA201 or times out. **It is not
CUDA quiescence proof:** the MPS server can outlive its client and retain GPU
work beyond the interval. Late accesses could corrupt reused memory. Repeated
passing tests provide empirical confidence for the tested driver/workloads,
not a universal bound. Status is `reclaimed-best-effort`, with
`gpu_quiesced=False`; it must not be reported as qualified MPS termination.
Use the default strict policy when that residual risk is unacceptable.

The cohort-death checks add little beyond the CPU fence: admission already
waited for the predecessor's exclusive lifetime guard, so every registered
writer process had exited before phase one. What the policy actually relies
on is elapsed time since that fence (the bounded proof attempt plus the grace)
being long enough for a surviving MPS server or driver to drain the dead
client's work.

Both settings are validated when frozen-predecessor mode starts, so a typo
fails the boot instead of silently stranding capacity. Probe refusals caused
by contention between TP ranks or I/O errors, and phase-two attempts blocked
by live successor lease mutations, are retried with capped backoff; process
death is monotonic, so none of them is a terminal refusal. The per-rank status
file stays `pending` until a terminal outcome: `reclaimed`,
`reclaimed-best-effort`, or `quarantined` (strict policy without proof).
Status files are named `<backend>-<role>-engine<ENGINE_ID>-<pid>.json`, so
primary and shadow containers sharing one directory cannot overwrite each other.

**Repeated failover.** Phase two only reclaims quarantine stamped by its own
recovery owner. If a successor dies before its phase two, its stamped
quarantine would otherwise never become allocatable again. Under this policy
the next successor's phase one re-stamps that quarantine to itself, clearing
read pins left by the dead, fenced cohort, and reclaims it with its own pages.
The older cohort was fenced before the dead successor was admitted, so it has
been dead for at least that successor's lifetime. The strict policy does not
inherit: a GPU proof covers only the immediate predecessor.

## Protocol

    engine worker             persistent GMS                 MPS control
         | register(pid, cohort, rank, crash_interlock=true)       |
         |-------------------------->|                              |
         |<--- crash-notification FD |                              |
         | initialize CUDA / join MPS                               |
         | install native handlers  |                              |
         |                          ... normal serving ...          |
         | catchable fatal signal    |                              |
         | pipe.write(fixed record)  |                              |
         | SIGSTOP self              |                              |
         |                           | verify exact PID is stopped  |
         |<------------------ SIGCONT|                              |
         |-------------------------->| terminate_client(server,pid) |
         |                           |----------------------------->|
         |                           |<---------- CUDA result 0 -----|
         |<------------------ SIGKILL|                              |
         X                           | verify PID absent from list   |
                                     |----------------------------->|
                                     | cache exact-cohort proof      |
                                     | successor may reclaim HBM     |

The handler does not call Python, CUDA, malloc, logging, locks, or RPC. It uses
one fixed-size write smaller than PIPE_BUF and then pauses its own thread; GMS
may birth-check and SIGSTOP the host process during the termination protocol.
Live testing shows that MPS can return `CUDA_ERROR_MPS_RPC_FAILURE` while the whole client is
group-stopped. GMS therefore waits until `/proc` proves the exact birth-checked
PID is stopped, resumes it solely for the MPS termination protocol, and does not
authorize the successor during that interval. If MPS fails, GMS birth-checks
and stops the client again. GMS accepts only CUDA result 0 from
terminate_client. That result authorizes a birth-checked SIGKILL of a stopped
crash client; GMS then verifies that the PID disappeared from the MPS client
inventory before recording proof. This ordering is required because a client
paused in its native signal handler cannot service the final teardown needed
to retire its MPS inventory entry. A missing client, command exit status 0 with CUDA result 201,
traffic cessation, CPU-process death, and time alone are not proof.

One rank's notification fences its exact registered cohort. For tensor
parallelism, every CUDA rank must be registered with the GMS/MPS authority for
its physical GPU. Reclaiming predecessor-writable pages remains gated until
every affected rank supplies proof; a partial multi-rank result is retryable
but is never promoted to a whole-cohort proof.

### Frozen-generation early serving

`DYN_GMS_FAILOVER_FROZEN_PREDECESSOR=1` is an opt-in availability mode for a
fully initialized, mapped vLLM or SGLang standby. It requires KV leases, an
authoritative content directory with a compatible manifest, a read-only
standby directory before takeover, and a configured GPU-quiescence provider.
Under the default `gpu-proof` policy it does not weaken the MPS proof required
for reclaiming predecessor-writable HBM; the opt-in process-death fallback
above is the only exception.

After the global active-writer lock changes hands, the successor first waits
for all predecessor **CPU writer guards** on every TP rank. The leader then
promotes and classifies its local directory/lease ring; in vLLM each remote
worker does the same for its own pod-local GMS directory and `/dev/shm` ring.
The all-rank RPC must complete before the mapped shadow resumes generation:

| Lease record | Before GPU proof |
|---|---|
| Exact directory-backed `SEALED` generation with no predecessor readers | Readable and eligible for ownership adoption |
| Exact directory-backed `SEALED` generation with predecessor readers | Frozen read-only cache hit; successor pins are tracked, but adoption and writable reuse wait for GPU proof and pin release |
| Completed `IDLE` or already `FREE` page | Available for a new successor allocation |
| Mutable or interrupted transition page | Quarantined: neither read nor allocate; a transition is discarded after GPU proof rather than treated as valid KV |

The shadow may serve using only the first two categories. MPS proof runs in
the background. Under the strict policy, only a positive capability result
reclaims quarantine, and failed or timed-out proof leaves it quarantined.
Background attempts are bounded by `DYN_GMS_FAILOVER_BACKGROUND_GPU_PROOF_SECS`
(default 2 seconds); changing that timeout does not change the safety boundary.

SGLang publishes boot-scoped markers for the allocator: `.gms-reclaimed` after
phase two under either policy, `.gpu-quiesced` only with strict proof, and
`.gms-reclaim-refused` when the strict policy terminally keeps quarantine. A
promoted shadow with no writable page waits, bounded by
`GMS_SGLANG_HANDOFF_CAPACITY_WAIT_SECS` (default 30), for phase two to settle
before failing.
Frozen-page reclamation rejects `gms-mps-inventory` even when the separate
best-effort survivor-retirement mode is enabled: inventory absence cannot
prove completion of previously submitted CUDA work.

On a peer-rank crash, a healthy rank's CUDA client may still be registered
with MPS. Both vLLM and SGLang make one bounded exact-client termination
attempt before killing that survivor's host processes; this is the last
reliable opportunity for MPS to certify it. A failed attempt does not prevent
early sealed/free serving, but leaves that rank's GPU-touched pages
quarantined. Frozen mode defaults MPS control to 0.4 seconds for SGLang and
1.5 seconds for vLLM; the backend-specific
`DYN_VLLM_GMS_GPU_QUIESCENCE_TIMEOUT_SECS` or
`DYN_SGLANG_GMS_GPU_QUIESCENCE_TIMEOUT_SECS` can override this
latency-versus-capacity tradeoff. The crashed rank and any remaining
quarantine are retried asynchronously.

This is a best-effort provider, not a guarantee that all predecessor HBM can
be reclaimed after an arbitrary crash. In local TP=2 vLLM tests, MPS sometimes
returned CUDA_SUCCESS only after 0.4–1.3 seconds; once the control call timed
out, later calls returned CUDA 201 because the client had exited. A 1.0-second
budget failed the strict all-rank reclamation test, while 1.5 seconds passed
one rank-0 and one rank-1 TP=2 trial; this is not a reliability guarantee.
GMS must retain quarantine unless it receives a positive result. Deployments
that require recovery under full-pool pressure need a qualified alternative
GPU-quiescence mechanism or spare capacity on an independent replica.

`DYN_GMS_FAILOVER_FROZEN_HEADROOM_BLOCKS` reserves 16 anonymous FREE blocks
per lease ring by default. Neither primary nor standby prewarm may consume the
reserve. Once the successor has fenced CPU writers and classified the ring, it
releases that reserve for early decode. For a near-full pool, launch and warm
the standby before admitting primary traffic: a reserve established after the
pool is already full cannot manufacture free pages. Size this reserve for the
expected number of concurrent replay streams and the interval until GPU proof
or capacity recovery; exhaustion is explicit backpressure, not permission to
reuse quarantined pages.

This mode depends on the engine's seal contract: a block is published `SEALED`
only after its KV writes complete, and every later write or eviction must
atomically leave `SEALED` before touching the bytes. A software bug that writes
through a stale pointer into a sealed block violates that contract; neither
process death nor this mode can make such a write safe. A second crash before
the replacement standby has armed its own headroom can reduce availability,
but must still fail closed for KV correctness.

### Proactive TP teardown

The native handler covers a CUDA worker that receives a catchable fatal signal.
Multi-node engines also need to retire the other members of the now-broken TP
cohort. Their liveness callbacks request exact-cohort quiescence while the local
CUDA client is still visible to MPS. That RPC may set
`terminate_host=true`: GMS first requires CUDA result 0 from
`terminate_client`, then birth-checks and kills only the registered client PID,
and finally verifies MPS inventory absence. Ordinary successor recovery never
sets this flag.

vLLM headless ranks use a narrow `MultiprocExecutor` subclass. On the exact
worker-process sentinel, it stops rank heartbeats and fail-stops the launcher
instead of spending the takeover interval in upstream graceful executor cleanup.
Normal shutdown remains on the upstream implementation. SGLang uses its existing
child watchdog: it obtains the same local proof before fencing scheduler children.

These actions run in parallel across GPUs. Heartbeats are failure suspicion, not
the reuse proof. Writer-cohort retirement plus each local GMS/MPS result remains
the proof boundary.

Heartbeats run in a helper process (`DYN_GMS_RANK_LIVENESS_ISOLATED=1`, the
default), so GIL stalls in the engine process cannot delay them. The helper
exits, and the rank is then fenced, if its parent stops ticking for
`DYN_GMS_RANK_LIVENESS_PARENT_HANG_MS` (default 10000). A missed heartbeat
deadline (`DYN_GMS_RANK_LIVENESS_TIMEOUT_MS`, default 750) is still a false
failover. It fences a healthy cohort and starts the standby. Crash detection
does not depend on this deadline: a crash reaches survivors through the crash
interlock and child sentinels in a few hundred milliseconds. So size the deadline
for the worst scheduling delay of the helper, not for detection speed. In
CPU-limited containers, CFS throttling can stall every thread of a container
for a full period while a co-located standby compiles. At TP16 with 4-CPU
engine containers, one helper went 354 ms without sending a heartbeat. Use at
least 1000 ms there, or give engine containers enough CPU to avoid throttling.

### Opt-in TP survivor retirement

Some MPS/driver combinations do not safely support calling
`terminate_client` concurrently on every healthy member of a broken TP cohort
while an initialized warm shadow shares the same MPS servers. TP=8 B200
qualification observed both multi-second command latency and a fatal GPU error
propagated into the warm shadow. Deployments may explicitly choose a weaker,
best-effort path for the healthy survivor ranks:

    DYN_GMS_ALLOW_SURVIVOR_INVENTORY_RETIREMENT=1
    # Default is 300 ms; vLLM-specific setting takes precedence.
    DYN_GMS_INVENTORY_RETIREMENT_SETTLE_MS=300
    DYN_VLLM_GMS_INVENTORY_RETIREMENT_SETTLE_MS=300

The rank that actually receives the catchable fatal signal still requires
`terminate_client` to return CUDA result 0. On the other ranks, GMS
birth-checks and kills the old host CUDA worker, waits until the MPS server no
longer lists that PID, and labels the result `gms-mps-inventory`. The successor
preserves this weaker label when aggregating local predecessor proofs and waits
the settle interval once before remapping shared KV.

Inventory disappearance proves that the old client can no longer submit work.
It does **not** prove that driver-side teardown of already submitted work has
completed. The settle interval closes an empirically observed teardown race; it
is not a CUDA capability proof. Strict deployments must leave this mode
disabled and fail closed when every rank cannot obtain CUDA result 0. The
best-effort mode must be requalified for each GPU, driver, MPS version, TP
degree, and workload combination.

## Deployment requirements

- GMS and its engine workers must share a PID namespace. Linux pidfd support
  is required for host signaling. GMS opens a pidfd, checks the process birth
  identity, then signals that handle so PID reuse cannot redirect a host signal.
- All CUDA clients and the control process must use the same MPS pipe directory
  and compatible UID. The MPS daemon must start before any CUDA context.
- The owning Unix user is the trust boundary. Any process that can connect to
  the mode-0600 GMS socket is trusted to register only its own CUDA client PID.
- Exactly one MPS server must be discoverable, or its PID must be configured.
  Retry state is scoped to that server PID; after a server restart, GMS requires
  a fresh termination result rather than reusing partial proof from its predecessor.
- The engine must arm the handler only after CUDA joined MPS. vLLM arms after
  Worker.init_device; SGLang arms after its GMS-backed allocators exist.
- A cohort identifier is single-use. Late registration after recovery begins
  is rejected.
- GMS must remain alive and able to run MPS control. If proof fails, the crashed
  process intentionally remains stopped and shared HBM remains quarantined.
- `terminate_client` can return a nonzero CUDA result (observed: 201 or 806)
  during a crash, even when the host later exits or its MPS inventory entry
  disappears. Neither observation upgrades that attempt to proof. A later
  attempt may succeed against the same server; if a native crash was already
  recorded, GMS re-freezes that exact process before retrying rather than
  sending a second fatal signal.
- If `terminate_client` already returned CUDA result 0 but inventory retirement
  timed out, GMS retains that per-client, per-server-PID proof for a retry. It
  still checks PID birth identity and inventory absence, but does not signal a
  host process that may have exited after the successful CUDA termination.

## Failure matrix and limitations

| Failure | Intercepted? | Warm shared-HBM recovery |
|---|---:|---|
| SIGSEGV, SIGABRT, SIGBUS, SIGILL, SIGFPE | Yes, after arm | Yes only after strict MPS proof |
| SIGTERM | No; reserved for graceful engine/orchestrator shutdown | Existing graceful or cold path |
| Normal Python exit or caught exception | No signal handler; pipe EOF only | Usually fail closed/cold |
| Uncaught Python exception | Runtime-dependent; often normal teardown | Fail closed unless it becomes a handled fatal signal with strict proof |
| SIGKILL | No handler | Warm only if the pipe-EOF attempt still obtains strict MPS proof; otherwise fail closed/cold |
| cgroup or kernel OOM kill | Usually no (SIGKILL) | Fail closed/cold |
| Host or GMS crash | No | Fail closed/cold |
| MPS control/server failure | Signal is caught | No proof; process stays stopped |
| Fatal GPU/Xid or poisoned MPS server | Not reliably recoverable | Restart MPS/GPU workload |
| Crash before the handler is armed | No | Existing cold/fail-closed path |

The table describes strict **writable** HBM recovery. Frozen-generation mode
may serve exact sealed generations and safe free pages before that proof; it
does not make any quarantined page writable or allocatable.

Additional gotchas:

- CUDA documents MPS client termination as an administrative recovery tool,
  not a universal context-adoption API. A fatal MPS or GPU fault can affect
  every client sharing that server.
- Other libraries may replace these signal handlers after installation. There
  is no portable ownership protocol for fatal-signal handlers.
- SIGTERM is deliberately not intercepted. Treating an orchestrator's normal
  termination as a fatal CUDA crash can stop an otherwise healthy worker and
  hang graceful shutdown. Qualification injects SIGABRT for the catchable
  crash path and tests SIGKILL separately for fail-closed behavior.
- Forking after CUDA/interlock initialization is unsupported. A fork child
  refuses to report using its parent's registration and exits instead.
- The alternate signal stack is intentionally process-lifetime memory and is
  installed only on the calling thread. It does not guarantee recovery from
  stack exhaustion on every CUDA or framework thread.
- MPS control still accepts a numeric PID, not a pidfd or context generation.
  Birth checks prevent known identity mismatches, but cannot atomically bind
  the MPS RPC to that identity if an external SIGKILL and PID reuse intervene.
  Eliminating that remaining MPS identity race requires driver API support.
- If pipe delivery itself fails, the process still stops. This is fail-closed
  for KV correctness but requires an operator to kill/restart the process.
- GMS briefly resumes a stopped client because current MPS needs it runnable to
  complete `terminate_client`. Other threads in that process can run again in
  this interval; the handler only parks the faulting thread. Predecessor KV
  reuse remains gated on proof. This can
  drain already-submitted CUDA work and adds crash-path latency, but it adds no
  serving-hot-path synchronization.
- Local-rank recovery, successor requests, and failure re-stopping share one
  daemon lock. Only the reporting rank must have entered the native handler;
  other registered ranks are terminated through MPS without waiting for them
  to crash. Cross-node proof still requires the authority on each GPU. GMS
  never signals an arbitrary process supplied by the caller: registration,
  cohort identity, PID birth identity, and MPS server identity must all match.
- In strict mode, empty MPS inventory is accepted only after CUDA termination
  success. The explicit survivor-retirement mode records inventory absence as
  weaker, best-effort evidence and never relabels it as strict proof. Both
  modes require a successful, fully parsed inventory command; diagnostics and
  truncated lists cannot serve as absence evidence. Commands target MPS v2;
  MPS v3 requires a separately qualified command adapter.
- The interlock cannot preserve in-flight request execution. It protects
  committed/sealed shared KV; the router still replays the interrupted request.
- MPS teardown is a crash-path operation. It adds registration and handler
  installation at startup, but no request-, token-, allocation-, or lease-path
  synchronization during steady-state serving.

## Operational guidance

Keep standby preparation out of the serving window. With snapshot restore, a
standby resumes quickly and does not compete with the primary. Without it,
set `DYN_GMS_FAILOVER_SERVE_AFTER_STANDBY=1`. The initial primary then
finishes engine initialization but registers for traffic only once a standby
reports armed. The standby needs the primary's weights and KV geometry, so it
cannot finish first. Start the standby without waiting for the primary to
become discoverable; it waits for the published KV geometry itself (raise
`DYN_VLLM_GMS_SHADOW_INIT_GEOMETRY_WAIT_MS` for large models). An orchestrator
that starts the standby only after the primary serves deadlocks against this
gate until `DYN_GMS_FAILOVER_STANDBY_GATE_SECS` expires.

Exercise both modes in qualification:

1. A catchable crash such as SIGABRT must show the native notification, CUDA
   result 0, client-inventory disappearance, cohort proof, and warm promotion.
2. A SIGKILL test must demonstrate fail-closed behavior. It must never pass
   merely because the PID or MPS inventory entry vanished.
3. For TP, fault each rank independently and verify that partial proofs do not
   release another rank's quarantined pages.
4. Compare stabilized TTFT, ITL, and throughput with the interlock disabled.
   The handler adds no serving-hot-path work, but enabling MPS itself can change
   GPU scheduling and resource use. Compare no-MPS, MPS without the interlock,
   and MPS with the interlock before making a performance-equivalence claim.

Use a supervisor to detect and kill a process left in the stopped state after a
failed proof, then restart the engine through the cold path. Do not send SIGCONT
to such a process yourself: only GMS performs the birth-checked, fenced resume
immediately around `terminate_client` and re-stops it if proof fails.
