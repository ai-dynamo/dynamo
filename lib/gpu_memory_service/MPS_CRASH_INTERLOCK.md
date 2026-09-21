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
one fixed-size write smaller than PIPE_BUF, SIGSTOP, and pause. Live testing
shows that MPS can return `CUDA_ERROR_MPS_RPC_FAILURE` while the whole client is
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
its physical GPU. Recovery remains gated until every affected rank supplies a
proof; a partial multi-rank result is retryable but is never promoted to a
whole-cohort proof.

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
