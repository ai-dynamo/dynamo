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
    DYN_GMS_GPU_QUIESCENCE_TIMEOUT_SECS=1.0
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
         |<-------- one-shot pipe FD |                              |
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
         |                           | verify PID absent from list   |
         |                           |----------------------------->|
         |<------------------ SIGKILL|                              |
         X                           | cache exact-cohort proof      |
                                     | successor may reclaim HBM     |

The handler does not call Python, CUDA, malloc, logging, locks, or RPC. It uses
one fixed-size write smaller than PIPE_BUF, SIGSTOP, and pause. Live testing
shows that MPS can return `CUDA_ERROR_MPS_RPC_FAILURE` while the whole client is
group-stopped. GMS therefore waits until `/proc` proves the exact birth-checked
PID is stopped, resumes it solely for the MPS termination protocol, and does not
authorize the successor during that interval. If MPS fails, GMS birth-checks
and stops the client again. GMS accepts only CUDA result 0 from
terminate_client and then verifies that the PID disappeared from the MPS client
inventory. A missing client, command exit status 0 with CUDA result 201,
traffic cessation, CPU-process death, and time alone are not proof.

One rank's notification fences its exact registered cohort. For tensor
parallelism, every CUDA rank must be registered with the GMS/MPS authority for
its physical GPU. Recovery remains gated until every affected rank supplies a
proof; a partial multi-rank result is retryable but is never promoted to a
whole-cohort proof.

## Deployment requirements

- GMS and its engine workers must share a PID namespace. GMS validates
  /proc/<pid>/stat at registration and again before signaling, which also
  prevents PID-reuse mistakes.
- All CUDA clients and the control process must use the same MPS pipe directory
  and compatible UID. The MPS daemon must start before any CUDA context.
- Exactly one MPS server must be discoverable, or its PID must be configured.
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
- The alternate signal stack is intentionally process-lifetime memory.
- If pipe delivery itself fails, the process still stops. This is fail-closed
  for KV correctness but requires an operator to kill/restart the process.
- GMS briefly resumes a stopped client because current MPS needs it runnable to
  complete `terminate_client`. The shadow remains fenced throughout. This can
  drain already-submitted CUDA work and adds crash-path latency, but it adds no
  serving-hot-path synchronization.
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
   Since it is absent from the serving hot path, statistically significant
   steady-state regressions are a bug, not an accepted cost.

Use a supervisor to detect and kill a process left in the stopped state after a
failed proof, then restart the engine through the cold path. Do not send SIGCONT
to such a process yourself: only GMS performs the birth-checked, fenced resume
immediately around `terminate_client` and re-stops it if proof fails.
