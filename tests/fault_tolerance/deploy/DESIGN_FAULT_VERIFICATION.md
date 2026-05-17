# Verifying that a fault actually took effect

Today every fault event "fires" the action (signal sent, pod deleted, iptables rule installed) and trusts the kernel/k8s to do the rest. The `RstInjection`-with-`tcpkill` regression (silent no-op for two whole test runs because tcpkill wasn't in the netshoot image) is the proof we can't trust without verifying.

This doc maps each fault event to a *post-execute verification check* that confirms the precondition for the cascade actually happened on the cluster.

## Common verification primitives we'll need

1. **`/proc/<pid>/stat` state byte**: 3rd field, single char. `R` (running), `S` (sleeping), `D` (uninterruptible), `T` (stopped — what SIGSTOP produces), `Z` (zombie), missing → process gone. Cheap, exec'd inside the target pod.
2. **`containerStatuses[*].restartCount`**: integer, read from pod manifest. Increments on kubelet-driven restart (SIGKILL → kubelet restart, OOMKill, liveness fail). Take a snapshot pre-fault, compare post-fault.
3. **`containerStatuses[*].lastState.terminated.reason`**: string like `"OOMKilled"`, `"Error"`, `"Completed"`. Distinguishes *why* a restart happened.
4. **`kubectl get events --field-selector involvedObject.name=<pod>`**: kube emits events for `Killing`, `Started`, `BackOff`, `OOMKilling`, `Unhealthy`. Querying these by pod name gives a timeline.
5. **Prometheus 1-Hz scrape** (already in `server_metrics_export.jsonl`): for a given metric + label set, query the window `[fault_start, fault_start + duration]`. Flat region = stall happened; spike/decline = restart happened; etc.
6. **iptables-rules dump** (RstInjection only): the helper pod's stdout, grepped for the `RULE-INSTALLED`/`RULE-REMOVED` sentinels we now emit.

The verification block should write a `fault_verification.txt` under `ctx.log_dir/` summarising what each fault event observed, and raise if anything didn't happen.

## Per-scenario verification

### StallProcess (SIGSTOP → SIGCONT)

**What we send**: SIGSTOP to one process inside one or more pods; sleep `duration`; SIGCONT.

**What "took effect" means**: the targeted PID was in state `T` for the full `duration` window. Partner ranks in the same pod remained in `R`/`S`.

**How to verify**:
- Just after sending SIGSTOP, exec `cat /proc/<pid>/stat` in the target pod, parse field 3, assert `T`.
- (Optional) Halfway through `duration`, repeat — process should still be `T`.
- Just before sending SIGCONT, repeat once more.
- After SIGCONT + 2s grace, repeat — assert state is `R`, `S`, or `D` (not `T`, not missing).
- Cross-check from Prometheus: for `vllm:num_requests_running{pod=<target>}` over the stall window, assert the value is constant (no progression). For decode pods this works cleanly; for prefill pods at very low load the value can be 0 already and not move — fall back to /proc check there.

**False positives we want to catch**:
- We hit the wrong PID (parent launcher instead of a rank → our recent EngineCore/Worker naming fix).
- SIGSTOP went to a process that was already exiting (race) → state `Z` or missing.

### TerminateProcess (SIGKILL / SIGINT)

**What we send**: SIGKILL (framework demotes to SIGINT for Python) to one process inside one or more pods.

**What "took effect" means**: the targeted PID is gone within ~5s. For SIGKILL/SIGINT of pid 1, the container terminates and kubelet restarts it; restartCount increments. For SIGKILL of a TP rank subprocess (not pid 1), the parent EngineCore detects the loss and exits, which again increments the container's restartCount.

**How to verify**:
- Snapshot the pod's `restartCount` before fault inject.
- After 30s, re-read the pod manifest.
- Assert `restartCount_after >= restartCount_before + 1`.
- Read `lastState.terminated.reason` and `exitCode`: log them. `Error` or non-zero exit is expected; `OOMKilled` would surprise us here.
- Cross-check from kube events: filter `reason in (Killing, Started, BackOff)` for the pod name, assert at least one `Killing` within the fault window.

**False positives we want to catch**:
- We hit nothing (process_name regex too tight → no matches → silent no-op).
- We hit a child the kubelet doesn't see (some sidecar) → restartCount doesn't move.

### DeletePod (k8s delete)

**What we do**: `kubectl delete pod <name> --grace-period=0 --force`.

**What "took effect" means**: the named pod is gone within ~10s; the DGD operator schedules a replacement; the new pod (different name) appears in the same service within ~60s.

**How to verify**:
- Snapshot the pod name list for the service pre-delete.
- Verify the deleted pod's UID does NOT appear in the service after 10s (or appears with `DeletionTimestamp` set).
- After 60s, assert a pod in the service has a name not in the pre-delete list (i.e. a fresh pod scheduled).
- Cross-check from kube events: `Killing` and `Started` for the same service within the fault window.

**False positives we want to catch**:
- Delete returned 200 but pod's controller is broken and never reschedules → 60s timeout passes with no replacement.
- We targeted a pod that was already terminating.

### RstInjection (iptables FORWARD REJECT --reject-with tcp-reset)

Already implemented in the rewrite that landed today:

- `_wait_for_running` blocks until the helper pod reaches `Running`/`Succeeded`; raises within 60s if it doesn't.
- Helper pod's stdout is captured via `read_namespaced_pod_log` and grep'd for `RULE-INSTALLED` + `RULE-REMOVED`.
- `rst_injection_verification.txt` is written per scenario.
- If iptables installation failed on any helper, the event raises.

**Additional checks worth adding** (next iteration):
- Cross-check from the *frontend* side: increment in `dynamo_frontend_requests_total{status="error",error_type="cancelled"}` or similar during the injection window — confirms the FE actually observed connection failures.
- Decrement in `dynamo_component_inflight_requests{pod=<target>}` during the window.

### GpuMemoryHog (planned for T5b)

**What we do**: schedule a sidecar pod on the same node as the target, allocating ~40 GB of VRAM via a tiny CUDA program for `duration` seconds.

**What "took effect" means**: the target pod's GPU sees less FB available; vLLM either preempts heavily or its KV allocator errors.

**How to verify**:
- Snapshot `DCGM_FI_DEV_FB_FREE` per GPU on the target pod's node pre-fault.
- During the fault, assert at least one GPU on that node lost ≥ 30 GB of free FB.
- Helper pod's stdout: print `ALLOCATED <bytes>` and `RELEASED <bytes>` sentinels; grep for them.
- Failure mode: helper pod ImagePullBackOff'd or didn't get a GPU → no FB drop → raise.

### Cross-cutting hardening

- **Every fault event** gains a `verify(ctx)` method called immediately after `execute(ctx)` (or built into `execute`'s tail). Verification failure raises `RuntimeError` and bubbles up to the test, which then fails — visible in `test.log.txt` and the pytest report.
- **Pre-fault snapshot** is taken by a common helper at the start of `execute(ctx)`: pod IPs, names, container restart counts, key metric values. The verification block compares against this baseline.
- **Append to `fault_verification.txt`** so a single file per test summarises every fault's pre/post state. Trivially diff-able between runs.

## Implementation order

1. **Already done**: `RstInjection` verify (wait-for-Running + iptables stdout capture + sentinel check).
2. **Next** (~30 min): `TerminateProcess.verify` — restartCount delta + lastState.terminated.reason snapshot. Small, high-value.
3. **Next** (~30 min): `StallProcess.verify` — `/proc/<pid>/stat` state check inside the pod.
4. **Next** (~20 min): `DeletePod.verify` — new-pod-name appears, old-pod-uid gone.
5. **Defer**: `GpuMemoryHog` (whole event TBD as part of T5b).

After each one lands, re-run the corresponding scenario at c=18 and confirm `fault_verification.txt` contains an explicit "verified" line. Two passes:
- A control where we deliberately break the targeting (wrong process_name) — should fail loudly, not silently.
- The real scenario — should verify and pass.

## Existing test_outputs

The `test_outputs/test_n3_fault_scenario[18-decode_rst_inject]/` and `[18-prefill_rst_inject]/` from today's run are NOT trustworthy — tcpkill was silently missing. Mark them in the SUMMARY when we re-build the report, and re-run after the iptables-based fix lands. The remaining 9 scenarios (stalls, kills, pod_delete) MIGHT have applied — but we don't have hard evidence, so they should be re-verified via this design too. A second pass with verification baked in is the right next step.
