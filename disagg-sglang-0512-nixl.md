# Debug: disagg.sh NIXL KVReceiver Exception on SGLang 0.5.12

**Date**: 2026-05-18
**Source**: PR #9677 launch script walk — disagg.sh regression
**Status**: investigating
**Environment**: 2x NVIDIA L40S (PHB, no NVLink) on `/ephemeral/dynamo` branch `idhanani/sgl-to-0.5.12`

## Problem

Same `examples/backends/sglang/launch/disagg.sh` that PASSED on 0.5.11 fails on 0.5.12 with decode-side timeout:

```
ERROR conn.poll: Request <room> waiting_timeout
ERROR decode.pop_transferred: Decode transfer failed for request rank=0
  decode_req.req.rid='...'
  decode_req.req.bootstrap_room=<room>
  with exception NIXL KVReceiver Exception
```

Both workers come up cleanly. Prefill warmup at `bootstrap_room=0` completes. Real user requests hit prefill, prefill scheduler reports `report_prefill_stats #new-seq: 1`, but the KV transfer to decode never lands — decode trips `waiting_timeout` after ~45s and dynamo cancels the HTTP request at 60s.

## Reproduction Steps

```bash
source /ephemeral/dynamo/.venv-sgl-0.5.12/bin/activate
export SGLANG_DISABLE_CUDNN_CHECK=1
unset HF_TOKEN
cd /ephemeral/dynamo
bash examples/backends/sglang/launch/disagg.sh &
# wait for "added model" in log
curl -s --max-time 60 -X POST localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-0.6B","messages":[{"role":"user","content":"hi"}],"max_tokens":16}'
# -> 60s timeout
```

## Expected vs Actual

- **Expected**: Same behavior as 0.5.11 — prefill computes KV, hands off to decode via NIXL, decode generates tokens, HTTP response returns in <2s.
- **Actual**: Decode side times out waiting for the KV transfer. Prefill logs say request completed. NIXL transfer never lands.

## Suspect upstream commits (v0.5.11..v0.5.12 in `python/sglang/srt/disaggregation/nixl/conn.py`)

`conn.py` diff is +1030 / -264 LoC. Candidates ranked by likelihood:

1. `811d138c8` **Nixl async transfer (#23967)** — refactored the actual transfer flow (most likely)
2. `2a4d382b0` **[Disagg][NIXL] Add staging buffer support for heterogeneous TP KV transfer (#22536)** — new staging buffer logic; even for same-TP it might be on the path
3. `c7019ff33` **[NIXL][XPU] Use np.uint64 for pointer/length arrays in disaggregation KV transfer (#24188)** — pointer/length representation change
4. `22cf7d2b4` **[Fix] Handle nixlRemoteDisconnectError in NixlKVSender (#24296)** — error handling change
5. `e264b5785` **[PD] Centralize per-room cleanup in common backend (#24601)** — cleanup refactor

## Investigation Log

### 2026-05-18 13:23 — env baseline + py-spy

- Started disagg.sh, prefill+decode workers come up cleanly on `172.27.29.185`. Bootstrap HTTP server on `0.0.0.0:12345` listens fine.
- Fired request → curl hangs 90s → dynamo cancels.
- py-spy on prefill scheduler 1310997 mid-request:
  - MainThread in `event_loop_overlap_disagg_prefill` (idle loop at `prefill.py:431/458`).
  - **4× `transfer_worker` threads all idle on `queue.get()`** at `nixl/conn.py:570`.
  - `bootstrap_thread` blocked on `recv_multipart` at `conn.py:1748`.
- py-spy on decode scheduler 1310929 mid-request:
  - MainThread in `process_decode_queue → pop_transferred → poll_and_all_reduce → all_reduce`.
  - i.e. decode IS polling, prefill IS receiving bootstrap registration — but no actual NIXL transfer fires.
- `ss -tnp` shows multiple ESTABished ZMQ conns between prefill (port 48891) and decode (port 41257), some with non-zero Send-Q. Wire is up.

### 2026-05-18 13:42 — first instrumentation pass

Added `logger.warning("DBG …")` at five strategic points in `conn.py`:
1. Prefill bootstrap_thread `recv_multipart` entry
2. Decode `send_metadata` `sock.send_multipart` call
3. Prefill `add_transfer_request` enqueue
4. Prefill `transfer_worker` chunk-dequeue
5. Decode `update_transfer_status` (empty + non-empty notif batches)
6. Per-notif `tag={kv|aux|state|stg}` decode + transfer-status snapshot

End-to-end trace for one request:

```
13:53:59.769  bootstrap_thread recv first_frame=NixlMsgGuard nframes=17   ← KVArgsRegister
13:53:59.774  decode send_metadata room=1338130518044941528 aux_index=0   ← TransferInfo from decode
13:54:00.464  bootstrap_thread recv first_frame=NixlMsgGuard nframes=10   ← TransferInfo on prefill side
13:54:00.555  add_transfer_request room=… nkv=1 aux_index=1 shard=0
13:54:00.556  transfer_worker got chunk
13:54:00.640  transfer_worker DONE n_handles=2 elapsed=0.066s             ← BOTH KV + AUX transferred
13:54:00.640  decode NOTIFS counts={agent: 1}  → tag=kv  → is_done=False  (need aux still)
13:54:00.641  decode NOTIFS counts={agent: 1}  → tag=aux → is_done=True   ← unblocks
```

Sequence is the **expected** flow — works once both upstream bugs (below) are fixed.

Before either fix, the bug showed up as `n_handles=1` (kv only, aux skipped) on prefill, and `is_done=False` forever on decode.

## Root cause: two upstream SGLang regressions in commit `d7f4761a4`

Both introduced in `[PD] Refactor hybrid state transfer (#24932)` on the `v0.5.11..v0.5.12` range. Affect every dense LLM (Qwen3, LLaMA, Gemma, etc. — anything without Mamba/SWA/NSA state).

### Bug 1 — prefill skips aux RDMA write when `state_indices` is empty

**`python/sglang/srt/disaggregation/nixl/conn.py`**, `NixlKVManager.transfer_worker` (post-refactor).

```python
# BROKEN
if kv_chunk.is_last and kv_chunk.state_indices:
    dst_info = ...
    state_xfer_handles = self.maybe_send_extra(...)
    handles.extend(...)

    # aux send INSIDE the state-gated branch
    aux_notif = f"{req.room}_aux"
    aux_xfer_handle = self.send_aux(...)
    handles.append(aux_xfer_handle)
```

`kv_chunk.state_indices` is empty (`[]`) for dense models because SGLang's `state_types` is empty. The whole branch short-circuits and **aux is never sent**.

In 0.5.11 the aux send sat directly inside `if is_last:`, gated only on `is_last`.

**Fix:** split the two — gate state send on `state_indices`, gate aux send on `is_last` only:

```python
if kv_chunk.is_last:
    dst_info = self.decode_kv_args_table[req.agent_name]
    if kv_chunk.state_indices:
        state_xfer_handles = self.maybe_send_extra(...)
        handles.extend(h for h in state_xfer_handles if h is not None)
    ...
    aux_xfer_handle = self.send_aux(...)
    handles.append(aux_xfer_handle)
```

### Bug 2 — decode declares `expects_state=True` for non-stateful models

**`python/sglang/srt/disaggregation/nixl/conn.py`**, `NixlKVReceiver.send_metadata`:

```python
# BROKEN
if state_indices is not None:
    self.kv_mgr.transfer_statuses[self.bootstrap_room].expects_state = True
```

Decode receives `state_indices=[]` (empty list, not None) for dense models. `[] is not None` is True → `expects_state=True`. `is_done()` then waits forever for a state notif the prefill side never sends.

Prefill uses `if kv_chunk.state_indices:` (truthy check) — `[]` is falsy — so prefill correctly skips state. Decode and prefill disagree on whether state is in play.

**Fix:** match the prefill side's truthy check:

```python
if state_indices:
    self.kv_mgr.transfer_statuses[self.bootstrap_room].expects_state = True
```

### Combined patch (full diff `git diff python/sglang/srt/disaggregation/nixl/conn.py`)

22 lines added / 17 removed across two hunks. Both fixes preserve the staged/Mamba/SWA paths unchanged.

## Verification

Applied both fixes to `/ephemeral/sglang` (editable install). Restarted disagg.sh. Single curl request returns tokens cleanly in <2s:

```
content= <think>\nOkay, the user just said "hi" to
usage= {'prompt_tokens': 14, 'completion_tokens': 12, 'total_tokens': 26}
```

DBG trace confirms `n_handles=2` (kv + aux) on the prefill side and `is_done=True` after the aux notif on decode.

## Decision for dynamo PR #9677

Two paths considered:

A) **Wait for upstream fix; ship dynamo bump with disagg flagged broken.** Doesn't unblock the bump for anyone running disagg.

B) **Apply the patch downstream.** Two ways to land it in dynamo's released artifacts:
   - **B1**: Vendor a runtime patch inside `dynamo.sglang._compat` that monkey-patches `NixlKVManager.transfer_worker` and `NixlKVReceiver.send_metadata` on import. Fragile — the patch reaches deep into a `while`/`try` block.
   - **B2**: Ship an `sglang_runtime.Dockerfile` postinstall step that `sed`-patches the two locations in the upstream lmsysorg/sglang image. Brittle to upstream line numbers but contained.

**Recommended path: file the upstream bug + PR immediately, ship dynamo with a clear callout in the PR body, and target a `0.5.12.post1` pin if upstream cuts a patch release. If they don't, evaluate B2 once we have more disagg users hitting this.** The fix is 4 lines on the upstream side — they should accept it fast.

## Lesson — debugging hooks I should have used first

- `DYN_LOG_LEVEL=debug` on the dynamo side, paired with `--log-level debug` passed to the SGLang worker via the launch script's EXTRA_ARGS, would have surfaced the existing `logger.debug` lines in `nixl/conn.py` (bootstrap recv, register-kv-args, transfer-status, kv/aux notif decode) without any source edits. **Next disagg debug session: configure logging first, source-patch only if the log level isn't enough.**
- `examples/backends/sglang/launch/disagg.sh` currently rejects all flags except `--enable-otel` / `--unified`. Worth a tiny patch to forward unknown flags via `EXTRA_ARGS` like `agg.sh` already does — every other launch script in the dir has this and disagg.sh's stricter parser bit me here.

## Upstream artifacts to file

- Issue title: `[Bug][PD][NIXL] disagg.sh hangs on dense models (Qwen3, LLaMA, …) — state-gated aux + decode expects_state mismatch (#24932 regression)`
- Bisect: `git bisect` between v0.5.11 and v0.5.12 lands on `d7f4761a4`.
- Repro: 2× any GPU, single host, sglang 0.5.12, two `python -m sglang.launch_server` with `--disaggregation-mode {prefill,decode}` + nixl backend, fire one request.


