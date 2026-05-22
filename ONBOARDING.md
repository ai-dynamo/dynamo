# Onboarding: KVBM Remote Search — run the smoke & understand it

Welcome. Your task: **manually bring up the remote-search smoke test and confirm it passes**,
then understand what it proves. This doc gets you from a fresh checkout to a green run plus a
rendered trace, then explains how the feature works and points at the code.

---

## 0. What the feature is (one paragraph)

A KVBM instance, on a cold-cache request, asks the **hub's KV indexer** which *other* instance
already holds the prompt's KV blocks, opens a **transfer session** on that holder, and
**RDMA-pulls** the blocks into its own local G2 (host) cache — instead of recomputing them. It is
the engine leader driving an out-of-band pull during `get_num_new_matched_tokens` (GNMT). The
request **stalls** on the search/pull, then prefills the uncached tail + decodes normally.

The smoke stands up **two plain aggregated vLLM instances (A, B)** + a **hub running `indexer` +
`p2p`**. Warm A (it computes + indexes its blocks). Then hit B cold with the same prompt — B
remote-searches, finds A, pulls A's blocks, and decodes. We assert the transfer happened and that
B's output equals A's.

---

## 1. Prerequisites

- A Linux box with **one NVIDIA GPU** and the CUDA toolkit at `/usr/local/cuda`.
  The model is tiny (Qwen3-0.6B); any modern datacenter GPU works. Two instances share **one** GPU.
- This repo checked out. All paths below are relative to the worktree root
  (`.../dynamo/.claude/worktrees/hub` here; adjust to your checkout).
- `git`, `cargo` (Rust toolchain), Python, `uv`, `curl`. `patchelf` optional (a warning only).

---

## 2. One-time setup: the `.sandbox` venv

The smoke drives raw `vllm.entrypoints.openai.api_server` from an isolated venv at `./.sandbox`
(torch + cu13x, a vLLM nightly, nixl). If `./.sandbox/bin/python3` already exists, skip this.

Otherwise build it (this is the slow part, ~15–30 min):

```bash
# From the worktree root. This is a user-invocable skill:
/dynamo:kvbm:sandbox-venv --fresh
```

> ⚠️ **Never point `KVBM_VENV` at another worktree's `.sandbox`** — `maturin develop` would
> overwrite that venv's `kvbm` `.so` with this branch's build.

---

## 3. Build the artifacts

Two things must be built from *this branch's* Rust:

**a) The connector Python extension (`.so`)** — this is what vLLM loads. Rebuild it after any Rust
change to `kvbm-connector` / `kvbm-engine` / `kvbm-config`:

```bash
source .sandbox/bin/activate
export CUDA_PATH=/usr/local/cuda CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH KVBM_REQUIRE_CUDA=1
( cd lib/bindings/kvbm && maturin develop --release --features v2 )
# maturin's pip step rolls nccl back to satisfy vllm's pin; re-bump it:
uv pip install --force-reinstall --no-deps 'nvidia-nccl-cu13>=2.29'
```

Look for `🛠 Installed kvbm-1.2.0`. (A `patchelf` rpath warning is harmless — `LD_LIBRARY_PATH`
in the launcher handles nixl.) The user-invocable shortcut is `/dynamo:kvbm:maturin-dev` (note:
this branch's bindings only have the `v2` feature, **not** `v1`).

**b) `kvbm_hub` + `kvbmctl`** — built automatically by the smoke's hub launcher, but you can
prebuild: `cargo build --bin kvbm_hub --bin kvbmctl`.

---

## 4. Run the smoke

```bash
# Spark / GB10 (default profile): just run it.
bash .claude/skills/remote-search-smoke/remote-search-smoke.sh

# Any OTHER single GPU (H100/A100/L40/…): use the h100-a100 profile for the model/sizing,
# but LOWER the per-instance GPU memory so TWO 0.6B instances fit on ONE GPU:
KVBM_HARDWARE_PROFILE=h100-a100 KVBM_GPU_MEMORY_UTILIZATION=0.25 \
  bash .claude/skills/remote-search-smoke/remote-search-smoke.sh
```

Why the override: the `h100-a100` profile defaults to `0.70` GPU memory util *per instance* — two
of those on one GPU OOMs. `0.25` is plenty for Qwen3-0.6B ×2. (The default `spark-gb10` profile is
already `0.15`.) Profiles live in `.claude/skills/disagg-bringup/hardware-profiles.sh`
(valid: `spark-gb10`, `h100-a100`, `custom`).

Useful knobs: `KVBM_GPU_MEMORY_UTILIZATION`, `KVBM_VLLM_READY_TIMEOUT` (default 300s),
`RUST_LOG` (defaults to `info,kvbm_connector=debug,kvbm_engine=debug,kvbm_audit=info`).

The run takes a few minutes (model load dominates). Logs + the trace land in a fresh
`/tmp/kvbm-experiments/<ts>-remote-search/` dir (printed as `EXP=…`).

---

## 5. What success looks like

Last lines should read `remote-search smoke PASSED`. The script prints B's per-request **timeline**
and renders `trace.html`. A passing timeline on B looks like:

```
gnmt_pending           request="cmpl-…"            ← request STALLS: a find/search is in flight
transfer_pull_started  session=… source=<A's id>   ← RDMA pull from A begins
  …repeated gnmt_pending while vLLM re-polls…        ← the stall, visibly waiting
transfer_pull_completed pulled=3                    ← 3 blocks landed in B's local G2
gnmt_matched           matched_tokens=48 async_load=true  ← stall resolves to an external match
onboard_start          num_external_tokens=48       ← G2→G1 onboard begins
onboard_complete       ok=true                      ← KV in GPU; request runnable
request_finished                                    ← remaining prefill + decode done
```

The script also asserts:
- **A skipped remote search on its warm (2nd) request** — `transfer_pull_started` count on A is 0
  (A holds the blocks locally; it must not pull, least of all from itself).
- **B's decode == A's golden decode** — same prompt, greedy (`temperature=0`), so identical text
  proves the *pulled* KV is correct, not just present.

Open the trace:
```bash
xdg-open /tmp/kvbm-experiments/<ts>-remote-search/trace.html   # 3 lanes: A | Hub | B
```
Click a `request_id` in the sidebar to filter to B's request and read the timeline top-to-bottom.

---

## 6. How it works, and where the trace shows each step

The whole flow happens inside **one `get_num_new_matched_tokens` (GNMT) request lifecycle** on B.
The leader runs in vLLM's EngineCore subprocess; its `kvbm_audit` events surface in the instance
log (hence `kvbm_audit=info` in `RUST_LOG`).

| Step | What happens | Trace marker (lane) | Code |
|---|---|---|---|
| 1. Cold match | vLLM calls GNMT. Local G2 match is empty → there are remote blocks to find. | `gnmt_pending` (B) | `mod.rs:811` GNMT; `search.rs:225` `process_match` |
| 2. Search spawned | The engine's `find_matches_with_options` spawns the remote-search driver and returns an async session; GNMT returns `(None,false)` → vLLM **re-polls** (the stall). | (driver starts) | `instance.rs:2204` / `:2267` `use_indexer_search` |
| 3. Discover | Driver asks the hub indexer who holds the missing hashes; filters out its own id; resolves the candidate peer. | (hub lane RPC) | `remote_search.rs:107` `search`; `:132` self-filter; connector impl `remote_search.rs:38`; hub `client.rs:59` `find_blocks` |
| 4. Open + pull | Driver `open_session` on A (holder pins its G2 prefix), then `pull_from_session` RDMA-copies into B's local G2 and registers the blocks, then `close_session`. | `transfer_session_opened` (A), `transfer_pull_started` / `transfer_pull_completed` (B) | `remote_search.rs:161` `pull_from`; engine `transfer.rs:364` `open_transfer_session`, `:512` `pull_from_session` |
| 5. Match resolves | Driver re-matches local G2 (now warm) and sets the session `Complete`; next GNMT poll returns `(Some(n), true)`. | `gnmt_matched matched_tokens=…` (B) | `mod.rs:879` |
| 6. Onboard | `update_state_after_alloc` copies the matched G2 blocks → G1 (GPU). | `onboard_start` / `onboard_complete` (B) | `onboard.rs:385` |
| 7. Compute | Request runs: prefill the uncached tail + decode. Ends at `request_finished`. | `request_finished` (B) | `finish.rs:29` |

**Key design points to internalize:**

- **The request stalls; it does not recompute cold.** `(None,false)` from GNMT means "find still
  running" — vLLM re-polls each scheduler step. The repeated `gnmt_pending` rows bracketing the
  pull are that stall. Resolution is `gnmt_matched`.
- **Engine owns the state; the connector is thin.** The connector only injects a *discovery*
  implementation; the search/pull/cancel state machine lives in the engine. Because the engine
  registers the session, request cancellation (`release_session`) cleanly tears down an in-flight
  pull (`instance.rs:1241`).
- **Warm requests skip remote search entirely.** If the synchronous local G2 match already covers
  the whole prefix (`local_covers_all`, `instance.rs:2264`), the engine returns `Ready` — no driver,
  no discovery, no self-pull. That's why A's 2nd request shows zero pulls.
- **Threshold.** A search is issued only when remaining full remote blocks `>= min_remote_blocks`.
  `remote_search.min_remote_tokens = None` ⇒ "any remote match" (≥1 block); `Some(n)` ⇒ `⌈n/bs⌉`
  blocks. `remote_search.rs` config `min_remote_blocks` at `remote_search.rs:54`.
- **Feature requirements.** Remote search needs the hub to offer **both** `indexer` (discovery)
  and `p2p` (the transfer plane). Enforced at startup: `hub_handshake.rs:71`
  `validate_remote_search_availability`.

---

## 7. Config: how remote search is turned on

It's a connector-side `KvbmConfig` field, rendered into vLLM's `--kv-transfer-config` by `kvbmctl`
from the live hub. In the smoke it's seeded into the hub's base config
(`start-hub.sh`, `KVBM_HUB_KVBM` → `leader.remote_search.enabled=true`) so every connector inherits
it. The rendered blob carries `"leader": { …, "remote_search": { "enabled": true }, "hub": {
"features": ["indexer","p2p"] } }`. Optional tuning: `leader.remote_search.min_remote_tokens=<N>`.

Struct: `lib/kvbm-config/src/remote_search.rs:30`.

---

## 8. Code map (for deeper review)

**Smoke harness** — `.claude/skills/remote-search-smoke/`
- `start-hub.sh` — hub with `KVBM_HUB_FEATURES=indexer,p2p` + remote_search seeded.
- `launch-instance.sh` — one aggregated vLLM + KVBM v2 connector; renders config via `kvbmctl`.
- `remote-search-smoke.sh` — the driver + assertions + trace render.
- Trace renderer: `.claude/skills/disagg-trace/p2p-trace.py`.

**Config** — `lib/kvbm-config/src/remote_search.rs`

**Engine (owns the state machine)**
- `lib/kvbm-engine/src/leader/discovery.rs:38` — `RemoteBlockDiscovery` trait + `RemoteCandidates`
  (the seam; engine must not depend on the hub).
- `lib/kvbm-engine/src/leader/remote_search.rs` — `RemoteSearchDriver`: `search` (`:107`),
  self-filter (`:132`), `pull_from` (`:161`), watchdog + degrade-to-local.
- `lib/kvbm-engine/src/leader/instance.rs:2204` — `find_matches_with_options`: gate
  (`use_indexer_search` `:2267`), warm short-circuit (`local_covers_all` `:2264`), driver spawn,
  cancellation (`release_session` `:1241`).
- `lib/kvbm-engine/src/leader/control/modules/transfer.rs` — the transfer control plane:
  `open_transfer_session` (`:364`, holder), `pull_from_session` (`:512`, puller→local G2).

**Connector (thin: discovery impl + injection + request-scoped audit)**
- `lib/kvbm-connector/src/connector/leader/remote_search.rs:38` — `HubRemoteDiscovery` over the
  hub `IndexerLookupClient` + `HubPeerResolver`.
- `lib/kvbm-connector/src/connector/leader/init.rs:1212` — discovery injected into the leader.
- `lib/kvbm-connector/src/connector/leader/mod.rs:811` — `get_num_new_matched_tokens` (+ `gnmt_*`
  audit `:879`).
- `lib/kvbm-connector/src/connector/leader/search.rs:225` — `process_match` / shard reconciliation.
- `lib/kvbm-connector/src/connector/leader/onboard.rs:385` — onboard audit markers.
- `lib/kvbm-connector/src/connector/leader/finish.rs:29` — `request_finished` audit.
- `lib/kvbm-connector/src/connector/leader/hub_handshake.rs:71` — feature validation.

**Hub** — `lib/kvbm-hub/src/features/indexer/client.rs:59` (`find_blocks`).

---

## 9. Troubleshooting

- **Instance never reaches `/v1/models`** → read `instance_a.log` / `instance_b.log`. Common: OOM
  (lower `KVBM_GPU_MEMORY_UTILIZATION`), or `import kvbm` failing (rebuild the `.so`, re-bump nccl).
- **`remote-search smoke` fails at "no blocks indexed"** → A's index publisher didn't wire; check
  A's log for `indexer publisher wired`. Confirm the hub serves `indexer` (`curl
  http://127.0.0.1:1337/v1/config`).
- **B never pulls (`transfer_pull_completed` 0)** → confirm both instances logged `standalone P2P
  participation registered with hub` and `remote-search discovery injected into leader`. If the
  hub lacks `p2p`, startup should have failed fast (that's the validation in §6).
- **`audit` events missing from the log** → ensure `RUST_LOG` includes `kvbm_audit=info` (the
  smoke sets this by default).
- **Stale processes / GPU busy** → the smoke kills stale `vllm`/`kvbm_hub` at start; if wedged,
  `pkill -9 -f vllm.entrypoints.openai; pkill -9 -f kvbm_hub`.

---

## 10. Quick mental model

> Indexer = "**who** has these blocks." P2P transfer plane = "**pull** them." Remote search =
> the leader wiring those two together during GNMT, stalling the request until the KV is local.
> Warm requests skip it; cancellation tears it down; both features must be on the hub.
