# Resume: wire kv-router → kv-consolidator → instance

## TL;DR of where you are

Worktree `/home/ryan/repos/dynamo/.claude/worktrees/kvcc` (branch
`disagg-merge-connector`). Two smokes are now live and passing on the
DGX Spark (GB10, aarch64, unified memory):

- **Conditional-disagg R1+R2 two-request smoke** (existing, untouched in
  this session) — `.claude/skills/disagg-smoke/two-request-smoke.sh`.
- **NEW: P2P G2 block transfer smoke** — `.claude/skills/p2p-smoke/p2p-smoke.sh`.
  Brings up hub + 2 same-model vLLMs, R1→A, scrapes audit hashes,
  drives the 3-call hub control-plane chain (`open_session` →
  `pull_from_session` → `close_session`), R2→B, asserts G2 hit rate.

The next task: a third smoke that proves a **kv-router** instance can
be wired to a **kv-consolidator** that's attached to a normal KVBM
instance. Treat the P2P smoke as the structural template — the new
smoke will reuse the bringup helpers and audit-driven validation
pattern.

## Box-specific gotchas (load-bearing — don't rediscover)

- **DGX Spark / GB10**, aarch64, Ubuntu 24.04. **Unified memory** — 119
  GiB total Grace+Blackwell shared pool. `nvidia-smi` reports memory
  as "Not Supported" (driver/hardware quirk).
- After OS + rust-analyzer + claude processes baseline (~25 GiB used,
  ~78 GiB in caches), **~94 GiB is free** for KVBM. Anything > 0.78
  GMU on a single vLLM overcommits.
- For two coexisting Qwen3-0.6B vLLMs use **`GMU=0.15` per instance**
  (matches `.claude/skills/disagg-bringup/launch-{prefill,decode}.sh`).
- Launch them **sequentially** — vLLM's GPU memory profiler runs at
  startup and on unified-memory both racing instances see the same
  "free", so the second errors with "No available memory for the
  cache blocks." Wait for `/v1/models` on the first before launching
  the second. The P2P smoke encodes this via `wait_for_models`.
- **Active venv**: `/home/ryan/.venvs/dynamo-kvbm` symlinks to
  `/home/ryan/repos/dynamo-workspaces/ryan-velo-messenger/.sandbox`.
  Pass `KVBM_VENV=/home/ryan/repos/dynamo-workspaces/ryan-velo-messenger/.sandbox`
  and prepend `$KVBM_VENV/bin` to `PATH` so vLLM subprocesses find
  the `vllm` binary.
- **Post-maturin nccl re-bump**: every `maturin develop` rolls
  nvidia-nccl-cu13 back to 2.28.9 to satisfy vllm's pin; torch 2.11
  needs ≥ 2.29 (`ncclDevCommDestroy`). Always run
  `uv pip install --force-reinstall --no-deps 'nvidia-nccl-cu13>=2.29'`
  after maturin. Note: `torch.cuda.nccl.version()` will still report
  the compile-time 2.28.9 even after re-bump — that's a torch caching
  artifact; the actually-loaded `libnccl.so.2` is 2.30.4 and exports
  `ncclDevCommDestroy`. See the kvbm-maturin-dev skill.

## What's committed (don't redo)

Commit `6ecf4fcfd30` on `disagg-merge-connector`:
- KVBM v2 agg determinism matrix
  (`tests/kvbm_integration/test_determinism_agg_v2_matrix.py`) — 4
  cells (G1 LW/FC × G2 Op/Uni) on Qwen3-0.6B.
- Skill `.claude/skills/kvbm-v2-agg-matrix/` (run.sh + SKILL.md).
- Connector bug fix: `self.vllm_config` → `self._vllm_config` in
  `lib/bindings/kvbm/python/kvbm/v2/vllm/schedulers/connector.py` at
  3 sites (the JSON-config FC override + the pre-existing auto-detect
  path — the latter had been silently falling back to LW
  indefinitely; this matrix caught it).
- Fixture extensions:
  `tests/kvbm_integration/fixtures/server.py` now threads
  `block_layout` + `prefer_fc` under
  `kv_connector_extra_config.default`.

## What's UNCOMMITTED (the P2P work from this session)

Staged but Ryan committs himself — DO NOT auto-commit:

```
M lib/kvbm-config/src/messenger.rs
M lib/kvbm-engine/src/runtime/builder.rs
M lib/kvbm-engine/src/leader/instance.rs
M lib/kvbm-engine/src/leader/control/core.rs
M lib/kvbm-engine/src/offload/pipeline.rs
M lib/kvbm-protocols/src/control/modules/core.rs
M lib/kvbm-connector/src/lib.rs
M lib/kvbm-connector/src/connector/leader/init.rs
M lib/kvbm-connector/src/connector/leader/disagg/leader.rs
M lib/kvbm-connector/src/connector/leader/disagg/mod.rs
M lib/bindings/kvbm/src/v2/runtime.rs
A .claude/skills/p2p-smoke/{SKILL.md,p2p-smoke.sh,launch-instance.sh,RESTART_PROMPT.md}
A .claude/skills/disagg-trace/p2p-trace.py
```

These together:

1. **Hub-as-velo-`PeerDiscovery`** wiring (the architectural answer to
   "should we use the hub as PeerDiscovery?" — yes, and it wasn't
   wired). `HubClient` already implements `velo::discovery::PeerDiscovery`
   (per kvbm-hub/CLAUDE.md); we just had to plumb it:
   - `kvbm-config::messenger::build_velo_with_discovery(Option<Arc<dyn PeerDiscovery>>)`
   - `KvbmRuntimeBuilder::with_discovery(...)`
   - `kvbm_connector::seed_leader_builder_with_hub_discovery(config, builder)`
     — builds `HubClient` from `disagg.hub_url` and feeds it in
   - Called from `bindings/kvbm/src/v2/runtime.rs::build_leader`
2. **`InstanceLeader` carries the full `Velo`** (not just
   `Messenger`) so `core/register_leader` can call
   `velo.discover_and_register_peer` instead of
   `messenger.discover_and_register_peer`. The messenger-only path
   skipped the streaming-transport registry and the next
   `attach_anchor` failed with "TCP streaming: peer <id> not
   registered". The user's exact words: "we need to call the velo.xxx
   methods not the velo.messenger.xxx". Velo is at 0.4.1.
3. **`RegisterLeaderResponse::status` → `outcome`** —
   `ControlReply<T>` is `#[serde(tag = "status")]`, and the inner
   `status` collided to produce JSON `{"status":"ok", "status":"registered",
   ...}` which the hub's HTTP→velo proxy refused to decode (duplicate
   field). The peer registration side-effect was happening; the
   caller just couldn't see the success.
4. **New audit event** `event="offload_register_complete"` in
   `lib/kvbm-engine/src/offload/pipeline.rs` at the dst-tier register
   site (`shared.dst_manager.register_block(complete)`). Carries `src`,
   `dst`, `num_blocks`, `sequence_hashes_hex` (comma-separated
   32-char hex u128s, BE). The smoke greps this from instance A's
   log to learn the ISL hashes. Generic — fires for G1→G2, G2→G3, etc.
5. **The P2P smoke + p2p-trace.py** described above.

If you want to commit, group as one or two commits (suggested split:
"engine/connector: HubClient as velo PeerDiscovery + InstanceLeader
carries Velo" + "test(kvbm): P2P G2 block transfer smoke").

## Architecture recap (for the next smoke)

```
   ┌─────────────┐       ┌─────────┐       ┌─────────────┐
   │ instance_a  │◄──────│   hub   │──────►│ instance_b  │
   │  port 8000  │ velo  │  HTTP   │ velo  │  port 8002  │
   │             │       │ + velo  │       │             │
   │  leader+v   │       │         │       │  leader+v   │
   └─────────────┘       └─────────┘       └─────────────┘
                              ▲
                              │ control plane:
                              │ /v1/instances/{id}/control/
                              │   core/register_leader
                              │   transfer/{open,pull,close}_session
                              │   transfer/search_{prefix,scatter}
                              │   dev/reset, test/register_test_blocks
                              │   metrics/snapshot
```

Key facts you'll need:

- **HTTP endpoints live on the hub**. Hub looks up the addressed
  instance's velo leader client and dispatches via velo RPC to that
  instance's leader. Each per-tool handler in
  `lib/kvbm-hub/src/features/control_plane/manager.rs` does
  `leader_client(&mgr, instance_id).<tool>().<method>(req).await`.
- **`open_session`** opens a transfer session on the holder
  (`SearchMode::Prefix`, `FindMode::Sync`), returning a
  `TransferSessionCapability { session_id, instance_id, endpoint }`
  plus `committed: Vec<SequenceHash>` for sync mode.
- **`pull_from_session`** is addressed to the puller; body carries
  `session_id`, `source_instance_id`, and the holder's `endpoint`
  (from the open response). It's a long-poll — returns when the pull
  is complete, with `pulled: Vec<SequenceHash>` and a per-tier
  `MatchBreakdown`.
- **`close_session`** is idempotent.
- Schemas in `lib/kvbm-protocols/src/control/modules/transfer.rs`.
- `SequenceHash` wire shape is **16-byte u8 array** in big-endian
  order (`u128::from_be_bytes`). The audit event emits hex; smokes
  decode hex → bytes → JSON array of u8.
- Workers do **not** need velo peer discovery for cross-instance
  data transfer — that rides NIXL/UCX. Only **leaders** need velo
  peers, for session control plane.
- The leader's `core/register_leader` is the safe primitive to
  pre-warm peer relationships before any transfer call (the P2P
  smoke does this for both directions).

## Next task — kv-router + kv-consolidator smoke

The user wants a new skill/smoke that wires a **kv-router** instance
to a **kv-consolidator** that's attached to a normal KVBM instance.

What I do NOT yet know (the next session needs to research):

1. Where does the kv-router live in the tree (`lib/kv-router` per
   workspace deps — `grep -rn "kv-router\|kv_router" lib/ | head`).
2. Where does kv-consolidator live, what does it do, and how does it
   "attach to an instance" — is it a new dynamo binary, a control
   module, or a velo-side feature?
3. What's the wire/RPC surface between router and consolidator?
4. What does the router actually need to learn from the consolidator
   to do its job (KV residency maps? tier hit rates? block hashes?)
5. What end-to-end behavior should the smoke validate? Suggested
   shape (subject to confirmation): issue a request that should be
   routed based on KV residency, observe the router picking the
   instance with the relevant blocks in G2, and assert via audit
   that the routing decision matches what the consolidator's view
   said was available.

### Recommended first steps (when you start)

```bash
cd /home/ryan/repos/dynamo/.claude/worktrees/kvcc
git status                               # confirm uncommitted state matches above
ls lib/kv-router/ lib/mocker/             # router lives here
grep -rn "consolidator\|Consolidator" lib/ 2>/dev/null | head -30
grep -rn "fn attach\|fn route\|RouteDecision\|KvCacheView" lib/kv-router/src/ | head -20
```

Then ask the user to confirm:
- (a) Topology: 1 router instance + 1 consolidator + 1 KVBM
      instance? Or N instances behind the router with one
      consolidator each? Or one consolidator aggregating across
      instances?
- (b) Whether to build directly on top of the P2P smoke (router +
      consolidator added to the same hub-mediated 2-instance setup)
      or as a separate bringup.
- (c) What signal proves the routing is correct (audit event on
      router showing the decision basis, vs hit-rate at the chosen
      instance, vs end-to-end TTFT comparison).

### Pattern to reuse from the P2P smoke

- `launch-instance.sh` (parameterized port/role) — clone for any
  per-process bringup (router, consolidator each).
- `wait_for_models port timeout` — startup gate.
- Audit-driven validation: the P2P smoke greps `kvbm_audit`
  `event="offload_register_complete"` and parses fields. For the
  router smoke, find or add an audit event (e.g.
  `event="route_decision"` carrying request_id + chosen instance +
  hit basis) and assert it.
- **Hard assertions** at the end with explicit `exit N` for each
  failure class. Codex stop-time review previously flagged the P2P
  smoke for "can report success without validating the transfer" —
  followup added `pulled >= 1`, `pulled == committed`, and `R2
  Host hit rate > 0%`. Apply the same discipline.
- `p2p-trace.py` clone if the new smoke has > 3 distinct sources
  (router, consolidator, instance, hub) — generalize the lane logic
  rather than fork again. Or keep the renderer to 3 lanes by
  collapsing instance + consolidator into one log file if they
  share a process.

### Smoke layout to propose

```
.claude/skills/kvrouter-consolidator-smoke/
├── SKILL.md
├── launch-router.sh
├── launch-consolidator.sh   # (or fold into the instance launcher if attached)
└── kvrouter-consolidator-smoke.sh
```

## Things to NOT redo

- Don't re-run the maturin develop unless you change Rust code; the
  bindings are current as of the last P2P-smoke pass.
- Don't try to set `--gpu-memory-utilization` higher than 0.65 for
  single-instance work or 0.15 for two-instance — both hit the
  Spark's unified-memory headroom and fail with "No available
  memory for the cache blocks".
- Don't use `messenger.discover_and_register_peer` directly — the
  fix landed; use `velo.discover_and_register_peer`.
- Don't re-derive the audit-emit-with-hex format — it's working;
  the new smoke can grep the same `sequence_hashes_hex` field if it
  needs to discover block hashes.
- Don't propose adding a `DiscoveryConfig::Hub { url }` variant in
  kvbm-config — the current `seed_leader_builder_with_hub_discovery`
  approach is the chosen path. Auto-derive from `disagg.hub_url`
  happens at the bindings entry point in
  `lib/bindings/kvbm/src/v2/runtime.rs::build_leader`.

## Verification before starting

```bash
# Confirm tree matches expected uncommitted set
cd /home/ryan/repos/dynamo/.claude/worktrees/kvcc
git status --short
git log -3 --oneline

# Confirm bindings have the new audit + velo wiring
/home/ryan/.venvs/dynamo-kvbm/bin/python -c "
import kvbm
print('kvbm version:', kvbm.__version__)
"

# Run the P2P smoke as a sanity check (should PASS):
export KVBM_VENV=/home/ryan/repos/dynamo-workspaces/ryan-velo-messenger/.sandbox
export PATH=$KVBM_VENV/bin:$PATH
bash .claude/skills/p2p-smoke/p2p-smoke.sh
# expect final line: "p2p-smoke PASS: pulled=24 (of 24 committed, 24 requested); R2 Host hit=100.0%"
```

If P2P smoke fails, do not start the router smoke — the regressions
are in the P2P infrastructure and need to be fixed first.

## Cluster + auth notes

- Username on computelab is `rolson` (not `ryan`); scratch is
  `/home/scratch.rolson_hw`; working SSH key is `id_rsa`. (Reference
  memory, not used in this session yet.)
- `gh` PR-write requires `env -u GH_TOKEN -u GITHUB_TOKEN gh ...`
  and the GraphQL `updatePullRequest` mutation rather than
  `gh pr edit` (which has a projectCards bug).
