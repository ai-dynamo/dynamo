# DEP-XXXX Plugin Architecture — Agent Handoff

> **For any AI agent picking up DEP-XXXX implementation work**
> Last updated: 2026-04-22 (after PR 1/2/3 v1/4/5 all coded; PR 6 6-1 + 6-5 foundations)

This is the single-page entry for resuming work. Read it top-to-bottom
once, then dive into the specific PR you're going to implement.

---

## TL;DR — Where we are

| Item | Status |
|---|---|
| **Main DEP doc** | `DEP-XXXX_Dynamo_Planner_Plugin_Architecture_zh.md` v11 (2546 lines) |
| **Implementation plan** | `DEP-XXXX_Implementation_Breakdown_zh.md` v2 (cross-check report v1+v2) |
| **8 PR detailed docs** | `DEP-XXXX_PR{1..8}_Detailed_zh.md` — all revised to align with v11 |
| **PR 1 (Proto + Type)** | ✅ **CODED** (v2.0); 27 round-trip tests pass |
| **PR 2 (Transport + Clock)** | ✅ **CODED** (v2.0); 88 tests pass (incl. 50 contract tests) |
| **PR 3 (Registry + Scheduler)** | ✅ **v1 CODED** (v2.0); 86 tests pass; K8s SA / SPIFFE deferred to PR 3.5 |
| **PR 4 (Merge algorithms)** | ✅ **CODED** (v2.0); 65 tests pass (basic/constrain/short_circuit/worked_examples/chain_augment/9 hypothesis) |
| **PR 5 (Orchestrator)** | ✅ **CODED** (v2.1); all 10 sub-tasks; 6/6 G3 fixtures replay byte-identical through orchestrator |
| **PR 6 / 7 / 8** | 📄 docs v1.x ready; not coded |
| **Pre-PR 5 fixture lock** | ✅ Code + test in repo; **awaiting `git tag pre-plugin-architecture` from human** |

**Test count baseline**: `python -m pytest dynamo/planner/tests/plugins -q` → **394 passed in ~6s**
(121 PR1+PR2 + 65 PR 4 + 86 PR 3 + 39 PR 5 + 83 PR 6 partial: lifecycle 7 + reconcile 7 + load_predictor 12 + throughput_propose 8 + load_propose 11 + budget_constrain 10 + install_regressions+bootstrap_plugins 10 + g3 parity 10 × 3 tests = 30)

---

## How to resume work

### Step 1 — Verify clean baseline (always do first)

```bash
cd /root/kang/dynamo

# 1. Verify proto stubs match committed (no drift)
tools/build/gen_planner_proto.sh --check
# expect: "OK: generated stubs match committed files (no drift)"

# 2. Run all DEP-XXXX tests
cd components/src && python -m pytest dynamo/planner/tests/plugins -q
# expect: "394 passed"

# 3. Run full planner-test marker scope (CI parity)
python -m pytest dynamo/planner/tests/ -m "pre_merge and planner and gpu_0" -q \
  --ignore=dynamo/planner/tests/unit/test_prometheus.py \
  --ignore=dynamo/planner/tests/unit/test_diagnostics_recorder.py
# expect: "690 passed, 1 skipped, 11 deselected" (or similar)
```

If anything fails, **stop and investigate** before adding new code — you
might be on a stale branch.

### Step 2 — Read the right PR's detailed doc

Each PR has a top-level "修订历史" (revision history) section listing
every decision/footnote made during planning **and** implementation:

```bash
# Read PR 4 (next to code) detailed doc
head -200 DEP-XXXX_PR4_Detailed_zh.md
```

Coded PRs (PR 1, PR 2) have a `v2.0 实施完成` section in their revision
history that documents implementation deviations from the original plan.
**Read these before assuming the code matches the plan exactly.**

### Step 3 — Pick the next sub-task

**PR 5 is fully CODED** (v2.1 — all 10 sub-tasks). See
`DEP-XXXX_PR5_Detailed_zh.md` `修订历史 v2.0` + `v2.1` for details.
Key decisions:

- **Option B for 5-7/5-8**: single `PsmShimProposePlugin` + `PSMBridge`
  rather than 5-way split (original wording was infeasible — PSM
  tick-level bookkeeping like `_reset_diag` / `_build_diagnostics` /
  `_next_scheduled_tick` doesn't decompose across plugins). Real 5-way
  split is PR 6's scope with its own builtin state.
- **Baseline policy in placeholder layer**: the parity test passes
  `{}` so PSM's `scale_to=None` maps to M-4 `skip_no_targets` instead
  of a spurious baseline passthrough. PR 6 builtins can interpret
  `baseline` per-plugin.
- **TickInput / PlannerEffects bridging**: implemented in the parity
  test (`_tick_input_to_context` / `_outcome_to_effects`); PR 7
  NativePlannerBase can reuse those two functions directly.

**Next candidates** (all unblocked now):

- **PR 6** — 5 real decomposed builtins; replaces the single shim
  plugin. Preserves G3 parity by construction (same merged output).
- **PR 8** — observability / replay harness; PSM-independent; can
  proceed in parallel with PR 6.
- **PR 7** — NativePlannerBase dual-path + feature flag; needs PR 6
  done first so production cutover actually gains something.

**Pending housekeeping**:

- Add `hypothesis>=6.0` to `pyproject.toml` dev/test extras (PR 4 Q4).
- PR 3.5 follow-up: implement `K8sSATokenAuth` + `SpiffeJwtAuth` against
  the schemas already defined in `registry/config.py`.
- `DEP-XXXX_PR5_5-7_5-8_Notes.md` can now be deleted or kept as a
  postmortem reference — its contract extraction is all captured in
  the `修订历史 v2.1` section.

If you want to do something else, the dependency graph is:

```
PR 1 (proto)        ✅ coded
   └─→ PR 2 (transport)  ✅ coded
   └─→ PR 4 (merge)       ← next; pure functions, no PR 2/3 dep
   └─→ PR 3 (registry + scheduler)  needs PR 1 + PR 2
PR 5 (orchestrator)  needs PR 1+2+3+4
PR 6 (5 builtins)    needs PR 5
PR 7 (NativePlannerBase dual-path)  needs PR 5+6
PR 8 (observability + replay)        needs PR 5+6
```

### Step 4 — Code, test, sync docs

For each sub-task:

1. **Read the sub-task table** in the PR detailed doc (each sub-task has
   位置 / 接口 / 实现要点 / 单测 / 依赖 / 估算 columns)
2. **Implement** following dynamo conventions (next section)
3. **Test**: add tests to `components/src/dynamo/planner/tests/plugins/<area>/`
   with markers `[pre_merge, planner, gpu_0, unit]`
4. **Verify** all 121+ tests still pass
5. **If reality diverges from the plan** (extra edge case, rename, etc.),
   add an entry to that PR's `修订历史 → v2.x 实施完成` section. This is
   how the next agent learns about your decisions.

---

## Dynamo conventions you MUST know

### Python tooling

- **Pydantic v2** (already used in `components/src/dynamo/planner/connectors/protocol.py`)
- **grpcio-tools ≤ 1.76** pinned in `container/deps/requirements.common.txt` (NO buf)
- **Pytest markers** auto-discover into `planner-test` CI job:
  `pytest.mark.{pre_merge, planner, gpu_0, unit}` — all four required for
  CI to pick up the test file
- **Type checker**: `basedpyright` runs in IDE; `google.protobuf.*` import
  warnings can be ignored (existing dynamo code already does)

### File layout

```
components/src/dynamo/planner/
├── plugins/                          ← all DEP-XXXX new code
│   ├── proto/v1/
│   │   ├── plugin.proto              ← single source of truth
│   │   ├── plugin_pb2.py             ← generated, check-in
│   │   ├── plugin_pb2_grpc.py        ← generated, check-in
│   │   └── plugin_pb2.pyi            ← generated, check-in
│   ├── types.py                       ← Pydantic v2 mirror
│   ├── _proto_bridge.py               ← Pydantic↔proto converter
│   ├── transport/
│   │   ├── base.py                    ← PluginTransport ABC
│   │   ├── in_process.py / uds.py / grpc_remote.py
│   │   ├── _grpc_base.py              ← uds + grpc shared mixin
│   │   ├── _method_dispatch.py        ← method name → stub map
│   │   ├── _mtls.py                   ← MtlsConfig
│   │   ├── config.py                  ← TransportConfig + factory
│   │   └── README.md                  ← Threat Model + decision tree
│   ├── clock.py                       ← Clock ABC + Wall + Virtual
│   ├── merge/                         ← (PR 4 will create)
│   ├── registry/                      ← (PR 3 will create)
│   └── orchestrator/                  ← (PR 5 will create)
├── tests/plugins/
│   ├── proto/test_round_trip.py       ← 27 cases
│   ├── transport/                     ← 79 cases (incl. 50-case contract test)
│   ├── clock/                         ← 9 cases
│   └── g3_fixtures/                   ← Pre-PR 5 fixture lock + CI guard
└── core/                              ← READ-ONLY (until PR 11 cleanup)
```

### Build commands

```bash
# Regenerate proto stubs (after editing plugin.proto)
tools/build/gen_planner_proto.sh

# CI mode: regenerate to temp dir + diff vs committed
tools/build/gen_planner_proto.sh --check
```

### CI integration

The existing `.github/workflows/pr.yaml` `planner-test` job picks up
test files via markers automatically. **You do NOT need to edit any
.yml** to add new tests.

### Forbidden patterns (will be caught by lint in PR 5 5-9)

- `time.time()` / `time.monotonic()` / `asyncio.sleep` directly in
  production code → use injected `Clock` (PR 2 `clock.py`)
- `asyncio.wait_for(asyncio.gather(...))` to wrap a stage's plugin
  calls → per-plugin `request_timeout_seconds` already handles it
  (M-7 OR option, see PR 2 README)
- bare `except:` catching transport errors → must catch
  `PluginCallError` subclass (PR 2 `transport/errors.py`)

---

## Hidden knowledge (NOT obvious from reading code/docs)

These are footnotes from PR 1 + PR 2 implementation that the next agent
might rediscover painfully:

### PR 1 `_proto_bridge.py` 4 edge cases

When converting between proto and Pydantic mirror:

1. **`MessageToDict` enum default** is the string name (`"AT_LEAST"`) but
   Pydantic IntEnum expects int → must pass `use_integers_for_enums=True`
2. **Pydantic `mode="json"` UTF-8-decodes bytes** which fails on
   binary FPM payloads (e.g. `\xff`) → use `mode="python"` then base64-encode
   bytes manually
3. **proto map fields** appear in `ListFields()` as Python dicts (not
   list); the descriptor recurse loop must skip them
4. **`RepeatedCompositeContainer`** has no `__iter__` attribute but
   supports `iter()` — use `list(iter(value))` for repeated message walking,
   not `hasattr(value, "__iter__")`

### PR 2 `VirtualClock` cancellation cleanup

`asyncio.sleep(N)` → `Future` parked in heap. If task is cancelled before
deadline, the future stays in heap. `VirtualClock.advance()` does a
cleanup pass: pops cancelled futures and silently discards them. This is
v11 P1-4 review fix; without it long-running tests have unbounded heap growth.

### PR 2 sync plugin red line

`InProcessTransport` dispatches sync plugin methods via `asyncio.to_thread`,
which uses the default 32-thread pool. **A few sync plugins doing blocking
IO (HTTP, file, `time.sleep > 100ms`) will exhaust the pool and stall the
orchestrator.** Documented in `transport/README.md`. PR 7 production config
should cap with `executor_max_workers ≤ 8`.

### PR 4 chain-augment final reverse-priority quirk

In PR 4 chain-augment (PREDICT stage), the chain is sorted **priority-descending**
(largest priority number first; smallest = highest priority runs last).
`final=true` triggers chain break at the first occurrence — meaning the
**lowest-priority** plugin's `final=true` will short-circuit higher-priority
plugins. PR 4 implements **runtime detection** + WARNING log + Prometheus
counter for misuse. PR 4 README has a "强制契约" subsection explaining
that `final=true` plugin in PREDICT MUST be configured with the lowest
priority number.

### CONSTRAIN SET runtime drop (NOT register-time reject)

v10 main doc said "register-time static rejection of CONSTRAIN plugins
that emit SET" but this is **infeasible** — proto3 has no plugin-self-
declared output-type metadata. v11 changed it to **runtime drop + audit**:
when a CONSTRAIN plugin returns an `OverrideResult` with `OverrideType.SET`,
the orchestrator (PR 5) silently drops that target entry, emits audit
event `plugin_constrain_set_dropped`, and increments Prometheus counter
`plugin_constrain_set_dropped_total{plugin_id}`. PR 4 4-2 type_aware_merge
implements the drop logic; PR 5 5-4 emits the audit/metric.

### `result` oneof empty = plugin contract violation

If a Propose/Reconcile/Constrain plugin returns a response where
`WhichOneof('result')` is `None` (i.e. forgot to set `accept` /
`override` / `reject`), the orchestrator (PR 5) MUST raise
`PluginSerializationError` and trigger circuit breaker. **Do NOT
silently treat empty oneof as ACCEPT** — that masks plugin bugs.

### REJECT > final priority

If any plugin in the same stage returns `RejectResult`, the entire
stage short-circuits — even when `final=true` plugins are also present.
This matches K8s admission controller `deny > allow` semantics.

---

## Pending human actions (Agent cannot do)

These are blocked on human:

1. **`git tag pre-plugin-architecture`** to lock the G3 fixture as the
   golden reference. Recommended commands:
   ```bash
   cd /root/kang/dynamo
   git add components/src/dynamo/planner/tests/plugins/
   git commit -m "test(planner): lock G3 behavior parity fixtures pre-plugin-architecture"
   git tag -a pre-plugin-architecture -m "Locked fixtures + CI guard for plugin architecture refactor (DEP-XXXX)"
   ```

2. **Commit PR 1 + PR 2 + DEP docs** as separate commits so the timeline
   is reviewable:
   ```bash
   # PR 1
   git add components/src/dynamo/planner/plugins/{__init__.py,proto/,types.py,_proto_bridge.py}
   git add components/src/dynamo/planner/tests/plugins/proto/
   git add tools/build/gen_planner_proto.sh
   git add DEP-XXXX_PR1_Detailed_zh.md
   git commit -m "feat(planner): DEP-XXXX PR 1 — proto + Pydantic mirror + round-trip tests"

   # PR 2
   git add components/src/dynamo/planner/plugins/{transport/,clock.py}
   git add components/src/dynamo/planner/tests/plugins/{transport/,clock/}
   git add DEP-XXXX_PR2_Detailed_zh.md
   git commit -m "feat(planner): DEP-XXXX PR 2 — transport + clock + 50-case contract test"

   # Docs
   git add DEP-XXXX_*.md
   git commit -m "docs(planner): DEP-XXXX v11 main doc + 8 PR detailed breakdown + handoff"
   ```

3. **Open GitHub PR / DEP issue** (skill: `dep-create`); maintainer review.

---

## Pointers — where to find what

| Question | Look at |
|---|---|
| What's the overall design? | `DEP-XXXX_Dynamo_Planner_Plugin_Architecture_zh.md` v11 |
| What 8 PRs and their dependencies? | `DEP-XXXX_Implementation_Breakdown_zh.md` (top has Mermaid graph) |
| What did each PR's planning decide? | `DEP-XXXX_PR{1..8}_Detailed_zh.md` (sub-task tables) |
| What deviations during implementation? | Each PR doc's `修订历史 → v2.x 实施完成` section |
| What v11 review found? | `DEP-XXXX_Implementation_Breakdown_zh.md` "PR 1-4 Review 残留问题" + main doc v11 changelog |
| What 8 implementation footnotes (M-1~M-8)? | Main doc § "v11 实施细节注解" (line ~2429) |
| Which Mr-N is落实 in PR 1+2? | This HANDOFF's "Hidden knowledge" + each PR doc revision history |
| What did Pre-PR 5 fixture lock do? | `components/src/dynamo/planner/tests/plugins/g3_fixtures/README.md` |
| How does the new code integrate with existing PSM? | It doesn't — PR 5/6/7/8 全程 read-only PSM; PR 11 cleanup deletes PSM |
| Module-local conventions (forbidden patterns / Pydantic v2 / markers) | `components/src/dynamo/planner/plugins/CLAUDE.md` |

---

## Active tasks (last session's todos)

```
[partial]      PR 6 (10/10 logically: 6-1..6-7 + 6-8 partial 10/36 + 6-9 bootstrap + 6-11 e2e) — 2026-04-23
[completed]    PR 5 全部 10 sub-task — 2026-04-22
[completed]    PR 3 v1 minimal (9 sub-task; K8s SA + SPIFFE 推 PR 3.5) — 2026-04-22
[completed]    PR 4 全部 6 sub-task — 2026-04-22
[completed]    PR 1 全部 9 sub-task
[completed]    PR 2 全部 8 sub-task
[completed]    Pre-PR 5 fixture lock
[completed]    主文档 v10 → v11 修订
[completed]    8 PR 详细文档全部就绪 (PR 1/2/3/4/5/6 v2.x; others v1.0/1.1)

[partial]      PR 7 (10/11: 7-1/2/3/4 + 7-7 predicted_load + 7-8 parity + 7-10 runbook
               + 7-5/7-6 design-intent-satisfied via _apply_effects preserve-and-extend) — 2026-04-23
               dev/staging READY; prod canary READY including global-planner env

[completed]    PR 6 6-7 finish — PsmShimProposePlugin + PSMBridge + test_g3_orchestrator_parity.py
               deleted 2026-04-23; internal_register.py was already generic.
               G3 coverage now: test_g3_real_parity (10) + test_engine_adapter (14) +
               test_dual_path_parity (30) — all via production OrchestratorEngineAdapter.
[completed]    hypothesis>=6.0,<7.0 declared in container/deps/requirements.test.txt
               (PR 4 Q4); property tests in tests/plugins/merge stay gated via
               pytest.importorskip and pass with 9 cases.
[completed]    PR 3.5 follow-up — K8sSATokenAuth + SpiffeJwtAuth landed 2026-04-23:
               sync K8s TokenReview via asyncio.to_thread + PyJWT/PyJWKClient.
               build_auth_validator now wires both; NotImplementedError branches
               replaced with ValueError("sub-config missing"). 23 unit tests; real
               SPIRE / kind integration test is follow-up (same infra block as 7-9).

[next]         PR 7 7-9 mode e2e (BLOCKED — needs full runtime wiring infrastructure;
                 user on hold until that infra exists; same block as PR 3.5 integration tests)
               → PR 6 6-8 remaining (10→36 scenarios; mechanical — testing work, deprioritised
                 per user "不急着测试" guidance; optional coverage expansion)
               (no other open sub-tasks — PR 7 7-5/7-6 closed as design-intent-satisfied in v2.3)
```

If you're finishing PR 6 (the PSM-algorithm-port builtins):

```bash
cd /root/kang/dynamo
# 1. Read DEP-XXXX_PR6_Detailed_zh.md — 修订历史 v2.0 documents what's
#    already done (6-1 lifecycle + 6-5 reconcile passthrough).
# 2. Per builtin (6-2/6-3/6-4/6-6) — read the PSM mixin source listed
#    in the spec's "算法搬运来源" cell, port to a plugin that extends
#    BuiltinPluginBase, unit test against PSM mixin output byte-for-byte.
# 3. Register each real builtin in test_g3_orchestrator_parity.py
#    AS YOU GO, verifying 6/6 stays green. When all five are in, remove
#    the PsmShimProposePlugin + PSMBridge.
# 4. 6-8 G3 sweep: expand fixtures from 6 → 36 scenarios (scenarios.py +
#    dump_tool regenerate + new golden files under pre-plugin-architecture
#    tag). This is a substantial sub-task on its own.
```

---

## Skills available in this repo

```
.claude/skills/
├── dep-create/        # File a new DEP issue on GitHub
├── dep-status/        # Check DEP statuses
├── dep-update/        # Update existing DEP fields
├── dynamo-docs/       # Maintain Fern docs site
├── gh-issue-bug/      # File a bug
└── tool-parser-generator/   # Generate tool-call parsers
```

For DEP-XXXX work, none of these are directly needed except possibly
`dep-create` when the human is ready to file the upstream issue.
