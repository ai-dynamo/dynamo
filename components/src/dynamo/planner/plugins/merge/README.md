# Plugin output merge algorithms (DEP-XXXX PR 4)

Two **pure-function** algorithms that the PR 5 orchestrator composes into
its pipeline:

| Stage | Algorithm | Why |
|---|---|---|
| PROPOSE / RECONCILE / CONSTRAIN | `type_aware_merge` | All plugins run in parallel; outputs have explicit SET / AT_LEAST / AT_MOST semantics and need priority-aware merging per ComponentKey. |
| PREDICT | `chain_augment` | Prediction is built up layer-by-layer; each plugin sees prior layers via `PipelineContext.predictions` and may refine a subset of fields. |

Both are deterministic (no I/O, no Clock, no global state). `type_aware_merge`
is sync; `chain_augment` is async only because it awaits plugin RPCs. The
algorithmic logic of both is fully synchronous.

---

## `type_aware_merge`

PROPOSE / RECONCILE / CONSTRAIN plugins are dispatched in parallel; their
`OverrideResult` outputs (plus priorities) are merged into a single
`ScalingProposal`.

### Algorithm (4 steps)

1. **REJECT short-circuit.** Any `RejectResult` in `plugin_results` makes
   the whole merge return `short_circuited=True`, `proposal=None`, and a
   `short_circuit_reason` naming the rejecter. This out-ranks `final`
   (v11 § G-2: "REJECT > final priority" — safety veto trumps authoritative
   override).
2. **`final` priority.** If any `OverrideResult` has `final=True`, the
   priority-smallest final wins and its targets become the proposal
   verbatim. The remaining plugins (final or not) are fully discarded.
3. **Bucket by `ComponentKey`**: `(sub_component_type, component_name)`.
4. **Per-bucket merge + clamp.** For each bucket:
   - `floor = max(AT_LEAST replicas)` (default `0` when no AT_LEAST)
   - `ceiling = min(AT_MOST replicas)` (default `+inf` when no AT_MOST)
   - `recommendation = priority-smallest SET replicas` else `baseline[key]`
     (else `0` when the key is absent from baseline too)
   - `result_replicas = max(floor, min(ceiling, recommendation))` — the
     outer `max` ensures `floor` wins when `floor > ceiling`
   - Cast to `int` for the output `ComponentTarget`

Keys that only appear in `baseline` (no plugin touches them) pass through
with their baseline replicas — downstream stages always see a complete
proposal.

### CONSTRAIN mode (`set_allowed=False`)

SET targets are silently dropped in **both** the final-path and the
bucket-merge path. Each dropped key is appended to
`MergeOutcome.set_dropped`; the PR 5 orchestrator reads this list to
emit a `plugin_constrain_set_dropped_total{plugin_id}` Prometheus
counter + audit event.

Why runtime drop rather than register-time rejection: proto3 has no way
for a plugin to self-declare its output types, so register-time static
rejection is infeasible. v11 § 4.3.2 resolves this as "drop + audit".

### Worked examples

The 9 worked-example cases (single-component SET/AT_LEAST/AT_MOST,
multi-component, hierarchical pools, final override) are the **source of
truth** for the algorithm. They live in
[`tests/plugins/merge/test_type_aware_worked_examples.py`](../../tests/plugins/merge/test_type_aware_worked_examples.py)
and mirror the table in `DEP-XXXX_Dynamo_Planner_Plugin_Architecture_zh.md`
v11 PROPOSE section verbatim. **Any edit to the main doc table MUST come
with a matching edit to that test file**, and vice versa.

### Complexity

`O(P × C)` where `P` = plugin count and `C` = total ComponentTarget
entries. At current production scale (`P ≤ 10`, `C ≤ 5`) this measures
well under 1 ms; no pre-optimisation warranted.

---

## `chain_augment`

PREDICT plugins run in a **sequential chain**, each seeing the running
partial prediction via `PipelineContext.predictions`.

### Algorithm

1. Sort the chain by `priority` **descending** (largest priority number
   first). Lowest priority number (highest precedence) therefore runs
   **last**, so its partial-merge has final say on conflicting fields.
2. Initialize `prediction = None`.
3. For each plugin in order:
   - Build `ctx = initial_context.model_copy(update={"predictions": prediction})`
   - `resp = await plugin.call("Predict", ctx)`
   - If `resp.predictions is not None`, `prediction = _partial_merge(prediction, resp.predictions)`
   - If `resp.final`, break (subsequent plugins are never called)
4. Return `ChainAugmentOutcome(prediction, final_from, degraded, misuse_warnings)`.

### Partial-merge rule

`PredictionData` uses `Optional[float]` for each prediction field — proto
`optional float` in `plugin.proto`. A concrete value (including `0.0`)
on `new` overrides the same field on `prev`; `None` on `new` preserves
`prev`. The `source` field takes `new.source` when non-empty, else
`prev.source`.

The `Optional` distinction is load-bearing: a "layered predictor" pattern
(e.g. one plugin asserts `predicted_num_req` only, another asserts
`predicted_isl` / `predicted_osl`) relies on `None`-means-unset semantics
to compose correctly.

### ACCEPT / REJECT in PREDICT

The proto v1 `PredictStageResponse` has only `predictions` / `reason` /
`final`. `predictions=None` is the only "no-opinion" signal (≈ ACCEPT);
there is no explicit REJECT message. `ChainAugmentOutcome.degraded`
therefore always evaluates to `[]` in v1. A future proto revision may
introduce an explicit reject; until then, a plugin that wants to abstain
simply returns `PredictStageResponse()` (all fields default).

---

## final semantics differ between stages

| Stage | `final=True` winner rule | User contract |
|---|---|---|
| PROPOSE / RECONCILE (`type_aware_merge`) | Priority-smallest (most precedent) final wins | Any priority may set `final`; simpler + intuitive |
| PREDICT (`chain_augment`) | **First** final encountered in the priority-descending chain wins — meaning the **lowest-precedence** final in a chain breaks it first | **Strong contract**: `final=True` PREDICT plugin MUST be configured with priority = lowest-number (= highest precedence), so it runs last in the sorted chain |
| CONSTRAIN | `final` is silently ignored (v11 § G-3) | Don't set it |

### Chain-augment `final` usage rule (v11 § P1-2 — strong contract)

**Rule.** A `final=True` PREDICT plugin MUST have `priority = the smallest
priority number in the chain` (= highest precedence, runs last).

**Why.** The chain is sorted `priority` descending, so a non-lowest-priority
`final=True` plugin runs *before* a higher-precedence plugin and breaks
the chain — the higher-precedence plugin is never consulted.

**Counter-example** (misuse):

```
Plugin A (priority=100, final=True)  → runs first, breaks chain
Plugin B (priority=5)                → NEVER called
```

The operator expected `B` (higher precedence) to win. Instead, `A`'s
output becomes the final prediction and `B`'s signal is lost.

**Correct usage**:

```
Plugin A (priority=100)               → runs first, contributes base prediction
Plugin B (priority=5, final=True)     → runs last, refines + breaks cleanly
```

**Runtime detection.** `chain_augment` compares each `final=True` plugin's
priority against the chain's minimum. On mismatch it:

- Emits a `WARNING` log with plugin_id, priority, and lowest_priority.
- Appends the message to `ChainAugmentOutcome.misuse_warnings`.

PR 5 orchestrator drains `misuse_warnings` and increments Prometheus
counter `predict_chain_final_at_non_lowest_priority_total{plugin_id}`
(PR 8 defines an alerting rule on this).

**Fix guidance.** When the alert fires, either:

- Reduce the offending plugin's `priority` to the smallest number in the
  chain (so it runs last), **or**
- Change the plugin to return `final=False` — it will still contribute
  via partial-merge, just without breaking the chain.

---

## Testing

| File | Covers |
|---|---|
| `test_type_aware_basic.py` | Basic paths: baseline passthrough, SET, AT_LEAST, AT_MOST, clamp order, multi-component, hierarchical pools, `replicas=None` skip. |
| `test_type_aware_constrain.py` | CONSTRAIN mode: SET drop, bounds merge, mixed, duplicate-SET audit. |
| `test_type_aware_short_circuit.py` | REJECT matrix + final priority matrix + final-in-CONSTRAIN. |
| `test_type_aware_worked_examples.py` | 9-case lock-step vs main doc worked-example table. |
| `test_chain_augment.py` | 4 usage patterns (replace / patch / augment / passthrough) + final break + misuse runtime detection. |
| `test_type_aware_properties.py` | Hypothesis invariants (clamp, baseline preserve, REJECT dominance, final dominance, AT_LEAST monotone, idempotency). |
| `test_chain_augment_properties.py` | Hypothesis invariants (input-order independence, final break, passthrough preservation). |

Property tests require the `hypothesis` package (dev extra per Q4 decision
in `DEP-XXXX_PR4_Detailed_zh.md`); they auto-skip via `pytest.importorskip`
when unavailable.

---

## Future work

- **Explicit REJECT in PREDICT.** Adding a reject message to
  `PredictStageResponse` in a proto revision would let `chain_augment`
  populate `ChainAugmentOutcome.degraded`, mirroring how `type_aware_merge`
  populates `set_dropped`.
- **`priority_strict_final` toggle** on chain_augment (deferred from v1):
  when set, automatically re-order `final=True` plugins to the end of
  the chain regardless of priority, silencing the misuse warning. v1
  keeps the "priority descending" rule plus contract + warning approach
  for simplicity.
