<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Metrics review

Review the metrics, log events, and trace spans that the proposed change adds or modifies, and
check them against the [metrics guidelines](../../.ai/metrics-guidelines.md). The guidelines own
the rules; cite their IDs, such as **N2**, in every finding. Read the diff and enough surrounding
code to establish each metric's exported name, type, labels, and the code that registers and
updates it. Apply the checks to any area of the repository that the change touches.

A metric's name, type, unit, and labels are a contract that dashboards and alerts depend on. A
rule violation is a contract defect, not a style preference.

## Find the telemetry

1. Look for metric definitions and changes:
   - Rust: new `pub const` names in `lib/runtime/src/metrics/prometheus_names.rs`;
     `.metrics().create_*` and `create_metric_with_options` calls; `prometheus` crate
     constructors such as `IntCounterVec::new`; custom `Collector` implementations.
   - Python: `prometheus_client` `Counter`, `Gauge`, `Histogram`, `Info`, and `Enum`; registry
     bridges such as `register_engine_metrics_callback`,
     `register_prometheus_expfmt_callback`, and `register_prometheus_typed_callback`.
   - Go: `client_golang` constructors such as `prometheus.NewGaugeVec`, and registration in
     `InitMetrics`.
   - Docs: `metrics-catalog.mdx`, `operator-metrics.mdx`, and `metric-labels.mdx` under
     `docs/fern/pages/reference/observability/`.

2. Look for added log events (`tracing::` macros, Python `logging` and `logger` calls) on request
   paths, and for added spans (`tracing` span macros, `#[instrument]`, `.instrument(...)`, and
   `dynamo.common.backend.telemetry`).

3. Skip test-only code (`tests/`, `#[cfg(test)]`, and code behind the `bench` feature) and the
   generated `lib/bindings/python/src/dynamo/prometheus_names.py`; review its Rust source
   instead. When the change adds, renames, or removes a constant in `prometheus_names.rs`
   without regenerating that file, report it under **R3** with the regenerate command. If the
   change adds or modifies no metric, log event, or span, say so and stop.

4. A metric that the change registers, exports, or updates for the first time is new for this
   review. Check its whole definition (name, type, unit, labels, and buckets), even where those
   lines are unchanged.

## Build the metric table

5. For each new or changed metric, record its exported name, type, declared labels with the
   source of each label's values, series per process, registration site, production update
   site, catalog entry, and stability level. Resolve the exported name the way the API builds
   it:
   - Rust `create_*` prepends `dynamo_component_`, or the prefix given in `MetricOptions`. A
     full `dynamo_...` name passed in gets the prefix twice.
   - Go joins `Namespace` and `Name` with `_`.
   - Python `prometheus_client` adds `_total` to a counter's exported name.

6. Count only the labels that the author declares. The runtime adds `dynamo_namespace`,
   `dynamo_component`, `dynamo_endpoint`, and `worker_id`, and the Python bridges add `model`
   and `model_name`; do not report those against **L1**, **L5**, or **L7**.

## Checks

### Metric or log

7. **C1, L2.** Report a label whose value is a request, session, user, or tenant ID, a prompt, a
   URL path, or a value copied from a request or protocol field. The correction moves the detail
   to a log event with fields and keeps, at most, an aggregate counter with bounded labels.

### Justification

8. **A1-A6.** Check the pull request description for the Metric, Reader, Labels, Series,
   Query?, and Docs block for each new metric. When the block or an answer is missing, ask for
   it under "Questions for the author"; do not report it as a defect. When you cannot see the
   description, ask the questions anyway. Check both numbers that **A6** asks for against the
   declared labels and buckets: label combinations, and series (1 per combination for a counter
   or gauge, buckets + 3 for a histogram in Rust and Go, and in Python buckets + 4 per histogram
   and 2 per counter combination). Report a metric whose declared labels alone can exceed 100
   combinations in one process when the pull request does not justify it; ask when the count
   depends on deployment values the diff does not show.

9. **A2, A4, A5.** Search `prometheus_names.rs`, the Metrics Catalog, and
   `metrics-comparison.md` for an existing metric with the same measurement point, or one that
   differs only by a modality, a direction, or a role. Report only a concrete match: name both
   metrics, and give the PromQL that replaces the new one or the label to add. Report static
   configuration exported as separate gauges, and give the `*_info` metric that replaces them.

10. **A3.** Report a new `dynamo_` metric that copies a value the engine already exports, unless
    the pull request says which engines it normalizes and documents the mapping.

### Naming

11. **N1-N5.** Check the exported name: the `dynamo_<subsystem>_` prefix; a unit in base form
    (not `_ms`, `_us`, `_ns`, `_percent`, `_kb`, or `_gb`); `_total` on a counter, after the
    unit; no `_total` on a gauge; no label name or value in the name. The correction gives the
    exact replacement name and, in Rust, the constant to change.

12. **N6.** Read the update site and confirm that the value matches the name and help text. For
    example, a `_seconds` histogram that observes milliseconds is a defect. N6 concerns the
    quantity, unit, and measurement point. A change that only alters when the value is published,
    such as batching or flushing updates, is not an N6 defect; raise stale wording on unchanged
    lines as a question, not a finding.

### Labels

13. **L1, L3, L4.** Confirm that each declared label takes its values from a bounded set: an
    enum, a constant, or a mapping with a fallback value. The frontend resolves `model` with
    `ModelManager::metric_model_for`; other components use a fixed `unknown` value. Report a
    label that mixes different quantities.

14. **L5.** Report a new declared label whose values grow with the number of workers, such as
    `worker_id` or `dp_rank` on a frontend, router, or planner metric. The correction exports
    the value from the worker's own metrics, where the runtime's `worker_id` identifies it. For
    a new `dp_rank` label on a worker's own metric, ask which autoscaler or router needs
    per-rank values when the pull request does not say.

15. **L6, R1.** Report a new labeled family whose children are created only in the request path,
    or whose registration happens on first use, for example inside `OnceLock::get_or_init` on
    the update path. Before reporting, search the code at the pull request head for a seeding
    call at startup or at model registration. A registered family with no children does not
    appear on `/metrics`.

16. **L7.** For a new histogram, check for about 10 buckets, an edge at each named alert
    threshold, and `model` as the only declared label. A Rust histogram created with `None`
    buckets gets the default edges for seconds (0.005 to 10), which do not fit tokens or bytes.
    When a correction introduces or changes a histogram, give a literal bucket list in the
    metric's unit, and name the edge that matches each alert threshold or say that none is named.

17. **R2.** Report a new metric with no production update site at the pull request head. Also
    confirm that each request path that the name and help text cover updates the metric (for
    example, streaming and unary, token and text modes, aggregated and disaggregated), and report
    a path that registers the family but never updates it.

### Stability

18. **S1-S4.** For a change to an existing metric, read its catalog entry at the pull request
    head: an entry with no `**Experimental.**` mark is stable, and one with `deprecated={true}`
    is deprecated. Report, on a stable metric, a rename, a type or unit change, or an added or
    removed label that does not add a new metric and deprecate the old one. Report a
    deprecation that lacks the catalog mark, the help-text note, the replacement or PromQL
    recipe, or the removal release. Report a removal of a stable metric that has not been
    deprecated for two releases, or any removal on a `release/*` base branch. Search the in-repo
    readers (dashboards under `dev/observability/` and `deploy/observability/`, and the planner)
    for a renamed or removed name, and report each reader that still uses it.

### Documentation

19. **D1.** Report a new or changed Dynamo metric whose catalog entry the pull request does not
    add or update, and a new entry that lacks the type, unit, labels and their values, or the
    `**Experimental.**` mark.

20. **D2.** When the pull request adds a metric and does not describe it in the `AGENTS.md`
    nearest to the owning code, ask for that description under "Questions for the author" and
    name the file. Do not ask for a new `AGENTS.md` file.

### Logs and traces

21. **T1, T2.** On added lines only, report a Rust request-path event that formats an ID into
    its message instead of passing it as a field, a spawned request-path task that does not
    carry the span (`.instrument(...)` or `in_current_span()`), a Python request-path warning or
    error that omits `context.id()`, and Python fields passed with `extra=`, which the log
    bridge drops. Do not report a log event for not being JSON; the output format is deployment
    configuration.

22. **T3.** Report a metric label added to follow one request across components; a span gives
    that view. Report span parentage defects only as the runtime review prompt describes them,
    and do not invent span naming rules.

## Out of scope

Do not report:

- A violation on a line that the change does not add or modify, including code that the change
  only moves, and the names that the guidelines list under "Existing names that break these
  rules" when they already exist before the change (check with `git grep -n <name> <base>`). A
  metric that the change registers, exports, or updates for the first time is not pre-existing
  (step 4).
- Engine metrics that Dynamo passes through (`vllm:`, `sglang:`, `lmcache:`, and the engine's
  `trtllm_` metrics), or changes to the prefix filters that pass them.
- A rule for which the pull request description records an exception with a reason. Mention the
  exception once; do not argue it again.

## Reporting findings

Report as findings only rule violations that the change introduces and that you verified in the
code at the pull request head. Each finding states:

- **Location:** the file and line of the definition or update.
- **Rule:** the rule IDs.
- **Problem:** the exported name, label, or line, quoted.
- **Consequence:** the concrete cost, such as the series count with its arithmetic, the
  dashboard or alert that would break, a unit that readers would misread, or a series that is
  absent until the first request.
- **Correction:** the exact change: the replacement name and constant, the label set, the
  seeding call, the log event or PromQL to use instead, or the catalog entry to add.

Keep one finding per underlying defect. When one fact breaks several rules, report one finding
that cites all of them; when one defect repeats across metrics added together, report it once
and list every location. Put findings about names, types, units, labels, stability, and
unbounded series first, because they are hardest to change after release. For example:

```text
lib/llm/src/http/service/metrics.rs:1412 (N2, N3)
Problem: new IntCounterVec "prefill_wait_ms" exports dynamo_frontend_prefill_wait_ms{model}.
Consequence: the unit is not base form and the counter has no _total, so rate() panels show
  milliseconds per second under a name that reads as a gauge; renaming after release breaks them.
Correction: record seconds and export dynamo_frontend_prefill_wait_seconds_total: rename the
  constant to PREFILL_WAIT_SECONDS_TOTAL = "prefill_wait_seconds_total" in prometheus_names.rs.
```

Put information that only the author can give (a reader, a series count, a label's source, which
processes update a metric) in a separate "Questions for the author" block, and do not present it
as a defect. End with the metric table from step 5. If no finding qualifies, state that no new
findings were identified; this does not establish that the change is ready to approve.
