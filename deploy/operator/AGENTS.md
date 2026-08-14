<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

- RBAC changes for operator code must update the `+kubebuilder:rbac` markers.
- Run `make manifests` to regenerate both `config/rbac/role.yaml` and the
  platform chart's `../helm/charts/platform/components/operator/files/manager-role.yaml`.
- Keep chart-only grants in the manual section of the platform chart's
  `../helm/charts/platform/components/operator/templates/manager-rbac.yaml`.

## Reconciliation and Admission Semantics

- Reconciler state convergence must be level-based. Derive actions from desired
  and observed state on every invocation; never depend on the request operation
  or informer event edge to determine what to do.
- Kubernetes Event emission is intentionally edge-based. Derive these edges
  from explicit object state, such as a spec/status difference or a durable
  annotation, never from informer delivery semantics.
- Admission validation should enforce state invariants independent of the
  request operation. Distinguish between `CREATE` and `UPDATE` only for
  exceptional, intentional API semantics, and document why the distinction is
  required.

## Go Code Style

- Put a one-line story comment above every multi-line block of logically
  connected code.
- Separate multi-line semantic blocks from surrounding code with one blank
  line. Do not add trailing blank lines before a closing delimiter or between
  a block-leading comment and its code.

## Function Preconditions and Nil Handling

- Treat pointer inputs as non-nil by default. Document non-nil preconditions in
  the function's Go doc.
- A function may accept `nil` only when this is explicitly supported as part of
  its domain semantics, such as a mode that does not require the argument.
  Document the supported `nil` cases in the function's Go doc.
- Do not defensively check pointer inputs for `nil` unless `nil` has an explicit
  domain meaning.
- When `nil` is not such a supported domain value, callers must establish the
  non-nil precondition before calling the function.
- If a function validates caller-provided input, return an error for invalid
  values. Do not disguise an invalid call by returning a zero value.
- Do not overload a zero value to mean "not found" when absence must be
  distinguished from a valid zero value. Return an explicit presence boolean
  alongside the value, for example:
  `func NestedValue(x any, path ...string) (value any, found bool)`.
- Getters and transformation functions must not translate a `nil` input into a
  zero value by default. Do so only when nil-tolerant behavior is an intentional,
  documented API convention, such as JSON-path-style getters.

## Go Test Style

- Use `t.Log` to tell the test's story, with one heading before each block that
  implements a test step. Preserve test paragraph comments by converting them
  to `t.Log` headings when moving or refactoring the code.
- Keep test fixtures in local variables or constants and owned by one test.
- Avoid hiding bespoke test logic in closures. Reserve closures for standard
  helpers such as `Eventually`.
- Prefer a shared construction DSL over shared fixture objects when setup is
  complex or repetitive; the DSL should describe the object at a high level.
- Prefer table tests over one-off tests and do not duplicate behavior already
  covered by a table.
