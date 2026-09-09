<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# One Variable Per Candidate

Change one independently testable knob per candidate relative to the current successful configuration so the measured
delta can be attributed to that change.

This rule is self-enforced by the agent proposing the candidate. Before writing a hypothesis, inspect the complete YAML
diff and verify that it contains one independently testable mechanism. Changing several independent knobs makes the
result ambiguous and can carry neutral or harmful settings forward without showing which one caused the outcome.

## Coupled Changes

A coupled bundle is allowed only when required for functionality or supported by evidence of an interaction. Record
every changed field, the single intended mechanism, and any required follow-up ablation.

Multiple YAML fields may represent one knob when they are all required to activate the same behavior. A coordinated
prefill/decode worker-count adjustment may likewise be one topology hypothesis when both counts must change to preserve
a fixed GPU budget. Do not include optional tuning in the same bundle.

For an interaction-based bundle, cite the prior isolated results that support the interaction. If the knobs have not
been tested independently and no existing evidence shows that they interact, split them into separate candidates. Test
the combined form only after the isolated results justify it.

## Self-Check

Before proposing a candidate:

1. List every field changed from the current successful configuration.
2. State the one intended mechanism and expected measurable effect.
3. Decide whether each field is required for that mechanism or is an independent knob.
4. Split independent knobs into separate candidates.
5. For an allowed bundle, state the exception, cite its supporting evidence, and record any required follow-up
   ablation.

A later integration candidate may combine independently verified `stack` changes, but it must be labeled as an
integration test, cite the isolated results, and be benchmarked as a new candidate. Do not use an integration candidate
to establish the contribution of any individual knob.
