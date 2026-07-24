<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Benchmark Evidence Eligibility

Separate evidence that can decide candidate promotion from observations that can only motivate a hypothesis.

## Decision-Grade Evidence

A benchmark result may support a gain, loss, Pareto, SLO, or promotion claim only when:

- it was produced through the canonical AIPerf path for the optimization run;
- its raw outputs remain unchanged and traceable to the executed configuration;
- `benchmark_audit.json` reports `valid` or `valid_with_recovery`;
- its benchmark-series identity matches the comparison target;
- its workload, endpoint, candidate, and execution identity are recorded; and
- no unresolved correctness, request-integrity, or benchmark-isolation issue invalidates the measurement.

Failed and invalid runs remain part of the experiment history, but they cannot establish performance gains.

## Contextual Evidence

Results from another benchmark client, an ad hoc script, a modified prompt generator, different request lengths, a
different traffic policy, or an unverified wrapper may guide investigation or hypothesis selection. Label them as
contextual and do not use them to promote or reject a candidate on performance.

Logs, Kubernetes events, and endpoint behavior may establish an operational failure or suggest a mechanism. They do
not substitute for an audited AIPerf comparison.
