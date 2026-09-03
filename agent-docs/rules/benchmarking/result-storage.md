<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Result Storage

Store benchmark evidence under the candidate `DEPLOY_ROOT`; no external result database is required initially.

## Evidence Layers

1. `raw_aiperf/`: immutable AIPerf output.
2. `benchmark_audit.json`: validity, completeness, workload identity, and any blockers.
3. `benchmark_summary.json`: normalized metrics and benchmark metadata without interpretation.
4. `performance_analysis.json`: SLO verdicts and comparisons to prior valid runs.
5. `performance_analysis.md`: concise human-readable findings.

## Comparison History

For every valid result, use only references from its active benchmark series. When available, report:

- the earliest valid result in the series;
- the immediately previous valid same-series iteration;
- the best prior valid same-series result for each objective;
- all prior valid same-series iterations in a compact history table.

If no prior valid same-series result exists, establish the current result as the series baseline and make no gain/loss
claim. If the active plan requires a specific reference, do not report a delta until that reference has a valid result
in the series. Never substitute an invalid or otherwise non-comparable result.

Store absolute values, signed percent deltas, whether each metric is higher or lower, and whether that direction is an
improvement or regression. Report uncertainty or missing metrics explicitly. Cross-series results may provide context,
and multiple series may inform promotion, but each direct claim must use comparable runs.
