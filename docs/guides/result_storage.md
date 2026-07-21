# Result Storage

Store benchmark evidence under the candidate `DEPLOY_ROOT`; no external result database is required initially.

## Evidence Layers

1. `raw_aiperf/`: immutable AIPerf output.
2. `benchmark_audit.json`: validity, completeness, workload identity, and any blockers.
3. `benchmark_summary.json`: normalized metrics and benchmark metadata without interpretation.
4. `performance_analysis.json`: SLO verdicts and comparisons to prior valid runs.
5. `performance_analysis.md`: concise human-readable findings.

## Comparison History

For every audited iteration after iteration 0, compare the current run against:

- the original valid baseline;
- the immediately previous valid same-series iteration;
- the best prior valid same-series result for each objective;
- all prior valid same-series iterations in a compact history table.

If no prior valid run exists, establish the current result as the baseline and make no gain/loss claim. Never substitute
an invalid, proxy-mismatched, fixed-schedule-mismatched, or otherwise non-comparable result.

Store absolute values, signed percent deltas, whether each metric is higher or lower, and whether that direction is an
improvement or regression. Report uncertainty or missing metrics explicitly.
