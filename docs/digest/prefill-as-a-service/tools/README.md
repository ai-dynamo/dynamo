# PFaS result normalization

`normalize_results.py` converts a validated DYN-3364 AIPerf run into the
normalized JSON and request-level CSV used by the article figures. It refuses
an invalid report, a report for another run directory, an unlocked or modified
specification, missing repetitions, request-count drift, and sequence-length
drift.

Run it through `uv` after the cluster-side validator has passed:

```bash
uv run --no-project python normalize_results.py \
  "$RUN_ROOT" \
  "$RUN_ROOT/aiperf/$RUN_ID" \
  "$RUN_ROOT/validation/$RUN_ID.json" \
  --json-output ../data/results.json \
  --csv-output ../data/requests.csv
```

The normalized files retain the locked specification fingerprint, validation
report checksum, source-artifact checksums, per-repetition metrics, worker
pairs, NIXL transfer counts, and request IDs. They do not infer internal
prefill, transfer, network, or decode latency from client-visible timing.
The locked workload records AIPerf's configured synthetic sequence lengths;
request rows record server-reported usage after the chat template is applied.

Render the three deterministic SVG figures from the normalized JSON without
additional Python dependencies:

```bash
uv run --no-project python plot_results.py ../data/results.json \
  --latency-output ../images/client-latency.svg \
  --throughput-output ../images/throughput.svg \
  --timeline-output ../images/client-timeline.svg
```

The latency figure plots request-level client ECDFs separately for each locked
repetition. The throughput figure reports canonical AIPerf averages. The
timeline selects the request nearest the median end-to-end latency and divides
it only into client-observed time to first token and first-to-last-token time;
it does not assign either interval to an internal prefill, NIXL, network, or
decode phase.
