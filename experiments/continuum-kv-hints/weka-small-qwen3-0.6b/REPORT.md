# WEKA / Qwen3-0.6B Workstation Results

## Scope

These runs validate request lifecycle annotations, Dynamo `SessionPrefixIndexer` resolution, deferred `kv.retain` and `kv.evict` hints, and vLLM G1 execution on one workstation GPU. The workload is a deterministic, scaled subset of `semianalysisai/cc-traces-weka-062126-256k`; `common/prepare_dataset.py` preserves session and subagent boundaries while scaling token counts by eight to fit Qwen3-0.6B and the configured KV capacity.

The earlier fixed-TTL runs used local image `continuum-weka-qwen-policy:local` (`sha256:a4f355ad536c7fa3fb04b7e3fd3017e4a14dfd6cd3277197bb1905e287a6bd12`). The source-clock oracle run used `continuum-weka-qwen-oracle:local` (`sha256:316537df9df8586b98e43423de51229d308e818145ad7fcb96a496953c24dfa6`). Every run directory captures the Dynamo and AIPerf commit IDs and working-tree patches; the vLLM overlay is commit `9b6e116be2d9efdd044df1d738bba5aabdfbbd56`.

## Source-Clock Oracle Retention Matrix

Artifacts: `artifacts/oracle-retention-20260921-20260921T192212Z`

All four cases completed 275 requests without request errors. AIPerf preserved source-clock request timing and attached the recorded pause until the next parent turn as `X-Dynamo-Retention-TTL-Ms`. Dynamo emitted five retain actions with request-derived TTLs of 15.702, 335.847, 354.531, 2.543, and 183.211 seconds. The fixed 300-second TTL was not used.

| Case | Cache read | Cache-read tokens | TTFT avg ms | TTFT p50 ms | TTFT p99 ms | Retain | Evict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| No hints | 78.53% | 1,964,480 | 151.2 | 47.2 | 1,413.3 | 0 | 0 |
| Parent retain | 76.11% | 1,904,192 | 187.5 | 49.1 | 1,625.0 | 5 | 0 |
| Root-final evict | 78.40% | 1,961,088 | 143.8 | 46.4 | 1,227.2 | 0 | 4 |
| Combined | 75.09% | 1,878,016 | 194.1 | 47.0 | 2,016.6 | 5 | 4 |

The five parent resumptions directly targeted by retention improved: cache-read tokens increased from 30,656 to 38,144 and average TTFT fell from 156.4 ms to 123.4 ms. Aggregate reuse still regressed because the five actions resolved 1, 2, 5, 11, and 2 lineages for the same parent session. These extra frontiers are orphaned completion tails created because the replay's next input does not include the model output generated locally; they are not child-subagent lineages. Unioning them increased cache pressure. The next policy refinement should select the carrying request's frontier while preserving SessionPrefixIndexer resolution.

## Retention Occupancy Characterization

Artifacts:

- `artifacts/retention-occupancy-unbounded-20260921`
- `artifacts/retention-cap-40-20260921`
- `artifacts/retention-cap-25-20260921`
- `artifacts/retention-cap-sweep-20260921`

The source-clock sweep compared no hints, unbounded oracle-TTL retention, and 40% and 25% retained-G1 admission caps using 1,024 G1 blocks. Every case ran for 630 seconds without request errors. The retained-block gauge was integrated over AIPerf's exact profiling window.

| Case | Mean retained | p95 retained | Peak retained | Cache read | TTFT p50 ms | TTFT p99 ms | Targeted cache-token gain | Targeted TTFT delta ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| No hints | 0.00% | 0.00% | 0.00% | 82.82% | 44.4 | 1,045.3 | - | - |
| Unbounded retain | 20.62% | 33.69% | 65.62% | 78.02% | 46.5 | 1,269.6 | 19,520 | -48.5 |
| 40% cap | 20.53% | 33.69% | 35.64% | 79.54% | 45.0 | 1,223.1 | 21,120 | -50.9 |
| 25% cap | 14.27% | 24.12% | 24.12% | 80.22% | 46.2 | 1,704.7 | 13,440 | -30.6 |

Eleven hints were emitted in each retention run. Nine had a subsequent parent resume before profiling ended; the remaining two occurred near the end of the window. The no-hint run missed 33,600 theoretically reusable tokens across those nine matching source positions. Unbounded retention recovered 19,520 tokens, or 58.1%; the 40% cap recovered 21,120, or 62.9%; and the 25% cap recovered 13,440, or 40.0%. These comparisons match each candidate's runtime resume to the mean no-hint result for the same source position, avoiding double-counting when a source trace is recycled.

Unbounded retention nevertheless reduced aggregate cache reuse by 4.80 percentage points versus no hints and forced reuse of retained blocks 29 times. Two short multi-frontier actions produced the 65.62% and 60.55% occupancy spikes. The 40% cap rejected those actions and improved aggregate reuse without sacrificing targeted recovery because those parent resumes were already fully cached in the baseline. The 25% cap rejected five of 11 actions, reduced mean retained occupancy to 14.27%, and improved aggregate reuse by another 0.67 percentage points over 40%, but it also reduced targeted recovery. No tested retention configuration beat the no-hint aggregate result. Replicated runs and a more selective trigger are required before choosing a production cap.

## Baseline Cache-Gap Localization

Artifacts: `artifacts/baseline-cache-gap-20260921`

The analyzer reconstructs AIPerf's theoretical metric from the generated WEKA hash streams and compares it with each request's observed vLLM cache-read tokens. AIPerf's theoretical result is an infinite-capacity per-trace seen set shared across root and child streams. The reconstruction exactly matches the recorded 77,846 reusable blocks out of 83,937 total blocks, or 92.74%. The no-hint run read 4,523,968 cached tokens, or 70,687 complete 64-token blocks. Its published 82.82% observed cache-read rate uses all prompt tokens as the denominator; the complete-block-normalized rate is 84.21%.

| Stream | Requests | Theoretical blocks | Observed blocks | Positive gap blocks |
| --- | ---: | ---: | ---: | ---: |
| Root | 95 | 15,976 | 13,769 | 2,233 |
| Subagent | 506 | 61,870 | 56,918 | 5,686 |
| Flat | 2 | 0 | 0 | 0 |

Of the 7,919 positive-gap blocks, 7,843, or 99.0%, were reusable within the same root or child stream. Cross-stream idealization therefore explains little of the deficit. Subagent streams account for 71.8% of the positive gap; root streams account for 28.2%.

| Time since the same runtime session's previous response | Requests | Positive gap blocks | Share of gap |
| --- | ---: | ---: | ---: |
| First turn | 103 | 1,067 | 13.5% |
| Less than 1 second | 102 | 155 | 2.0% |
| 1-10 seconds | 328 | 2,936 | 37.1% |
| 10-60 seconds | 54 | 2,408 | 30.4% |
| At least 60 seconds | 16 | 1,353 | 17.1% |

Existing sessions resumed after at least one second account for 84.6% of the positive gap, while sub-second resumes are nearly fully cached. This concentration is strongly consistent with finite-capacity LRU eviction during pauses, although joining each request to its exact `BlockRemoved` history would be required to prove the cause per block.

Most loss is not loss of the root block. Twenty requests lost their complete reusable prefix, accounting for 1,698 gap blocks, while 60 requests retained an initial prefix but lost a deeper suffix, accounting for 6,221 gap blocks. Four of the seven root requests immediately following a subagent lost their complete prefix and one lost a suffix; together these resumes account for 525 gap blocks. Parent-at-spawn retention recovered roughly this narrow opportunity, but it does not address the larger subagent-session gap.

The current policy already retains the carrying parent lineage through `include_current_request=true`. SessionPrefixIndexer resolution additionally contributes earlier frontiers belonging to that same parent session. This is not a correctness failure and does not mix child sessions into the action, but it can over-retain orphaned completion tails in this replay. It need not block the parent-retention experiment; the cap contains its cost. A future high-frequency child-pause policy should explicitly choose between request-scoped retention of the carrying lineage and session-scoped retention of every frontier. Sub-second pauses should remain on ordinary LRU. Longer retention should remain budgeted because its saved-prefill benefit competes directly with cache pressure on other sessions.

The subagent-side opportunity has not yet been exercised. Existing child sessions resumed after at least one second account for 4,464 positive-gap blocks, or 285,696 tokens: 2,657 blocks after 1-10 seconds, 1,289 after 10-60 seconds, and 518 after at least 60 seconds. The June 21 corpus does not populate per-request `stop`, `input_types`, or `output_types`, and it provides no tool-call IDs. Therefore these boundaries can only be annotated as oracle continuation pauses from the next same-session request's recorded delay, not asserted to be identified tool calls.

## Prioritized Retention Validation

Artifacts:

- `artifacts/prioritized-validation-baseline-20260922`
- `artifacts/prioritized-validation-policy-fixed-20260922`

The validation selected only policies whose oracle cost model predicted positive net cache-token value. Both used a hard 96-block retained-capacity limit, or 9.375% of the 1,024-block G1 cache. The missing-suffix policy issued 19 range-addressed retains and reached the cap; three additional leases were rejected atomically. The whole-prefix policy issued two retains, reached 92 blocks, and had no rejected leases. Both completed the same 275 requests as the fresh baseline without request errors.

| Case | Predicted net cached tokens | Observed target gain | Observed aggregate delta | Cache-read delta | Paired mean TTFT delta | Peak retained |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Missing suffix | +49,839 | +12,864 | -32,448 | -1.30 pp | +7.0 ms | 96 blocks |
| Whole prefix | +5,968 | +6,912 | -60,864 | -2.43 pp | +13.4 ms | 92 blocks |

Both policies benefited their intended next requests but displaced more reuse elsewhere than they recovered. Missing-suffix targeting realized 22.6% of its ideal target recovery and incurred 708 non-target lost blocks over 3,200 retained block-seconds. Whole-prefix targeting realized 61.0% of ideal target recovery and incurred 1,059 non-target lost blocks over 2,563 retained block-seconds. The corresponding observed opportunity-cost rates, 0.22 and 0.41 lost blocks per retained block-second, were much larger than the simulator's 0.026 coefficient. The model therefore did not predict either run's sign correctly.

An earlier invalid attempt reused a no-hint mmap dataset cache and emitted no scheduled headers; its artifacts were discarded. The launcher now disables this cache whenever `RETENTION_ORACLE_SCHEDULE` is set. A four-request ingress check and both full reruns verified header dispatch, Dynamo action generation, vLLM lease admission, expiration, and forced-reuse handling.

The fresh no-hint baseline read 1,992,384 cached tokens. A prior otherwise comparable 275-request no-hint run read 1,964,480, a 27,904-token spread. Both policies remained below both baselines, but the exact magnitude of their aggregate regressions is single-run evidence. The robust conclusion is that this scoring model is not yet accurate enough to select winning retention actions; retained byte-seconds need a substantially higher opportunity-cost penalty and admission should account for already-active leases.

## Correctness Replay

Artifacts: `artifacts/correctness-20260921/d-combined`

- 35 of 35 requests completed with no errors.
- The replay emitted exactly one parent-session retain and one final-root eviction.
- Both actions targeted the expected root session.
- 32 requests reported prompt-cache reuse, totaling 461,568 cached prompt tokens.
- G1 cache usage reached zero after final eviction, and removal-event evidence was observed.

## Earlier Fixed-TTL Matrices

Artifacts:

- `artifacts/weka-matrix-20260921`
- `artifacts/weka-matrix-replicate-2-20260921`
- `artifacts/weka-matrix-replicate-3-20260921`
- `artifacts/weka-replicates-20260921`

Every 300-second case completed without request errors. Each matrix contains `matrix-summary.{json,csv,md}` and per-request AIPerf records. The aggregate directory contains `replicate-summary.{json,csv,md}`. Values below are mean [minimum, maximum] across three runs.

| Case | Requests | Req/s | TTFT p50 ms | TTFT p99 ms | Latency p50 ms | Latency p99 ms | Observed cache read | Retain | Evict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| No hints | 324.3 [323, 327] | 0.989 [0.984, 0.996] | 234.2 [221.8, 244.2] | 9120.3 [8261.2, 10009.5] | 2829.1 [2678.6, 2964.9] | 31279.0 [28994.3, 34458.1] | 57.16% [55.82%, 59.59%] | 0 | 0 |
| Parent retain | 320.7 [320, 321] | 0.972 [0.970, 0.973] | 283.2 [226.4, 343.6] | 8303.6 [7460.1, 9166.3] | 2910.7 [2623.6, 3074.7] | 30148.5 [29147.7, 31721.8] | 54.62% [54.05%, 55.40%] | 15 | 0 |
| Root-final evict | 324.7 [321, 329] | 0.988 [0.978, 0.997] | 226.8 [195.6, 269.0] | 8980.4 [7045.6, 10423.7] | 2780.0 [2538.9, 3025.8] | 29805.8 [29436.8, 30264.2] | 57.44% [56.20%, 58.33%] | 0 | 7 |
| Combined | 323.0 [321, 326] | 0.979 [0.974, 0.988] | 251.4 [226.8, 283.6] | 7967.5 [7457.9, 8907.9] | 2829.9 [2677.4, 2979.6] | 28671.5 [28457.9, 28852.8] | 56.53% [55.53%, 57.32%] | 13 | 7 |

The lifecycle annotations and action counts were reproducible. Sparse parent retention reduced aggregate observed cache reuse by 2.54 percentage points versus baseline and did not improve throughput in these runs. Root-final eviction preserved baseline throughput and cache reuse within the observed run-to-run range. Tail-latency differences were mixed and are not evidence of a performance win at this scale.

## Removal Diagnostic

Artifacts: `artifacts/raw-kv-diagnostic-20260921/c-root-final-evict`

The raw ZMQ capture recorded 658 unique stored hashes and 398 unique removed hashes. Every removed hash had a preceding raw `BlockStored` event. Three worker-local FlashIndexer warnings occurred at removal positions 128, 384, and 385. Dynamo coalesces at 128 blocks; radix removal can eagerly prune descendant hashes in one batch, while the next batch no longer carries the prior batch's `eagerly_removed` set. The warning therefore describes an already-applied removal rather than a missing store. The frontend router did not report the warning, and the ordered SessionPrefixIndexer side queue still received every removal mutation.

## Limits

- Results are from three runs per case on a transformed four-trace subset.
- Token and think-time scaling preserve lifecycle shape, not production timing or model quality.
- G2/KVCR behavior is outside this experiment.
- Root-final eviction physically evicts shared hashes on the selected worker; session-exclusive shared-prefix handling remains future work.
