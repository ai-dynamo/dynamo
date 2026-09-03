# Prefill Continues Decode

**Experimental.** A prefill worker keeps a request and generates the whole response, instead of
handing it to a decode worker. No KV transfer happens. It is worth doing only when decode is the
scarcer resource.

Enable it with `--router-prefill-continue`, and configure it with
`--router-prefill-continue-config`, a JSON object.

Both the prefill and the decode set must use `--router-mode kv`. The load signals that drive the
decision exist only under KV routing.

## The settings

```json
{
  "decode_busy_threshold": 0.30,
  "prefill_busy_threshold": 582.0,
  "max_concurrent": 16,
  "max_budget_tokens": 8192,
  "force": false
}
```

| setting | required | unit | asks |
|---|---|---|---|
| `decode_busy_threshold` | yes, unless `force` | fraction of KV capacity | Is decode full enough to be worth avoiding? |
| `prefill_busy_threshold` | yes, or set `--router-queue-threshold` | batches of prefill work | Does this prefill worker have anything to give? |
| `max_concurrent` | **always** | count per worker | How many continuations may one worker hold? |
| `max_budget_tokens` | no | tokens | Refuse a request whose budget is larger than this |
| `force` | no | flag | Bring-up only. Skips the decode test |

The gates run in that order. Each one refuses on its own, and an unreadable signal refuses.

## `decode_busy_threshold`

```
continue when   decode_used_blocks / decode_total_blocks  >  threshold
```

The reading is the selected decode worker's own report, taken before this request is admitted.
Nothing is projected onto it, so the value means one thing: how full decode already is.

**It is a fill line, not a budget.** Setting 0.30 reserves nothing. Other requests keep arriving
after the reading, and part of the used share is evictable prefix cache that the engine reclaims
when it needs room.

### How to choose it

Do not guess. Measure, with two arms that differ in this value alone.

1. Set the threshold above 1.0. The router still reads occupancy and fills the
   `prefill_continue_decode_occupancy` histogram, and every request refuses with `decode_has_room`.
   Nothing continues.
2. Run your real workload on that arm. Replay the whole trace, because prefix cache warmth builds as
   it runs and a truncated replay measures only the warm-up.
3. Read the histogram and pick a value. Write it down.
4. Set that value and run the treatment arm.

Both arms pay the same probe cost, so they differ in one number.

### It is not portable across shapes

More prefill workers feed the same decode worker harder, so decode occupancy rises with prefill
count. One deployment measured two shapes at the same concurrency, on the same trace, both cold.
Share of reads at or below each line:

| `le` | 1 prefill worker | 2 prefill workers |
|---|---|---|
| 0.25 | 60 % | 5 % |
| 0.30 | 76 % | 7 % |
| 0.55 | **100 %** | 12 % |
| 0.70 | 100 % | 18 % |
| 0.85 | 100 % | 62 % |
| 0.95 | 100 % | 95 % |

The medians are 0.22 and 0.82. Adding one prefill worker moved the whole distribution by about 0.6.

| threshold | fires on 1 prefill worker | fires on 2 |
|---|---|---|
| 0.30 | 24 % | 93 % |
| 0.55 | 0 % | 88 % |
| 0.70 | 0 % | 82 % |

**Measure your own shape.** A value that fires usefully on one fires on nearly every request of the
other. Changing the prefill worker count is enough to invalidate a threshold, so treat it as a
per-deployment setting and re-measure when the shape changes.

## `prefill_busy_threshold`

```
hand off when   active_prefill_tokens  >  threshold x max_num_batched_tokens
```

Both sides belong to **one worker**: the prefill worker this request would be routed to. The router
takes an advisory route preview, reserves nothing, and reads that worker's own in-flight prefill
tokens against its own batch budget. Normal KV-aware selection picks it, so it is the best
candidate. A busy verdict means even the best prefill worker is loaded.

**Hand off means serve the request normally.** Nothing is rejected, queued, or delayed. The request
takes the ordinary disaggregated path: prefill here, KV transfer, decode there. It is what would
happen with the feature switched off. A `prefill_busy` count of 3,312 means 3,312 requests were
served the usual way, not that 3,312 requests were refused.

**This is not a fraction.** It is a multiplier on one batch's token budget, so it counts **batches
of prefill work queued** on the worker. A value of 1.0 hands off once one full batch is
queued, and values well above 1 are normal.

The setting is new. The quantity is not: `prefill_load_exceeds` and the unit come from the router,
and `--router-queue-threshold` uses the same predicate. This setting falls back to that one when it
is unset, so a deployment that already tuned queueing gets a sensible default. The name follows that
older setting, which its own documentation calls a fraction while writing the formula as a
multiplier. Read the formula, not the name.

It measures ordinary prefill work. It cannot see a continuation that is already generating, because
the router clears a request's prefill load at its first token. `max_concurrent` is what bounds
those.

### What you need to set it

Four numbers:

| | where it comes from |
|---|---|
| `max_num_batched_tokens` | the engine flag on your prefill workers |
| target concurrency | the load you expect |
| prefill worker count | your deployment shape |
| typical input length | your workload |

```
threshold = (concurrency / prefill_workers) x input_length / max_num_batched_tokens
```

Read it as: requests each prefill worker holds, times batches per request.

### A worked example

One deployment runs `max-num-batched-tokens 32768` on a trace whose inputs average about 74,500
tokens. Each request is therefore about 2.3 batches of prefill work.

| shape | concurrency | requests per prefill worker | threshold |
|---|---|---|---|
| 3 prefill workers | 128 | 42.7 | **97** |
| 3 prefill workers | 256 | 85.3 | **194** |
| 2 prefill workers | 256 | 128 | **291** |
| 1 prefill worker | 256 | 256 | **582** |

**Size it for the highest concurrency you will run.** One value serves one manifest, and a value
sized for low load binds at high load.

### What goes wrong

**Too low, and it refuses everything before the decode gate is consulted.** At 1.6 it admits a
single in-flight request. One measured run at 1.6 refused 92.8 % of requests at concurrency 128.
Another used a 3-worker value of 97 on a 1-worker shape and refused 3,312 of 3,411 requests. The arm
looked clean and measured nothing.

**Too high, and it never fires.** Then `max_concurrent` is the only thing protecting prefill.

If the router's queue rejects the advisory preview, the request counts as busy and hands off. An
unreadable load also hands off, under `prefill_load_unknown`. Both fail towards the ordinary path.

**Check it after every run.** Read `dynamo_frontend_prefill_continue_decisions_total`. If
`prefill_busy` dominates, the interlock is binding and the decode gate is not being exercised.

## `max_concurrent`

The number of continuations one prefill worker may run at once. Enforced at dispatch, when the
chosen worker is known.

**It is required whenever the feature is on**, and the configuration is rejected without it. The
prefill interlock is cleared at a request's first token, so it cannot count a continuation that is
still generating. This cap can.

Set it to at least 2 if migration is in play. A migration retry builds its replacement stream before
dropping the failed attempt, so the two overlap briefly. A cap of 0 is a kill switch.

**It is a safety ceiling, not a treatment dose.** Refusal counts measure arrival timing and service
time, not how hard the feature was applied. Do not tune it to reach a target continuation rate.

## Reading a run

| metric | says |
|---|---|
| `..._decode_occupancy_reads_total` | source coverage. `known` over the sum must be near 1 |
| `..._decode_occupancy` | the occupancy distribution, and where the line belongs |
| `..._decisions_total` | which gate refused, and how often |
| `..._demotions_total` | continuations withdrawn at dispatch, by reason |
| `..._active` | continuations generating right now |

Read coverage first. If `known` is not dominant, the workers are not publishing occupancy and every
other number is meaningless.

Read the decision counters second. A run whose `continue` count is zero measured nothing, whatever
its latency numbers say.
