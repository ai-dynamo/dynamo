# Custom Python Router Example

This example shows the intended extension point for custom routing strategies in
Dynamo: write strategy logic in Python on top of `dynamo.llm.KvRouter`.

## Files

- `base.py`: strategy base class and request forwarding helpers
- `run.py`: `dynamo_worker` entrypoint that serves the custom router
- `thompson_sampling/bandit.py`: Beta-Bernoulli bandit state
- `thompson_sampling/router.py`: Thompson Sampling strategy implementation
- `CLAUDE.md`: instructions for adding another strategy in the same pattern

## Why this example uses `KvRouter`

The example stays at the binding boundary and only uses the Python APIs already
exposed by Dynamo:

- `KvRouter.best_worker(...)`
- `KvRouter.get_potential_loads(...)`
- `KvRouter.generate(worker_id=..., dp_rank=...)`
- `KvRouter.mark_prefill_complete(...)` if a strategy ever needs explicit lifecycle control

It does **not** use lower-level KV primitives such as `RadixTree` or
`ZmqKvEventListener`. Those belong below the intended Python customization layer.

## Endpoints

Running `run.py` serves two endpoints:

- `generate`: select a worker using the Python strategy, then forward the request there
- `select_worker`: inspect the strategy decision without generating tokens

Typical flow:

```text
request
  -> custom_router.select_worker
     -> KvRouter.get_potential_loads(...)
     -> KvRouter.best_worker(...)
     -> Thompson Sampling score
  -> custom_router.generate
     -> KvRouter.generate(worker_id=..., dp_rank=...)
```

## Thompson Sampling Algorithm

The strategy in `thompson_sampling/router.py` is the binding-level adaptation of
the older Rust/NAT Thompson router.

### State

Per worker `(worker_id, dp_rank)`:

- `BetaBandit(alpha, beta)`

Per agent or prefix identity:

- `last_worker`
- `reuse_remaining`

The agent identity is extracted from one of:

- `request["agent_id"]`
- `request["prefix_id"]`
- `annotations` entries like `agent_id:<value>` or `prefix_id:<value>`

### Routing signals

For each request:

1. `get_potential_loads()` returns per-worker
   - `potential_prefill_tokens`
   - `potential_decode_blocks`
2. `best_worker()` returns one exact KV-overlap result
   - `(worker_id, dp_rank, overlap_blocks)`

### Overlap approximation

`KvRouter` does not currently expose a full overlap map for all workers.
Because of that, the strategy uses this approximation for every worker:

```text
overlap_ratio ~= 1 - potential_prefill_tokens / prompt_tokens
```

Then it refines the single exact best-KV candidate returned by `best_worker()`:

```text
exact_overlap_ratio = min(1, overlap_blocks * block_size / prompt_tokens)
```

That candidate keeps the larger of the proxy and exact values.

### Load score

For each worker candidate:

```text
prefill_ratio = potential_prefill_tokens / prompt_tokens
decode_ratio = potential_decode_blocks / max_decode_blocks
reuse_pressure = normalized sum of reuse budgets already pinned to that worker

load_score = 1 / (1
                   + prefill_penalty_weight * prefill_ratio
                   + decode_penalty_weight * decode_ratio
                   + reuse_penalty_weight * reuse_pressure)
```

### Thompson bandit sample

Each candidate samples from its own posterior:

```text
sample ~ Beta(alpha, beta)
```

The implementation uses Python's standard-library `random.betavariate()`. No
extra dependency is required.

### Sticky reuse logic

If the request belongs to an agent/prefix that has already been seen:

- if `reuse_remaining > 0`, the previous worker gets a sticky bonus and all
  others pay a switch penalty
- if `reuse_remaining == 0` and more than one worker exists, the previous
  worker is temporarily excluded to force exploration

### Combined score

The final score per worker is:

```text
combined = sample * load_score
           + overlap_weight * overlap_ratio
           + sticky_bonus
           - switch_penalty
```

The router chooses the worker with the maximum `combined` score.

### Bandit update

This example keeps the update local and immediate. It does not wait for ex-post
latency feedback from another processor component.

After selection, it computes a synthetic reward:

```text
reward = clamp(0.65 * overlap_ratio + 0.35 * load_score, 0, 1)
alpha += reward
beta += 1 - reward
```

This keeps the example small while still documenting the core bandit behavior.
If you want delayed reward updates from observed latency, add a feedback path on
top of this same structure.

## Running

Example:

```bash
python -m examples.custom_router.run \
  --worker-endpoint workers.worker.generate \
  --namespace dynamo \
  --component-name custom_router \
  --router-block-size 128
```

The component will serve:

- `dynamo.custom_router.generate`
- `dynamo.custom_router.select_worker`

## Request shape

`generate` and `select_worker` accept the same request shape used by
`KvRouter.generate(...)`:

```python
{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "token_ids": [1, 42, 99],
    "annotations": ["prefix_id:session-123"],
    "stop_conditions": {},
    "sampling_options": {},
    "output_options": {},
    "extra_args": {},
}
```

## Extending this example

To add another strategy:

1. Subclass `BaseCustomRouter`
2. Implement `select_worker()`
3. Reuse `run.py` or add a strategy registry there
