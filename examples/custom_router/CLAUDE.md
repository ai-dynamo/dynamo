# Custom Router Example Notes

This directory demonstrates the intended Python customization boundary for
routing strategies in Dynamo.

## Rules

- Build on `dynamo.llm.KvRouter`
- Do not use lower-level KV internals like `RadixTree` or `ZmqKvEventListener`
- Keep strategy-specific logic inside `select_worker()`
- Preserve the request shape accepted by `KvRouter.generate(...)`

## Pattern

- `base.py` owns request parsing and forwarding
- strategy modules only compute worker choice + metadata
- `run.py` owns the `dynamo_worker()` entrypoint and endpoint serving

## When adding a strategy

1. Create a new subpackage next to `thompson_sampling/`
2. Keep the algorithm documentation near the implementation
3. Expose the strategy from `__init__.py`
4. If the runner needs selection by name, add a small registry in `run.py`
