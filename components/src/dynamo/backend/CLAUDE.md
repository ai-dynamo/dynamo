# Backend Interface Package

Base classes for implementing LLM backend workers in Dynamo. Uses the Template
Method pattern to standardize the worker lifecycle while letting each backend
plug in framework-specific logic.

## Package Layout

```
dynamo/backend/
    __init__.py      # Public API (re-exports all 6 public classes)
    base.py          # Backend ABC (~610 lines)
    handler.py       # Handler ABC + _extract_logprobs helper
    args.py          # DynamoBackendConfig, DynamoBackendArgGroup (+ re-exports runtime-level classes)
    tests/
        test_config.py    # Config + ArgGroup tests (29 tests)
        test_handler.py   # Handler base class tests (17 tests)
```

## Public API

| Export | Purpose |
|---|---|
| `Backend` | ABC with 12-step lifecycle orchestration (`run()`) |
| `Handler` | ABC for request handlers (`generate()`) |
| `DynamoBackendConfig` | Extends `DynamoRuntimeConfig` with common backend fields (model, disagg mode, component) |
| `DynamoBackendArgGroup` | Argparse group for common backend CLI flags |
| `DynamoRuntimeConfig` | Re-exported from `dynamo.common.configuration.groups.runtime_args` |
| `DynamoRuntimeArgGroup` | Re-exported from same |

## Three-Tier Config Hierarchy

```
ConfigBase
  └─ DynamoRuntimeConfig          (dynamo.common) — namespace, planes, connectors, endpoint_types
       └─ DynamoBackendConfig     (dynamo.backend.args) — model, served_model_name, disaggregation_mode, component, use_kv_events
            └─ VllmConfig / etc.  (per-backend) — engine-specific fields on config.extra
```

- `DynamoRuntimeArgGroup` registers `--namespace`, `--discovery-backend`, `--endpoint-types`, etc.
- `DynamoBackendArgGroup` registers `--model`, `--served-model-name`, `--disaggregation-mode`, `--component`
- Each backend adds its own argparse group for engine-specific flags

**Important**: `ConfigBase.from_cli_args()` processes the MRO in reverse, so a
subclass annotation like `model: str = "my-default"` will NOT override the base
class default if the argparse namespace already contains the key (even as `None`).
Set backend-specific model defaults post-parse:
```python
config = MyConfig.from_cli_args(args)
if not config.model:
    config.model = "my-default"
```

## Import Conventions

- Import `Context` from `dynamo.common`, NOT `dynamo._core`
- Import `Backend`, `Handler`, config classes from `dynamo.backend`
- Backend-specific engine types come from their own packages (`dynamo.llm`, etc.)

## Key Design Decisions

- `base.py` (not `backend.py`) for the Backend ABC — avoids `dynamo.backend.backend`
- `handler.py` stays separate from `base.py` despite both being ABCs — different concerns
- `_extract_logprobs` is inlined in `handler.py` (not in `dynamo.common.utils`)
- `_get_model_name()` delegates to `_get_model_path()` to avoid duplication
- `config.extra` (not `config.backend`) for engine-specific fields — avoids confusion with the `Backend` class itself

## Making Changes

### Checklist

1. **Run tests**: `python -m pytest components/src/dynamo/backend/tests/ components/src/dynamo/example_backend/tests/ -x -q -o "filterwarnings=" -o "addopts="`
2. **Update `__init__.py`** if adding/removing public exports
3. **Update `dynamo/example_backend/`** to demonstrate any new pattern
4. **Update `README.md`** for user-facing API changes

### Adding a New Hook to Backend

1. Add the method to `base.py` with a sensible default (no-op or pass-through)
2. Call it from `run()` at the appropriate lifecycle step
3. Add a logged wrapper `_run_<hook>()` if the hook warrants lifecycle logging
4. Document in the class docstring's lifecycle list
5. Update `README.md` hooks table

### Adding a New Field to DynamoBackendConfig

1. Add the field with a default to `DynamoBackendConfig` in `args.py`
2. Add a CLI flag to `DynamoBackendArgGroup.add_arguments()` in `args.py`
3. Add tests in `tests/test_config.py` (field default, CLI propagation)
4. Update `base.py` if the Backend reads this field (replace any `getattr` fallback)
5. Update `example_backend/args.py` if the example should use it

### Modifying Handler

1. `handler.py` imports `Context` from `dynamo.common`
2. `_extract_logprobs()` is a module-level function, not a method
3. `process_generation_output()` calls `_extract_logprobs()` internally
4. Add tests in `tests/test_handler.py`

## Running Tests

```bash
# All backend + example tests (53 total)
python -m pytest components/src/dynamo/backend/tests/ components/src/dynamo/example_backend/tests/ -x -q -o "filterwarnings=" -o "addopts="

# Just config tests (29)
python -m pytest components/src/dynamo/backend/tests/test_config.py -v -o "addopts="

# Just handler tests (17)
python -m pytest components/src/dynamo/backend/tests/test_handler.py -v -o "addopts="
```

The `-o "addopts="` flag is required to override the repo-level `pyproject.toml`
which adds `--mypy` (not installed in the test venv).

## Test Stubs

Tests need `conftest.py` stubs for `dynamo._core` symbols (`Context`, `ModelType`,
`ModelInput`, etc.) because the installed `_core` binary may be out of date.

- `components/src/conftest.py` — root-level stubs (runs before any `dynamo.*` import)
- `dynamo/example_backend/tests/conftest.py` — example-specific stubs
