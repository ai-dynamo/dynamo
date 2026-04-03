# Example Backend

A minimal reference backend for Dynamo that replies with "Hello World!" streamed
token by token. Use it as a starting point when writing a new backend.

## Package Layout

```
dynamo/example_backend/
    __init__.py       # Public API: ExampleBackend, ExampleBackendConfig, ExampleHandler
    __main__.py       # Entry point: python -m dynamo.example_backend
    args.py           # Config (extends DynamoBackendConfig), parse_args()
    backend.py        # ExampleBackend — 3 abstract methods + extract_runtime_config()
    handlers.py       # ExampleHandler — streams "Hello World!" tokens
    tests/
        conftest.py             # Stubs for dynamo._core symbols
        test_example_handler.py # 7 tests for token delay, cancellation, format
```

## Architecture

```
DynamoBackendConfig        (from dynamo.backend.args)
  └─ Config                (args.py) — adds config.extra: ExampleBackendConfig

Backend (ABC)              (from dynamo.backend)
  └─ ExampleBackend        (backend.py) — implements create_engine, create_handler, get_health_check_payload

Handler (ABC)              (from dynamo.backend)
  └─ ExampleHandler        (handlers.py) — implements generate()
```

## Config Pattern

`Config` extends `DynamoBackendConfig` (which extends `DynamoRuntimeConfig`).
Engine-specific fields go on `config.extra`:

```python
class Config(DynamoBackendConfig):
    extra: Optional[ExampleBackendConfig] = None

config = Config.from_cli_args(args)
if not config.model:
    config.model = "example-model"        # post-parse default
config.extra = ExampleBackendConfig(...)  # engine-specific
```

**Important**: Do NOT set `model: str = "example-model"` as a class annotation
on Config. `ConfigBase.from_cli_args()` MRO ordering means the
`DynamoBackendConfig.model = None` default wins. Set model defaults post-parse.

`parse_args()` uses both arg groups:
```python
DynamoRuntimeArgGroup().add_arguments(parser)   # --namespace, --discovery-backend, etc.
DynamoBackendArgGroup().add_arguments(parser)    # --model, --disaggregation-mode, etc.
parser.add_argument("--token-delay", ...)        # example-specific
```

## Making Changes

### Checklist

1. **Run tests**: `python -m pytest components/src/dynamo/example_backend/tests/ -x -q -o "filterwarnings=" -o "addopts="`
2. **Run all backend tests** (example + base): `python -m pytest components/src/dynamo/backend/tests/ components/src/dynamo/example_backend/tests/ -x -q -o "filterwarnings=" -o "addopts="`
3. **Update `__init__.py`** if adding/removing public exports
4. **Update `README.md`** for user-facing changes

### Adding a New Example Feature

1. Add config field to `ExampleBackendConfig` in `args.py`
2. Add CLI flag in `parse_args()` (after the two arg groups)
3. Set `config.extra.<field>` in `parse_args()`
4. Read it in `ExampleBackend.create_handler()` via `self.config.extra.<field>`
5. Use it in `ExampleHandler`
6. Add test in `tests/test_example_handler.py`

### Updating Config

- `Config` inherits from `DynamoBackendConfig` — shared fields (model, component,
  disaggregation_mode, etc.) are already available
- Only add fields that are truly example-specific
- Backend-specific config goes on `config.extra`, not directly on `Config`

## Import Conventions

- `from dynamo.backend import Backend, Handler` — base classes
- `from dynamo.backend import DynamoBackendConfig, DynamoBackendArgGroup, DynamoRuntimeArgGroup` — config
- `from dynamo.common import Context` — NOT `from dynamo._core`

## Running Tests

```bash
# Example backend tests only (7)
python -m pytest components/src/dynamo/example_backend/tests/ -x -q -o "filterwarnings=" -o "addopts="

# All backend + example tests (53)
python -m pytest components/src/dynamo/backend/tests/ components/src/dynamo/example_backend/tests/ -x -q -o "filterwarnings=" -o "addopts="
```

The `-o "addopts="` flag is required to override the repo-level `pyproject.toml`
which adds `--mypy` (not installed in the test venv).

## Key Files in dynamo/backend/

This package depends on `dynamo.backend` — see `dynamo/backend/CLAUDE.md` for:
- Three-tier config hierarchy details
- Backend lifecycle (12 steps)
- Handler utilities (cancellation, output processing, trace headers)
- How to add new hooks or config fields
