# Example Backend

A minimal reference backend for Dynamo that replies with "Hello World!" streamed token by token. Use it as a starting point when writing a new backend.

## Quick Start

```bash
python -m dynamo.example_backend --model <model_name_or_path>
```

The model's tokenizer is used to encode the reply. If the tokenizer is unavailable, a single fallback token is returned instead.

## Options

All [common Dynamo runtime flags](../backend/README.md) are supported (`--namespace`, `--store-kv`, `--model`, etc.), plus:

| Flag | Default | Description |
|---|---|---|
| `--token-delay` | `0.0` | Delay in seconds between each generated token. Useful for simulating slow inference. |

Example with a 100 ms delay per token:

```bash
python -m dynamo.example_backend --model <model_name_or_path> --token-delay 0.1
```

## Files

| File | Purpose |
|---|---|
| `__main__.py` | Entry point (`python -m dynamo.example_backend`) |
| `args.py` | `Config` (extends `DynamoBackendConfig`), `ExampleBackendConfig`, and CLI argument parsing |
| `backend.py` | `ExampleBackend` — implements the three required `Backend` abstract methods |
| `handlers.py` | `ExampleHandler` — tokenizes "Hello World!" and streams tokens with cancellation support |

## How It Works

1. `ExampleBackend.create_engine()` returns `None` (no real engine needed).
2. `ExampleBackend.create_handler()` creates an `ExampleHandler` with `config.extra.token_delay`.
3. On each request, `ExampleHandler.generate()`:
   - Encodes "Hello World!" using the model's tokenizer.
   - Streams tokens one at a time, sleeping `token_delay` seconds between each.
   - Monitors for cancellation via `_cancellation_monitor` and stops early if the request is cancelled.
