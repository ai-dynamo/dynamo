# llama.cpp engine for Dynamo

Usage:
- `pip install -r requirements.txt` # Need a recent pip, `uv pip` might be too old.
- `python -m dynamo.llama_cpp --model-path /data/models/Qwen3-0.6B-Q8_0.gguf [args]`

## Request Migration

You can enable [request migration](../../../docs/architecture/request_migration.md) to handle worker failures gracefully. Use the `--migration-limit` flag to specify how many times a request can be migrated to another worker:

```bash
python3 -m dynamo.llama_cpp ... --migration-limit=3
```

This allows a request to be migrated up to 3 times before failing. See the [Request Migration Architecture](../../../docs/architecture/request_migration.md) documentation for details on how this works.
