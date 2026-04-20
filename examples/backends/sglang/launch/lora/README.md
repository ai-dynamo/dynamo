# LoRA with SGLang Backend

For the full LoRA integration guide (setup, usage, API reference, troubleshooting), see [the shared LoRA guide](../../../../common/lora.md).

## Quick Start

```bash
./setup_minio.sh    # Start MinIO, download & upload LoRA
./agg_lora.sh       # Launch SGLang frontend + worker with LoRA
```

## SGLang-Specific Notes

- The launch script uses `--lora-target-modules all` and `--max-lora-rank 64` by default
- Override with environment variables: `MODEL`, `LORA_NAME`, `DYN_SYSTEM_PORT`, `DYN_HTTP_PORT`
- SGLang LoRA loading goes through `engine.tokenizer_manager.load_lora_adapter()`
