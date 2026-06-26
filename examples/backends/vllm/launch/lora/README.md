# LoRA with vLLM Backend

For the full LoRA integration guide (setup, usage, API reference, troubleshooting), see [the shared LoRA guide](../../../../common/lora.md).

## Quick Start

```bash
./setup_minio.sh    # Start MinIO, download & upload LoRA
./agg_lora.sh       # Launch vLLM frontend + worker with LoRA
```

## vLLM-Specific Notes

- Default `--max-lora-rank 64` (same as SGLang)
- Override with environment variables: `MODEL`, `LORA_NAME`, `MAX_MODEL_LEN`, `MAX_CONCURRENT_SEQS`

### KV-Aware Routing (2 GPUs)

```bash
./agg_lora_router.sh
```

Launches two vLLM workers behind a KV-aware router. Load the LoRA to both workers (ports 8081 and 8082), then requests are routed with KV cache affinity for better cache hit rates.
