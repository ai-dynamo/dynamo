# CLI Flag and Env Var Migration Table

This document summarizes Dynamo 1.0 config naming changes, grouped by component.

## Shared Runtime (vLLM/SGLang/TRT-LLM)


| Old flag               | New flag              | Old env var    | New env var             |
| ---------------------- | --------------------- | -------------- | ----------------------- |
| `--store-kv`           | `--discovery-backend` | `DYN_STORE_KV` | `DYN_DISCOVERY_BACKEND` |
| `--dyn-endpoint-types` | `--endpoint-types`    | --             | `DYN_ENDPOINT_TYPES`    |
| `--endpoint`           | `--endpoint`          | --             | `DYN_ENDPOINT`          |


## vLLM


| Old flag                             | New flag             | Old env var                                 | New env var                 |
| ------------------------------------ | -------------------- | ------------------------------------------- | --------------------------- |
| `--multimodal-processor`             | `--route-to-encoder` | `DYN_VLLM_MULTIMODAL_PROCESSOR`             | `DYN_VLLM_ROUTE_TO_ENCODER` |
| `--ec-processor`                     | `--route-to-encoder` | `DYN_VLLM_EC_PROCESSOR`                     | `DYN_VLLM_ROUTE_TO_ENCODER` |
| `--multimodal-encode-prefill-worker` | (removed)            | `DYN_VLLM_MULTIMODAL_ENCODE_PREFILL_WORKER` | (removed)                   |
| `--vllm-native-encoder-worker`       | (removed)            | `DYN_VLLM_NATIVE_ENCODER_WORKER`            | (removed)                   |
| `--ec-connector-backend`             | (removed)            | `DYN_VLLM_EC_CONNECTOR_BACKEND`             | (removed)                   |
| `--ec-storage-path`                  | (removed)            | `DYN_VLLM_EC_STORAGE_PATH`                  | (removed)                   |
| `--ec-extra-config`                  | (removed)            | `DYN_VLLM_EC_EXTRA_CONFIG`                  | (removed)                   |
| `--ec-consumer-mode`                 | (removed)            | `DYN_VLLM_EC_CONSUMER_MODE`                 | (removed)                   |


## SGLang


| Old flag                                 | New flag              | Old env var | New env var                 |
| ---------------------------------------- | --------------------- | ----------- | --------------------------- |
| `--config` (Dynamo disagg selector flow) | `--disagg-config`     | --          | `DYN_SGL_DISAGG_CONFIG`     |
| `--config-key`                           | `--disagg-config-key` | --          | `DYN_SGL_DISAGG_CONFIG_KEY` |


## TRT-LLM


| Old flag                          | New flag                                   | Old env var | New env var                                  |
| --------------------------------- | ------------------------------------------ | ----------- | -------------------------------------------- |
| `--model-path`                    | `--model`                                  | --          | `DYN_TRTLLM_MODEL`                           |
| `--dyn-encoder-cache-capacity-gb` | `--multimodal-embedding-cache-capacity-gb` | --          | `DYN_MULTIMODAL_EMBEDDING_CACHE_CAPACITY_GB` |
| `--use-nixl-connect`              | (removed; use `--connector nixl`)          | --          | --                                           |


## Frontend


| Old flag                         | New flag                                       | Old env var                      | New env var                          |
| -------------------------------- | ---------------------------------------------- | -------------------------------- | ------------------------------------ |
| `--kv-overlap-score-weight`      | `--router-kv-overlap-score-weight`             | `DYN_KV_OVERLAP_SCORE_WEIGHT`    | `DYN_ROUTER_KV_OVERLAP_SCORE_WEIGHT` |
| `--kv-events` / `--no-kv-events` | `--router-kv-events` / `--no-router-kv-events` | `DYN_KV_EVENTS`                  | `DYN_ROUTER_USE_KV_EVENTS`           |
| `--router-ttl`                   | `--router-ttl-secs`                            | `DYN_ROUTER_TTL`                 | `DYN_ROUTER_TTL_SECS`                |
| `--durable-kv-events`            | `--router-durable-kv-events`                   | `DYN_DURABLE_KV_EVENTS`          | `DYN_ROUTER_DURABLE_KV_EVENTS`       |
| `--track-active-blocks`          | `--router-track-active-blocks`                 | `DYN_TRACK_ACTIVE_BLOCKS`        | `DYN_ROUTER_TRACK_ACTIVE_BLOCKS`     |
| `--assume-kv-reuse`              | `--router-assume-kv-reuse`                     | `DYN_ASSUME_KV_REUSE`            | `DYN_ROUTER_ASSUME_KV_REUSE`         |
| `--track-output-blocks`          | `--router-track-output-blocks`                 | `DYN_ROUTER_TRACK_OUTPUT_BLOCKS` | `DYN_ROUTER_TRACK_OUTPUT_BLOCKS`     |


## Router


| Old flag                                             | New flag                                                           | Old env var | New env var                          |
| ---------------------------------------------------- | ------------------------------------------------------------------ | ----------- | ------------------------------------ |
| `--block-size`                                       | `--router-block-size`                                              | --          | `DYN_ROUTER_BLOCK_SIZE`              |
| `--kv-overlap-score-weight`                          | `--router-kv-overlap-score-weight`                                 | --          | `DYN_ROUTER_KV_OVERLAP_SCORE_WEIGHT` |
| `--kv-events` / `--no-kv-events`                     | `--router-kv-events` / `--no-router-kv-events`                     | --          | `DYN_ROUTER_USE_KV_EVENTS`           |
| `--durable-kv-events`                                | `--router-durable-kv-events`                                       | --          | `DYN_ROUTER_DURABLE_KV_EVENTS`       |
| `--track-active-blocks` / `--no-track-active-blocks` | `--router-track-active-blocks` / `--no-router-track-active-blocks` | --          | `DYN_ROUTER_TRACK_ACTIVE_BLOCKS`     |
| `--assume-kv-reuse` / `--no-assume-kv-reuse`         | `--router-assume-kv-reuse` / `--no-router-assume-kv-reuse`         | --          | `DYN_ROUTER_ASSUME_KV_REUSE`         |
| `--track-output-blocks`                              | `--router-track-output-blocks`                                     | --          | `DYN_ROUTER_TRACK_OUTPUT_BLOCKS`     |


## Global Router


| Old flag                | New flag                | Old env var | New env var                             |
| ----------------------- | ----------------------- | ----------- | --------------------------------------- |
| `--config`              | `--config`              | --          | `DYN_GLOBAL_ROUTER_CONFIG`              |
| `--model-name`          | `--model-name`          | --          | `DYN_GLOBAL_ROUTER_MODEL_NAME`          |
| `--component-name`      | `--component-name`      | --          | `DYN_GLOBAL_ROUTER_COMPONENT_NAME`      |
| `--default-ttft-target` | `--default-ttft-target` | --          | `DYN_GLOBAL_ROUTER_DEFAULT_TTFT_TARGET` |
| `--default-itl-target`  | `--default-itl-target`  | --          | `DYN_GLOBAL_ROUTER_DEFAULT_ITL_TARGET`  |


## Rust-side env vars


| Old env var          | New env var              |
| -------------------- | ------------------------ |
| `ENABLE_KVBM_RECORD` | `DYN_KVBM_ENABLE_RECORD` |
| `DYNAMO_FATBIN_PATH` | `DYN_FATBIN_PATH`        |


## Notes

- Old flags marked obsolete in code (`obsolete_flag`) are still accepted for compatibility where configured.
- Some rows are migrations from flag-only behavior to env-backed configuration, so one side may be `--`.
- Some rows may have new flags the same as old flags when adding env-backed configuration.

