# FlexKV Integration in Dynamo

## Introduction

[FlexKV](https://github.com/taco-project/FlexKV) is a scalable, distributed runtime for KV cache offloading. It acts as a unified KV caching layer for inference engines like vLLM, TensorRT-LLM and SGLang.

## Usage

Enable FlexKV by setting the `DYNAMO_USE_FLEXKV` environment variable:

```bash
export DYNAMO_USE_FLEXKV=1
```

### Aggregated Serving

Use FlexKV with the `--connector flexkv` flag:

```bash
python -m dynamo.vllm --model $YOUR_MODEL --connector flexkv
```

Refer to [`agg_flexkv.sh`](../../../examples/backends/vllm/launch/agg_flexkv.sh) for quick setup.

### Aggregated Serving with Peer Node KV Cache Reuse

Refer to our project [README](https://github.com/taco-project/FlexKV/blob/main/docs/dist_reuse/README_en.md) for instructions on setting up peer KV cache reuse.

### Disaggregated Serving

Refer to [`disagg_flexkv.sh`](../../../examples/backends/vllm/launch/disagg_flexkv.sh) for quick setup.
