# GPT-OSS-120B Recipe Guide

This guide will help you run the GPT-OSS-120B language model using Dynamo's optimized setup.

## Prerequisites

Follow the instructions in recipe [README.md](../README.md) to create a namespace and kubernetes secret for huggingface token.

## Quick Start

To run the model, simply execute this command in your terminal:

```bash
cd recipe
./run.sh --model gpt-oss-120b --framework trtllm agg
```

## (Alternative) Step by Step Guide

### 1. Download the Model

```bash
cd recipes/gpt-oss-120b
kubectl apply -n $NAMESPACE -f ./model-cache
```

### 2. Deploy and Benchmark the Model

```bash
cd recipes/gpt-oss-120b
kubectl apply -n $NAMESPACE -f ./trtllm/agg
```

### Container Image
This recipe was tested with dynamo trtllm runtime container for ARM64 processors.

**Important Note:**

Before dynamo v0.5.1 release, following container image is supported:
```
nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime:0.5.1-rc0.pre3
```

After dynamo v0.5.1 release, following container image will be supported:
```
nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime:0.5.1
```

## Notes
1. The benchmark container image uses a specific commit of aiperf to ensure reproducible results and compatibility with the benchmarking setup.

2. storage class is not specified in the recipe, you need to specify it in the `deploy.yaml` file.