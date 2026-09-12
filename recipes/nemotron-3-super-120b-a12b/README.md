# Nemotron-3-Super-120B-A12B Recipe Guide

This guide will help you run the Nemotron-3-Super-120B-A12B language model using Dynamo's optimized setup.

## Prerequisites

Follow the instructions in recipe [README.md](../README.md) to create a namespace and kubernetes secret for huggingface token.

## Quick Start

To run the model, simply execute this command in your terminal:

```bash
cd recipes
./run.sh --model nemotron-3-super-120b-a12b --framework trtllm agg
```

## (Alternative) Step by Step Guide

### 1. Download the Model

```bash
cd recipes/nemotron-3-super-120b-a12b
kubectl apply -n $NAMESPACE -f ./model-cache
```

### 2. Deploy and Benchmark the Model

```bash
cd recipes/nemotron-3-super-120b-a12b
kubectl apply -n $NAMESPACE -f ./trtllm/agg
```

### Container Image
This recipe was tested with dynamo trtllm runtime container.

## Notes
1. storage class is not specified in the recipe, you need to specify it in the `model-cache.yaml` file.