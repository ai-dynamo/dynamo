# Quickstart Guide

## Examples Overview

This directory contains various examples demonstrating different use cases and capabilities of Dynamo. Each example showcases specific features and deployment patterns:

| Example | Description |
|---------|-------------|
| **hello_world** | Basic multi-service pipeline demonstrating core Dynamo concepts and deployment patterns |
| **llm_hello_world** | LLM pipeline with interface-based design |
| **llm** | Comprehensive LLM deployment examples with multiple frameworks |
| **vllm_v0** | vLLM deployment examples using dynamo-run for high-performance processing |
| **vllm_v1** | Latest vLLM deployment examples with advanced features |
| **tensorrt_llm** | TensorRT-LLM deployment examples for optimized inference |
| **sglang** | SGLang backend examples |
| **multimodal** | Multimodal model deployment examples (vision + video) |
| **cli** | Command-line interface examples using Python bindings |
| **router_standalone** | Experimental Standalone KV router implementation |

Each example includes detailed README files with specific setup instructions, configuration options, and usage examples. Choose the example that best matches your use case and requirements.

---

## Deploying examples on Kubernetes

This guide will help you quickly deploy and clean up the dynamo example services in Kubernetes.

### Prerequisites

#### Kubernetes
You have `kubectl` installed and access to your cluster and the correct namespace set in `$NAMESPACE` environment variable.

#### Dynamo Cloud

First, you need to install the Dynamo Cloud Platform. Dynamo Cloud acts as an orchestration layer between the end user and Kubernetes, handling the complexity of deploying your graphs for you.

Before you can deploy your graphs, you need to deploy the Dynamo Cloud. This is a one-time action, only necessary the first time you deploy a DynamoGraph.


### Create a secret with huggingface token

```bash
export HF_TOKEN="huggingfacehub token with read permission to models"
kubectl create secret generic hf-token-secret --from-literal=HF_TOKEN=$HF_TOKEN -n ${NAMESPACE}|| true
```

---

Choose the example you want to deploy or delete. The YAML files are located in `examples/<backend>/deploy/k8s/`.

### Deploy dynamo example

```bash
export INFERENCE_BACKEND=vllm_v1
export EXAMPLE_MANIFEST=agg.yaml
kubectl apply -f examples/$INFERENCE_BACKEND/deploy/k8s/$EXAMPLE_MANIFEST -n $NAMESPACE
```

### Uninstall dynamo example


```bash
export INFERENCE_BACKEND=vllm_v1
export EXAMPLE_MANIFEST=agg.yaml
kubectl delete -f examples/$INFERENCE_BACKEND/deploy/k8s/$EXAMPLE_MANIFEST -n $NAMESPACE
```

### Using a different dynamo container

To customize the container image used in your deployment, you will need to update the manifest before applying it.

You can use [`yq`](https://github.com/mikefarah/yq?tab=readme-ov-file#install), a portable command-line YAML processor.

Please follow the [installation instructions](https://github.com/mikefarah/yq?tab=readme-ov-file#install) for your platform if you do not already have `yq` installed. After installing `yq`, you can generate and apply your manifest as follows:

1. prepare your custom manifest file
```bash
export DYNAMO_IMAGE=my-registry/my-image:tag
yq '.spec.services.[].extraPodSpec.mainContainer.image = env(DYNAMO_IMAGE)' $EXAMPLE_FILE > my_example_manifest.yaml
```

2. install the dynamo example
```bash
kubectl apply -f my_example_manifest.yaml -n $NAMESPACE
```

3. uninstall the dynamo example
```bash
kubectl delete -f my_example_manifest.yaml -n $NAMESPACE
```