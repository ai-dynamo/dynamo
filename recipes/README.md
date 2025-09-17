# Dynamo model serving recipes

| Model family  | Backend | Mode                | Deployment | Benchmark |
|---------------|---------|---------------------|------------|-----------|
| llama-3-70b   | vllm    | agg                 |     ✓      |     ✓     |
| llama-3-70b   | vllm    | disagg-multi-node   |     ✓      |     ✓     |
| llama-3-70b   | vllm    | disagg-single-node  |     ✓      |     ✓     |
| oss-gpt       | trtllm  | aggregated          |     ✓      |     ✓     |
| DeepSeek-R1   | sglang  | disaggregated       |     🚧     |    🚧     |


## Prerequisites

1. Create a namespace and populate NAMESPACE environment variable
This environment variable is used in later steps to deploy and perf-test the model.

```bash
export NAMESPACE=your-namespace
kubectl create namespace ${NAMESPACE}
```

2. **Dynamo Cloud Platform installed** - Follow [Quickstart Guide](../docs/guides/dynamo_deploy/README.md)

3. **Kubernetes cluster with GPU support**

4. **Container registry access** for vLLM runtime images

5. **HuggingFace token secret** (referenced as `envFromSecret: hf-token-secret`)
Update the `hf-token-secret.yaml` file with your HuggingFace token.

```bash
kubectl apply -f hf_hub_secret/hf_hub_secret.yaml -n ${NAMESPACE}
```

6. (Optional) Create a shared model cache pvc to store the model weights.
Choose a storage class to create the model cache pvc. You'll need to use this storage class name to update the `storageClass` field in the model-cache/model-cache.yaml file.

```bash
kubectl get storageclass
```

## Running the recipes

Run the recipe to deploy a model:

```bash
./run.sh --model <model> --framework <framework> <deployment-type>
```

Arguments:
  <deployment-type>  Deployment type (e.g., agg, disagg-single-node, disagg-multi-node)

Required Options:
  --model <model>    Model name (e.g., llama-3-70b)
  --framework <fw>   Framework one of VLLM TRTLLM SGLANG (default: VLLM)

Optional:
  --skip-model-cache Skip model downloading (assumes model cache already exists)
  -h, --help         Show this help message

Environment Variables:
  NAMESPACE          Kubernetes namespace (default: dynamo)

Examples:
  ./run.sh --model llama-3-70b --framework vllm agg
  ./run.sh --skip-model-cache --model llama-3-70b --framework vllm agg
  ./run.sh --model llama-3-70b --framework trtllm disagg-single-node
Example:
```bash
./run.sh --model llama-3-70b --framework vllm --deployment-type agg
```


## Dry run mode

To dry run the recipe, add the `--dry-run` flag.

```bash
./run.sh --dry-run --model llama-3-70b --framework vllm agg
```

## (Optional) Running the recipes with model cache
You may need to cache the model weights on a PVC to avoid repeated downloads of the model weights.
 See the [Prerequisites](#prerequisites) section for more details.

```bash
./run.sh --model llama-3-70b --framework vllm --deployment-type agg --skip-model-cache
```
