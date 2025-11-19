# GPT-OSS-120B Disaggregated Mode

> **⚠️ INCOMPLETE**: This directory contains only engine configuration files and is not ready for Kubernetes deployment.

## Current Status

This directory contains TensorRT-LLM engine configurations for disaggregated serving:
- `decode.yaml` - Decode worker engine configuration
- `prefill.yaml` - Prefill worker engine configuration

## Missing Components

To complete this recipe, the following files are needed:
- `deploy.yaml` - Kubernetes DynamoGraphDeployment manifest
- `perf.yaml` - Performance benchmarking job (optional)

## Alternative

For a production-ready GPT-OSS-120B deployment, use the **aggregated mode**:
- [gpt-oss-120b/trtllm/agg/](../agg/) - Complete with `deploy.yaml` and `perf.yaml`

## Contributing

If you'd like to complete this recipe, see [recipes/CONTRIBUTING.md](../../../CONTRIBUTING.md) for guidelines on creating proper Kubernetes deployment manifests.

