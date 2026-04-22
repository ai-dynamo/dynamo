# EKS deployment examples

This directory contains AWS EKS-specific deployment guides and manifests for Dynamo.

## Trainium + CUDA disaggregated vLLM examples

The following manifests show how to run Trainium prefill and CUDA decode workers with KV-cache transfer over EFA using NIXL LIBFABRIC:

- `manifests/vllm/disagg_trainium_cuda_efa.yaml` - Trainium prefill stages KV cache through CPU DRAM before transfer to the CUDA decode worker.
- `manifests/vllm/disagg_trainium_cuda_efa_cpu.yaml` - Both workers use CPU KV buffers while transferring over EFA.
- `manifests/vllm/disagg_trainium_cuda_efa_gpudirect.yaml` - CUDA decode receives KV cache directly into GPU memory.

These examples target mixed `trn1n.32xlarge` and `p5.48xlarge` node groups with EFA enabled.
