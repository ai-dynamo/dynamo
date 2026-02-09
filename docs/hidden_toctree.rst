:orphan:

..
    SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
    SPDX-License-Identifier: Apache-2.0

.. This hidden toctree includes readmes etc that aren't meant to be in the main table of contents but should be accounted for in the sphinx project structure


.. toctree::
   :maxdepth: 2
   :hidden:

   development/runtime-guide.md
   development/jail_stream.md

   api/nixl_connect/connector.md
   api/nixl_connect/descriptor.md
   api/nixl_connect/device.md
   api/nixl_connect/device_kind.md
   api/nixl_connect/operation_status.md
   api/nixl_connect/rdma_metadata.md
   api/nixl_connect/readable_operation.md
   api/nixl_connect/writable_operation.md
   api/nixl_connect/read_operation.md
   api/nixl_connect/write_operation.md
   api/nixl_connect/README.md

   kubernetes/api_reference.md
   kubernetes/deployment/create_deployment.md
   kubernetes/chrek/dynamo.md
   kubernetes/chrek/standalone.md
   kubernetes/fluxcd.md
   kubernetes/model_caching_with_fluid.md

   reference/cli.md
   reference/glossary.md
   performance/tuning.md

   backends/vllm/deepseek-r1.md
   backends/vllm/gpt-oss.md
   backends/vllm/multi-node.md
   backends/vllm/prometheus.md
   backends/vllm/prompt-embeddings.md

   backends/sglang/expert-distribution-eplb.md
   backends/sglang/gpt-oss.md
   backends/sglang/diffusion-lm.md
   backends/sglang/profiling.md
   backends/sglang/sglang-disaggregation.md
   backends/sglang/prometheus.md

   backends/trtllm/multinode/multinode-examples.md
   backends/trtllm/llama4_plus_eagle.md
   backends/trtllm/kv-cache-transfer.md
   backends/trtllm/gemma3_sliding_window_attention.md
   backends/trtllm/gpt-oss.md
   backends/trtllm/prometheus.md

   features/speculative_decoding/README.md
   features/speculative_decoding/speculative_decoding_vllm.md

   examples/README.md
   examples/runtime/hello_world/README.md

   benchmarks/kv-router-ab-testing.md

   mocker/mocker.md
