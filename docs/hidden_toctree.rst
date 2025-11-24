:orphan:

..
    SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
    SPDX-License-Identifier: Apache-2.0

.. This hidden toctree includes readmes etc that aren't meant to be in the main table of contents but should be accounted for in the sphinx project structure


.. toctree::
   :maxdepth: 2
   :hidden:

   development/runtime-guide.md
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
   kubernetes/deployment/dynamomodel-guide.md

   kubernetes/fluxcd.md
   kubernetes/grove.md
   kubernetes/model_caching_with_fluid.md
   kubernetes/README.md
   reference/cli.md
   observability/metrics.md
   kvbm/vllm-setup.md
   kvbm/trtllm-setup.md
   agents/tool-calling.md
   guides/jail_stream_readme.md
   guides/request_plane.md

   router/kv_cache_routing.md
   planner/load_planner.md
   fault_tolerance/request_migration.md
   fault_tolerance/request_cancellation.md

   backends/trtllm/multinode/multinode-examples.md
   backends/trtllm/multinode/multinode-multimodal-example.md
   backends/trtllm/llama4_plus_eagle.md
   backends/trtllm/kv-cache-transfer.md
   backends/trtllm/multimodal_support.md
   backends/trtllm/multimodal_epd.md
   backends/trtllm/gemma3_sliding_window_attention.md
   backends/trtllm/gpt-oss.md
   backends/trtllm/prometheus.md

   backends/sglang/multinode-examples.md
   backends/sglang/dsr1-wideep-gb200.md
   backends/sglang/dsr1-wideep-h100.md
   backends/sglang/expert-distribution-eplb.md
   backends/sglang/gpt-oss.md
   backends/sglang/multimodal_epd.md
   backends/sglang/sgl-hicache-example.md
   backends/sglang/sglang-disaggregation.md
   backends/sglang/prometheus.md

   examples/README.md
   examples/runtime/hello_world/README.md

   design_docs/distributed_runtime.md
   design_docs/dynamo_flow.md

   backends/vllm/deepseek-r1.md
   backends/vllm/gpt-oss.md
   backends/vllm/multi-node.md
   backends/vllm/multimodal.md
   backends/vllm/prometheus.md

   benchmarks/kv-router-ab-testing.md

   frontends/kserve.md
   _sections/frontends.rst

..   TODO: architecture/distributed_runtime.md and architecture/dynamo_flow.md
     have some outdated names/references and need a refresh.
..   TODO: Add an OpenAI frontend doc and then add top-level Frontends section
     to index.rst pointing to both OpenAI HTTP and KServe GRPC docs.
