:orphan:

..
    SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
   kubernetes/chrek/README.md
   kubernetes/chrek/dynamo.md
   kubernetes/chrek/standalone.md

   kubernetes/fluxcd.md
   kubernetes/grove.md
   kubernetes/model_caching_with_fluid.md
   kubernetes/README.md
   reference/cli.md
   observability/metrics.md
   integrations/kv_events_custom_engines.md
   agents/tool-calling.md
   development/jail_stream.md

   components/planner/README.md
   components/planner/planner_guide.md
   components/planner/planner_examples.md
   components/kvbm/README.md
   components/kvbm/kvbm_guide.md
   components/router/README.md
   components/router/router_guide.md
   components/router/router_examples.md
   components/frontend/frontend_guide.md
   design_docs/kvbm_design.md
   integrations/flexkv_integration.md
   integrations/sglang_hicache.md
   fault_tolerance/README.md
   fault_tolerance/request_migration.md
   fault_tolerance/request_cancellation.md
   fault_tolerance/graceful_shutdown.md
   fault_tolerance/request_rejection.md
   fault_tolerance/testing.md
   design_docs/request_plane.md
   design_docs/event_plane.md

   backends/trtllm/multinode/multinode-examples.md
   backends/trtllm/llama4_plus_eagle.md
   backends/trtllm/kv-cache-transfer.md
   backends/trtllm/gemma3_sliding_window_attention.md
   backends/trtllm/gpt-oss.md
   backends/trtllm/prometheus.md

   backends/sglang/expert-distribution-eplb.md
   backends/sglang/gpt-oss.md
   backends/sglang/diffusion-lm.md
   backends/sglang/profiling.md
   backends/sglang/sglang-disaggregation.md
   backends/sglang/prometheus.md

   examples/README.md
   examples/runtime/hello_world/README.md

   design_docs/distributed_runtime.md
   design_docs/dynamo_flow.md

   backends/vllm/deepseek-r1.md
   backends/vllm/gpt-oss.md
   integrations/lmcache_integration.md
   backends/vllm/multi-node.md
   backends/vllm/prometheus.md
   backends/vllm/prompt-embeddings.md

   features/speculative_decoding/README.md
   features/speculative_decoding/speculative_decoding_vllm.md

   benchmarks/kv-router-ab-testing.md

   mocker/mocker.md

..   TODO: architecture/distributed_runtime.md and architecture/dynamo_flow.md
     have some outdated names/references and need a refresh.
