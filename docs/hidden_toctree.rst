:orphan:

..
    SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
    SPDX-License-Identifier: Apache-2.0

.. This hidden toctree includes readmes etc that aren't meant to be in the main table of contents but should be accounted for in the sphinx project structure


.. toctree::
   :maxdepth: 2
   :hidden:

   runtime/README.md
   API/nixl_connect/connector.md
   API/nixl_connect/descriptor.md
   API/nixl_connect/device.md
   API/nixl_connect/device_kind.md
   API/nixl_connect/operation_status.md
   API/nixl_connect/rdma_metadata.md
   API/nixl_connect/readable_operation.md
   API/nixl_connect/writable_operation.md
   API/nixl_connect/read_operation.md
   API/nixl_connect/write_operation.md
   API/nixl_connect/README.md

   kubernetes/api_reference.md
   kubernetes/create_deployment.md

   kubernetes/fluxcd.md
   kubernetes/gke_setup.md
   kubernetes/grove.md
   kubernetes/model_caching_with_fluid.md
   kubernetes/README.md
   guides/dynamo_run.md
   guides/metrics.md
   guides/run_kvbm_in_vllm.md
   guides/run_kvbm_in_trtllm.md
   guides/tool_calling.md

   architecture/kv_cache_routing.md
   architecture/load_planner.md
   architecture/request_migration.md
   architecture/request_cancellation.md

   backends/trtllm/multinode/multinode-examples.md
   backends/trtllm/multinode/multinode-multimodal-example.md
   backends/trtllm/llama4_plus_eagle.md
   backends/trtllm/kv-cache-transfer.md
   backends/trtllm/multimodal_support.md
   backends/trtllm/multimodal_epd.md
   backends/trtllm/gemma3_sliding_window_attention.md
   backends/trtllm/gpt-oss.md

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

   architecture/distributed_runtime.md
   architecture/dynamo_flow.md

   backends/vllm/deepseek-r1.md
   backends/vllm/gpt-oss.md
   backends/vllm/multi-node.md
   backends/vllm/prometheus.md


..   TODO: architecture/distributed_runtime.md and architecture/dynamo_flow.md
     have some outdated names/references and need a refresh.
