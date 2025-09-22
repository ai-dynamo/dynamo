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

   guides/dynamo_deploy/api_reference.md
   guides/dynamo_deploy/create_deployment.md

   guides/dynamo_deploy/fluxcd.md
   guides/dynamo_deploy/gke_setup.md
   guides/dynamo_deploy/grove.md
   guides/dynamo_deploy/model_caching_with_fluid.md
   guides/dynamo_deploy/README.md
   guides/dynamo_run.md
   guides/dynamo_deploy/sla_planner_deployment.md
   guides/metrics.md
   guides/run_kvbm_in_vllm.md
   guides/run_kvbm_in_trtllm.md

   architecture/kv_cache_routing.md
   architecture/load_planner.md
   architecture/request_migration.md
   architecture/request_cancellation.md

   components/backends/trtllm/multinode/multinode-examples.md
   components/backends/sglang/docs/multinode-examples.md

   examples/README.md
   examples/custom_backend/hello_world/README.md

   architecture/distributed_runtime.md
   architecture/dynamo_flow.md


..   TODO: architecture/distributed_runtime.md and architecture/dynamo_flow.md
     have some outdated names/references and need a refresh.
