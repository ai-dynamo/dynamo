# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ModelExpress v2 mid-training weight refit for Dynamo's vLLM worker.

Port of NeMo-RL's ``VllmInternalWorkerExtension.update_weights_via_mx``
(commit ``d58dca07`` on the NeMo-RL ``kavin/mx_integration`` branch) into the
Dynamo vLLM worker, so external trainers can publish per-rank DTensor shards
via NIXL RDMA into a running ``AsyncLLM`` mid-training. Same wire protocol,
same MX server, same ``modelexpress`` Python client — different host process.
"""

from .extension import MxConfig, MxRefitWorkerExtension

__all__ = ["MxConfig", "MxRefitWorkerExtension"]
