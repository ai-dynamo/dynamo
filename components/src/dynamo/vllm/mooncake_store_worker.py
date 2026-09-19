# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Worker-only composition; importing gc_policy starts its configured policy."""

from dynamo.vllm.gc_policy import FpmGcWorkerExtension
from dynamo.vllm.mooncake_store_runtime import MooncakeStoreWorkerExtension


class MooncakeStoreFpmWorkerExtension(
    MooncakeStoreWorkerExtension, FpmGcWorkerExtension
):
    """Retain both named RPC interfaces without starting GC in the launcher."""
