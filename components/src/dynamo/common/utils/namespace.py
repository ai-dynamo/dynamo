# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os


def get_worker_namespace(default="dynamo"):
    """Get the Dynamo namespace for a worker.

    If DYN_NAMESPACE_WORKER_SUFFIX is set, the namespace becomes
    "{DYN_NAMESPACE}-{DYN_NAMESPACE_WORKER_SUFFIX}". This enables supporting
    multiple sets of workers for the same model.
    """
    namespace = os.environ.get("DYN_NAMESPACE", default)
    suffix = os.environ.get("DYN_NAMESPACE_WORKER_SUFFIX")
    if suffix:
        namespace = f"{namespace}-{suffix}"
    return namespace
