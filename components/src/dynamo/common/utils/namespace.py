# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os


def get_namespace(default="dynamo"):
    """Get the effective Dynamo namespace for a worker.

    If DYN_NAMESPACE_WORKER_SUFFIX is set, the effective namespace becomes
    "{DYN_NAMESPACE}-{suffix}". This enables rolling updates where different
    worker versions register in different namespaces.
    """
    namespace = os.environ.get("DYN_NAMESPACE", default)
    suffix = os.environ.get("DYN_NAMESPACE_WORKER_SUFFIX")
    if suffix:
        namespace = f"{namespace}-{suffix}"
    return namespace
