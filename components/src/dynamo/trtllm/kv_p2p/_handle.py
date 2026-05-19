# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Source-side handle returned from remote-G2 setup helpers.

Holds the SourceG2DescriptorRegistry used for lease/TTL/refcount
bookkeeping. Setup flavors (alpha event-driven, beta direct-query) both
return one of these; setup-flavor-specific state (e.g. the alpha event
adapter) is kept alive externally and does not appear on the handle.
"""

from __future__ import annotations

from dataclasses import dataclass

from tensorrt_llm._torch.pyexecutor.connectors.remote_g2 import (
    SourceG2DescriptorRegistry,
)


@dataclass
class RemoteG2SourceHandle:
    registry: SourceG2DescriptorRegistry
