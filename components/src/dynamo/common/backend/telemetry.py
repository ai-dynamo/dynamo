# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Back-compat shim -> `dynamo.backend.telemetry`.

`dynamo.common.backend` moved to `dynamo.backend`; import from there instead.
Removed in the follow-up migration commit.
"""
import sys as _sys

import dynamo.backend.telemetry as _target

_sys.modules[__name__] = _target
