# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from dynamo.llm import KvRouterConfig


def test_removed_router_options_cannot_shift_positional_arguments() -> None:
    with pytest.raises(TypeError):
        KvRouterConfig(None, 0.75, 0.25, 0.0, True, False)
