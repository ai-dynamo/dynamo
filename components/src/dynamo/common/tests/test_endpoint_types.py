# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from dynamo.common.utils.endpoint_types import parse_endpoint_types
from dynamo.llm import ModelType

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]


def test_internal_endpoint_type_has_no_openai_surface() -> None:
    assert parse_endpoint_types("internal") == ModelType.Empty


@pytest.mark.parametrize("value", ["internal,chat", "completions,internal"])
def test_internal_endpoint_type_cannot_be_combined(value: str) -> None:
    with pytest.raises(ValueError, match="cannot be combined"):
        parse_endpoint_types(value)
