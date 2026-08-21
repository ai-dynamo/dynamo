# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dynamo import _core


def test_native_response_egress_replaces_ownership_based_api_name():
    assert hasattr(_core, "NativeResponseEgress")
    assert not hasattr(_core, "OwnedTokenEgress")
