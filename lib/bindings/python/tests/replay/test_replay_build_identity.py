# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

import dynamo.replay as replay
from dynamo import _core


pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.parallel,
    pytest.mark.pre_merge,
    pytest.mark.unit,
]


def test_native_build_identity_is_self_consistent() -> None:
    identity = _core.build_identity()

    assert identity["source_revision"] == _core.__source_revision__
    assert replay.__dynamo_source_commit__ == _core.__source_revision__
    assert identity["package_version"] == _core.__version__
    assert identity["source_revision"] == "unknown" or (
        len(identity["source_revision"]) == 40
        and all(character in "0123456789abcdef" for character in identity["source_revision"])
    )
    assert identity["cargo_features"] == sorted(identity["cargo_features"])
    assert len(identity["cargo_features"]) == len(set(identity["cargo_features"]))
    assert isinstance(identity["default_features"], bool)
    assert identity["build_profile"]
