# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Every declared backend value must yield a runtime container image.

`BackendType` is the public enumeration a DynamoGraphDeploymentRequest sets
its `backend:` field to. `derive_backend_image` maps a concrete backend to its
published runtime image name and raises for anything else, so the automatic
value has to be resolved to a concrete backend before an image is derived.
Every search path must therefore be able to resolve it.

This module imports only `dynamo.profiler.utils.profile_common` and the DGDR
types, so it does not depend on the AIC simulation package that
`dynamo.profiler.rapid` pulls in. It covers the backend enumeration rather
than image substitution mechanics, which `test_planner_image_selection.py`
already covers.
"""

import pytest

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.unit,
    pytest.mark.planner,
]

try:
    from dynamo.profiler.utils.dgdr_v1beta1_types import BackendType
    from dynamo.profiler.utils.profile_common import (
        BACKEND_IMAGE_NAMES,
        derive_backend_image,
        resolve_auto_backend,
    )
except ImportError as e:  # pragma: no cover - environment guard
    pytest.skip(f"Skip (missing dependency): {e}", allow_module_level=True)


_REQUEST_IMAGE = "nvcr.io/nvidia/ai-dynamo/dynamo-planner:1.2.3"
_CALLER = "this test"


def _image_name(image_ref: str) -> str:
    return image_ref.rsplit("/", 1)[-1].split(":")[0]


def test_every_declared_backend_yields_a_runtime_image():
    """No value of `BackendType` may reach `derive_backend_image` unresolved."""
    published = set(BACKEND_IMAGE_NAMES.values())

    derived = {
        backend: derive_backend_image(
            _REQUEST_IMAGE, resolve_auto_backend(backend.value, _CALLER)
        )
        for backend in BackendType
    }

    assert set(derived) == set(BackendType)
    for backend, image in derived.items():
        assert _image_name(image) in published, f"{backend.value} -> {image}"


def test_a_requested_backend_is_never_overridden():
    """Resolution applies to the automatic value only; an explicit request
    must reach the search path unchanged."""
    for backend in (BackendType.Sglang, BackendType.Trtllm, BackendType.Vllm):
        assert resolve_auto_backend(backend.value, _CALLER) == backend.value
