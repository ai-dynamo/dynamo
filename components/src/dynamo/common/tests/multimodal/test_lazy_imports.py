# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Import-hygiene tests for dynamo.common.multimodal (issue #11172).

The media loaders must be importable without pulling in torch; the
embedding-transfer / encoder-cache members resolve lazily via PEP 562.
The torch-free checks run in a subprocess so the result is independent of
whatever the current pytest session has already imported.
"""

import subprocess
import sys

import pytest

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
]


def _run_in_subprocess(code: str) -> None:
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert (
        result.returncode == 0
    ), f"subprocess failed\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"


def test_package_import_does_not_import_torch():
    """Importing the package itself must not pull in torch or vllm."""
    _run_in_subprocess(
        "import sys\n"
        "import dynamo.common.multimodal\n"
        "assert 'torch' not in sys.modules, 'package import pulled in torch'\n"
        "assert 'vllm' not in sys.modules, 'package import pulled in vllm'\n"
    )


def test_image_loader_import_does_not_import_torch():
    """The reported case: ImageLoader without torch (issue #11172)."""
    _run_in_subprocess(
        "import sys\n"
        "from dynamo.common.multimodal import ImageLoader\n"
        "from dynamo.common.multimodal import AudioLoader, VideoLoader\n"
        "assert 'torch' not in sys.modules, 'loader import pulled in torch'\n"
        "assert 'vllm' not in sys.modules, 'loader import pulled in vllm'\n"
    )


def test_lazy_members_resolve_and_torch_loads_on_access():
    """Lazy members still resolve via `from` imports, and only then load torch."""
    _run_in_subprocess(
        "import sys\n"
        "import dynamo.common.multimodal as mm\n"
        "assert 'torch' not in sys.modules\n"
        "from dynamo.common.multimodal import AsyncEncoderCache\n"
        "from dynamo.common.multimodal import TransferRequest\n"
        "assert 'torch' in sys.modules, 'lazy member access should load torch'\n"
        "assert AsyncEncoderCache.__name__ == 'AsyncEncoderCache'\n"
        "assert TransferRequest is not None\n"
    )


def test_factory_dicts_resolve_lazily_with_stable_identity():
    """Factory dicts build on first access, keep identity, and stay consistent."""
    _run_in_subprocess(
        "import sys\n"
        "import dynamo.common.multimodal as mm\n"
        "assert 'torch' not in sys.modules\n"
        "from dynamo.common.constants import EmbeddingTransferMode\n"
        "senders = mm.EMBEDDING_SENDER_FACTORIES\n"
        "receivers = mm.EMBEDDING_RECEIVER_FACTORIES\n"
        "assert senders is mm.EMBEDDING_SENDER_FACTORIES, 'dict identity not cached'\n"
        "assert receivers is mm.EMBEDDING_RECEIVER_FACTORIES\n"
        "assert set(senders) == set(EmbeddingTransferMode)\n"
        "assert set(receivers) == set(EmbeddingTransferMode)\n"
    )


def test_unknown_attribute_raises_attribute_error():
    with pytest.raises(AttributeError, match="has no attribute"):
        import dynamo.common.multimodal as mm

        _ = mm.definitely_not_a_real_attribute
