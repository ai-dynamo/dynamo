# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Import-hygiene tests for ``dynamo.common.multimodal``.

Importing a media loader from the package must not transitively load
``torch`` or ``vllm`` (issue #11172). Every assertion about a pristine
``sys.modules`` runs in a subprocess, because the ambient pytest session has
almost certainly already imported both.
"""

import subprocess
import sys
import textwrap

import pytest

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


def _run_in_subprocess(script: str) -> None:
    """Run ``script`` in a fresh interpreter, failing the test on nonzero exit."""
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.fail(
            "subprocess failed with exit code "
            f"{result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )


def test_media_loaders_do_not_import_torch_or_vllm() -> None:
    """Importing the media loaders must not drag in torch or vLLM.

    This is the regression guard for issue #11172. Before the lazy-import
    change this failed: ``AsyncEncoderCache`` (via the memory cache manager),
    ``embedding_transfer``, and ``audio_loader``'s module-level vLLM import
    each pulled in torch at package-import time.
    """
    _run_in_subprocess(
        """
        import sys

        from dynamo.common.multimodal import AudioLoader, ImageLoader, VideoLoader

        assert ImageLoader is not None
        assert AudioLoader is not None
        assert VideoLoader is not None

        assert "torch" not in sys.modules, (
            "importing the media loaders pulled in torch: "
            + repr(sorted(m for m in sys.modules if m.startswith("torch")))[:400]
        )
        assert "vllm" not in sys.modules, (
            "importing the media loaders pulled in vllm: "
            + repr(sorted(m for m in sys.modules if m.startswith("vllm")))[:400]
        )
        """
    )


def test_deferred_members_still_resolve_and_load_torch_on_demand() -> None:
    """The heavy members are deferred, not deleted.

    Negative control: every name in ``__all__`` still resolves, and torch
    becomes loaded only once a torch-dependent member is actually touched.
    """
    _run_in_subprocess(
        """
        import sys

        import dynamo.common.multimodal as mm

        assert "torch" not in sys.modules, "package import loaded torch eagerly"

        from dynamo.common.multimodal import AsyncEncoderCache, TransferRequest

        assert AsyncEncoderCache is not None
        assert TransferRequest is not None
        assert "torch" in sys.modules, "torch should load once a heavy member is used"

        missing = [name for name in mm.__all__ if not hasattr(mm, name)]
        assert not missing, f"names in __all__ failed to resolve: {missing}"
        """
    )


def test_embedding_factories_cover_every_transfer_mode_with_stable_identity() -> None:
    """The factory dicts build on demand, stay stable, and cover every mode.

    The sglang multimodal worker handlers index these dicts by
    ``EmbeddingTransferMode``, and callers may hold a reference across
    accesses, so both total coverage and object identity matter.
    """
    _run_in_subprocess(
        """
        import dynamo.common.multimodal as mm
        from dynamo.common.constants import EmbeddingTransferMode

        senders = mm.EMBEDDING_SENDER_FACTORIES
        receivers = mm.EMBEDDING_RECEIVER_FACTORIES

        modes = set(EmbeddingTransferMode)
        assert set(senders) == modes, f"sender factories missing {modes - set(senders)}"
        assert set(receivers) == modes, (
            f"receiver factories missing {modes - set(receivers)}"
        )
        assert all(callable(f) for f in senders.values())
        assert all(callable(f) for f in receivers.values())

        # Cached into globals(): repeat access must return the same objects.
        assert mm.EMBEDDING_SENDER_FACTORIES is senders
        assert mm.EMBEDDING_RECEIVER_FACTORIES is receivers
        """
    )


def test_unknown_attribute_still_raises_attribute_error() -> None:
    """``__getattr__`` must not swallow typos into silence."""
    import dynamo.common.multimodal as mm

    with pytest.raises(AttributeError, match="no attribute 'NotARealSymbol'"):
        mm.NotARealSymbol
