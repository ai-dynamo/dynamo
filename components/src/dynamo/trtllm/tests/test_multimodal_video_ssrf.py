# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SSRF guard for the TRT-LLM `video_url` path.

PR #11896 only rejected `""` / `file://` schemes before handing the URL
straight to `async_load_video`, leaving the worker free to fetch private
links (169.254/16, 10/8, `kubernetes.default`, ...) when the default
`DYN_MM_ALLOW_INTERNAL=0` policy was in effect — unlike the sibling
`image_url` path which goes through `validate_media_url`. These tests
pin the closed gap and the unchanged good-path behavior. They stub the
`tensorrt_llm` modules so they run on CPU-only CI without TRT-LLM.
"""

import importlib
import sys
import types
from unittest.mock import AsyncMock, MagicMock

import pytest

# Stub `tensorrt_llm` so importing `multimodal_processor` works without the
# real TRT-LLM install. `async_load_video` is an AsyncMock we can reset
# between tests to inspect / assert call args (single source of truth for
# whether the worker would actually fetch).
async_load_video_mock = AsyncMock(return_value=object())


def _install_trtllm_stubs() -> None:
    pkg = types.ModuleType("tensorrt_llm")
    pkg.__path__ = []  # mark as a package
    inputs = types.ModuleType("tensorrt_llm.inputs")
    inputs.__path__ = []
    inputs_utils = types.ModuleType("tensorrt_llm.inputs.utils")
    inputs_utils.async_load_video = async_load_video_mock
    inputs_create = types.ModuleType("tensorrt_llm.inputs.create_input_processor")
    # `from tensorrt_llm.inputs import create_input_processor` inside __init__
    inputs.create_input_processor = lambda *a, **kw: None
    llmapi = types.ModuleType("tensorrt_llm.llmapi")
    llmapi.__path__ = []
    llmapi_tokenizer = types.ModuleType("tensorrt_llm.llmapi.tokenizer")
    llmapi_tokenizer.tokenizer_factory = lambda *a, **kw: MagicMock()

    for name, mod in [
        ("tensorrt_llm", pkg),
        ("tensorrt_llm.inputs", inputs),
        ("tensorrt_llm.inputs.utils", inputs_utils),
        ("tensorrt_llm.inputs.create_input_processor", inputs_create),
        ("tensorrt_llm.llmapi", llmapi),
        ("tensorrt_llm.llmapi.tokenizer", llmapi_tokenizer),
    ]:
        sys.modules.setdefault(name, mod)


_install_trtllm_stubs()

from dynamo.common.http import HttpStatusError  # noqa: E402
from dynamo.common.http.url_validator import UrlValidationError  # noqa: E402
from dynamo.trtllm import multimodal_processor as mmp  # noqa: E402

pytestmark = [
    pytest.mark.unit,
    pytest.mark.trtllm,
    pytest.mark.multimodal,
    pytest.mark.pre_merge,
    # CPU-only test: `gpu_0` is the pre-merge CPU stage selector in
    # .github/workflows/pr.yaml (line 703: "pre_merge and trtllm and gpu_0").
    pytest.mark.gpu_0,
]


def _reload_processor(monkeypatch):
    """Reload `mmp` so `UrlValidationPolicy.from_env()` re-reads the env
    with the per-test monkeypatched vars, then force-bind the video fetcher
    to our AsyncMock. The bind matters: a real `tensorrt_llm.inputs.utils`
    can already live in `sys.modules` if another test in the same process
    imported TRT-LLM first, in which case `sys.modules.setdefault` in
    `_install_trtllm_stubs` no longer wins and the module-level
    `from ... import async_load_video` would resolve to the real loader —
    the opt-in test would then make a real network request. Patching the
    module name post-reload keeps every test hermetic."""
    importlib.reload(mmp)
    monkeypatch.setattr(mmp, "async_load_video", async_load_video_mock)


def _make_processor():
    """Build a processor with a mock tokenizer so `__init__` skips
    `tokenizer_factory` and the (irrelevant) input-processor path. Reads
    the class off the (possibly reloaded) module so `importlib.reload`
    between tests always uses the fresh class + its updated policy."""
    return mmp.MultimodalRequestProcessor(
        model_type="multimodal",
        model_dir="unused",
        max_file_size_mb=10,
        tokenizer=MagicMock(),
    )


def _video_request(url: str) -> dict:
    return {"multi_modal_data": {"video_url": [{"Url": url}]}}


@pytest.fixture(autouse=True)
def _reset_async_load_video():
    async_load_video_mock.reset_mock()
    yield
    async_load_video_mock.reset_mock()


@pytest.mark.asyncio
async def test_http_internal_ip_rejected_by_default(monkeypatch):
    """Link-local metadata IP over http:// is blocked by policy, never
    reaches `async_load_video`. Would-be SSRF payload: 169.254.169.254."""
    monkeypatch.delenv("DYN_MM_ALLOW_INTERNAL", raising=False)
    monkeypatch.delenv("DYN_MM_LOCAL_PATH", raising=False)
    # Reload the module so `UrlValidationPolicy.from_env()` re-reads the env
    # with the defaults restored.
    _reload_processor(monkeypatch)

    processor = _make_processor()
    request = _video_request("http://169.254.169.254/latest/meta-data/")

    with pytest.raises(HttpStatusError) as exc:
        await processor.process_openai_request(
            request, embeddings=None, ep_disaggregated_params=None
        )
    assert exc.value.status == 400
    async_load_video_mock.assert_not_called()


@pytest.mark.asyncio
async def test_https_internal_ip_rejected_by_default(monkeypatch):
    """Even with https://, an IP literal in the blocked range (RFC1918
    10/8) is rejected by `is_blocked_ip` before fetch."""
    monkeypatch.delenv("DYN_MM_ALLOW_INTERNAL", raising=False)
    monkeypatch.delenv("DYN_MM_LOCAL_PATH", raising=False)
    _reload_processor(monkeypatch)

    processor = _make_processor()
    request = _video_request("https://10.0.0.1/x.mp4")

    with pytest.raises(HttpStatusError) as exc:
        await processor.process_openai_request(
            request, embeddings=None, ep_disaggregated_params=None
        )
    assert exc.value.status == 400
    async_load_video_mock.assert_not_called()


@pytest.mark.asyncio
async def test_internal_service_host_rejected_by_default(monkeypatch):
    """`kubernetes.default.svc` is in the blocked-host list exactly so a
    client cannot probe the in-cluster API via a video_url."""
    monkeypatch.delenv("DYN_MM_ALLOW_INTERNAL", raising=False)
    monkeypatch.delenv("DYN_MM_LOCAL_PATH", raising=False)
    _reload_processor(monkeypatch)

    processor = _make_processor()
    request = _video_request("https://kubernetes.default.svc/api/v1/namespaces")

    with pytest.raises(HttpStatusError) as exc:
        await processor.process_openai_request(
            request, embeddings=None, ep_disaggregated_params=None
        )
    assert exc.value.status == 400
    async_load_video_mock.assert_not_called()


@pytest.mark.asyncio
async def test_http_public_scheme_rejected_by_default(monkeypatch):
    """`http://` (even to a public host) is not allowed by the default
    policy; the worker should not be doing cleartext media fetches."""
    monkeypatch.delenv("DYN_MM_ALLOW_INTERNAL", raising=False)
    monkeypatch.delenv("DYN_MM_LOCAL_PATH", raising=False)
    _reload_processor(monkeypatch)

    processor = _make_processor()
    request = _video_request("http://public.example.com/x.mp4")

    with pytest.raises(HttpStatusError) as exc:
        await processor.process_openai_request(
            request, embeddings=None, ep_disaggregated_params=None
        )
    assert exc.value.status == 400
    async_load_video_mock.assert_not_called()


@pytest.mark.asyncio
async def test_file_scheme_rejected(monkeypatch):
    """Pre-existing PR guardrail still in force: `file://` is refused
    outright, regardless of any `DYN_MM_LOCAL_PATH` config — videos are
    not whitelisted for local-path access like images optionally are."""
    monkeypatch.delenv("DYN_MM_ALLOW_INTERNAL", raising=False)
    monkeypatch.delenv("DYN_MM_LOCAL_PATH", raising=False)
    _reload_processor(monkeypatch)

    processor = _make_processor()
    request = _video_request("file:///etc/passwd")

    with pytest.raises(HttpStatusError) as exc:
        await processor.process_openai_request(
            request, embeddings=None, ep_disaggregated_params=None
        )
    assert exc.value.status == 400
    assert "Local file access is not allowed" in str(exc.value)
    async_load_video_mock.assert_not_called()


@pytest.mark.asyncio
async def test_https_public_ip_passes_validation(monkeypatch):
    """Happy path is unchanged: a public IP literal over https:// passes
    `validate_media_url` (no DNS, no network) and `async_load_video` is
    invoked with the normalized URL — proving the fix doesn't break the
    PR's intended video loading path."""
    monkeypatch.delenv("DYN_MM_ALLOW_INTERNAL", raising=False)
    monkeypatch.delenv("DYN_MM_LOCAL_PATH", raising=False)
    _reload_processor(monkeypatch)

    processor = _make_processor()
    # 1.1.1.1 is Cloudflare DNS, a public IP not in any blocked range.
    # `validate_url` early-returns on a parseable IP literal without DNS,
    # so this stays offline.
    url = "https://1.1.1.1/x.mp4"
    request = _video_request(url)

    await processor.process_openai_request(
        request, embeddings=None, ep_disaggregated_params=None
    )
    async_load_video_mock.assert_awaited_once()
    called_url = async_load_video_mock.await_args.args[0]
    assert called_url == url


@pytest.mark.asyncio
async def test_url_validation_error_propagates(monkeypatch):
    """Defect regression: before the fix the video_path swallowed
    `UrlValidationError` as a generic `Exception` and converted it to a
    400 with a "Failed to load video" message, hiding the SSRF reason.
    After the fix a `UrlValidationError` is turned into a 400 that keeps
    the validator's diagnostic (e.g. `http:// URLs are not allowed`)."""
    monkeypatch.delenv("DYN_MM_ALLOW_INTERNAL", raising=False)
    monkeypatch.delenv("DYN_MM_LOCAL_PATH", raising=False)
    _reload_processor(monkeypatch)

    processor = _make_processor()
    request = _video_request("http://example.com/x.mp4")

    with pytest.raises(HttpStatusError) as exc:
        await processor.process_openai_request(
            request, embeddings=None, ep_disaggregated_params=None
        )
    assert exc.value.status == 400
    # The validator's own SSRF diagnostic, not the generic video-load error.
    assert "not allowed" in str(exc.value)
    assert "Failed to load video" not in str(exc.value)
    async_load_video_mock.assert_not_called()


@pytest.mark.asyncio
async def test_http_internal_allowed_when_policy_opted_in(monkeypatch):
    """Operator-controlled escape hatch: with `DYN_MM_ALLOW_INTERNAL=1`
    the policy allows http:// and private IPs, mirroring the image path.
    The break-the-fix behavior here would be for the validator to reject
    http:// even when the env flag opted in."""
    monkeypatch.setenv("DYN_MM_ALLOW_INTERNAL", "1")
    monkeypatch.delenv("DYN_MM_LOCAL_PATH", raising=False)
    _reload_processor(monkeypatch)

    processor = _make_processor()
    request = _video_request("http://169.254.169.254/latest/meta-data/")

    await processor.process_openai_request(
        request, embeddings=None, ep_disaggregated_params=None
    )
    async_load_video_mock.assert_awaited_once()
    assert async_load_video_mock.await_args.args[0] == "http://169.254.169.254/latest/meta-data/"


# Suppress the unused-import warning for UrlValidationError; kept for
# downstream consumers that may want to assert against the type.
_ = UrlValidationError