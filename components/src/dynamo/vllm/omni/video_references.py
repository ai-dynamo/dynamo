# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Request-scoped typed reference materialization for video diffusion."""

import asyncio
import tempfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname

from dynamo.common.http import HttpBodyTooLargeError, fetch_bytes
from dynamo.common.http.url_validator import UrlValidationPolicy, validate_media_url
from dynamo.common.multimodal.media_source import read_local_media_bytes
from dynamo.common.protocols.video_protocol import (
    NvCreateVideoRequest,
    VideoInputReference,
)

_REFERENCE_LIMITS = {
    "image": 30 * 1024 * 1024,
    "video": 50 * 1024 * 1024,
    "audio": 15 * 1024 * 1024,
}
_MAX_REFERENCE_COUNT = 12
_MAX_TOTAL_REFERENCE_BYTES = 512 * 1024 * 1024
_DEFAULT_SUFFIXES = {"image": ".png", "video": ".mp4", "audio": ".wav"}
_ALLOWED_SUFFIXES = {
    "image": {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"},
    "video": {".mp4", ".mov"},
    "audio": {".wav", ".mp3"},
}
_MIME_SUFFIXES = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/heic": ".heic",
    "image/heif": ".heif",
    "video/mp4": ".mp4",
    "video/quicktime": ".mov",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/mpeg": ".mp3",
    "audio/mp3": ".mp3",
}


@dataclass
class MaterializedVideoReferences:
    """References grouped for ``OmniTextPrompt.multi_modal_data``."""

    multi_modal_data: dict[str, list[str]]
    temporary_directory: tempfile.TemporaryDirectory[str]

    def as_omni_data(self, *, allow_multiple: bool = False) -> dict[str, object]:
        """Adapt grouped paths to the selected vLLM-Omni pipeline ABI."""
        if allow_multiple:
            return dict(self.multi_modal_data)

        result: dict[str, object] = {}
        for media_type, paths in self.multi_modal_data.items():
            if len(paths) != 1:
                raise ValueError(
                    "generic vLLM-Omni video pipelines accept at most one "
                    f"{media_type} reference"
                )
            result[media_type] = paths[0]
        return result

    def cleanup(self) -> None:
        self.temporary_directory.cleanup()


class VideoReferenceMaterializer:
    """Securely fetch and stage typed video references."""

    def __init__(
        self,
        *,
        http_timeout: float = 60.0,
        url_policy: UrlValidationPolicy | None = None,
    ) -> None:
        self._http_timeout = http_timeout
        self._url_policy = url_policy or UrlValidationPolicy.from_env()

    async def materialize(
        self, request: NvCreateVideoRequest
    ) -> MaterializedVideoReferences | None:
        references = request.input_references
        if references is None:
            return None
        if len(references) > _MAX_REFERENCE_COUNT:
            raise ValueError(
                f"input_references accepts at most {_MAX_REFERENCE_COUNT} references"
            )

        temporary_directory = tempfile.TemporaryDirectory(
            prefix="dynamo_video_references_"
        )
        grouped: dict[str, list[str]] = {}
        total_bytes = 0
        try:
            for index, reference in enumerate(references):
                path, size = await self._materialize_one(
                    reference,
                    index=index,
                    directory=Path(temporary_directory.name),
                    remaining_bytes=_MAX_TOTAL_REFERENCE_BYTES - total_bytes,
                )
                total_bytes += size
                if total_bytes > _MAX_TOTAL_REFERENCE_BYTES:
                    raise ValueError("input_references exceed the 512 MiB total limit")
                grouped.setdefault(reference.type, []).append(path)
        except BaseException:
            temporary_directory.cleanup()
            raise

        return MaterializedVideoReferences(grouped, temporary_directory)

    async def _materialize_one(
        self,
        reference: VideoInputReference,
        *,
        index: int,
        directory: Path,
        remaining_bytes: int,
    ) -> tuple[str, int]:
        if remaining_bytes <= 0:
            raise ValueError("input_references exceed the 512 MiB total limit")
        normalized = await validate_media_url(reference.source, self._url_policy)
        parsed = urlparse(normalized)
        if parsed.scheme == "file":
            path = Path(url2pathname(unquote(parsed.path)))
            size = path.stat().st_size
            self._validate_size(size, reference.type)
            return str(path), size

        if parsed.scheme == "data":
            content = await read_local_media_bytes(normalized, self._url_policy)
        else:
            limit = _REFERENCE_LIMITS[reference.type]
            effective_limit = min(limit, remaining_bytes)
            try:
                content = await fetch_bytes(
                    normalized,
                    self._http_timeout,
                    policy=self._url_policy,
                    max_bytes=effective_limit,
                )
            except HttpBodyTooLargeError as e:
                if effective_limit < limit:
                    raise ValueError(
                        "input_references exceed the 512 MiB total limit"
                    ) from e
                raise ValueError(
                    f"{reference.type} reference exceeds the "
                    f"{limit // (1024 * 1024)} MiB limit"
                ) from e
        if not content:
            raise ValueError(f"{reference.type} reference is empty")
        self._validate_size(len(content), reference.type)

        path = directory / f"{index:02d}{self._suffix(reference)}"
        await asyncio.to_thread(path.write_bytes, content)
        return str(path), len(content)

    @staticmethod
    def _validate_size(size: int, reference_type: str) -> None:
        limit = _REFERENCE_LIMITS[reference_type]
        if size > limit:
            raise ValueError(
                f"{reference_type} reference exceeds the {limit // (1024 * 1024)} MiB limit"
            )

    @staticmethod
    def _suffix(reference: VideoInputReference) -> str:
        parsed = urlparse(reference.source)
        if parsed.scheme == "data":
            media_type = parsed.path.partition(",")[0].partition(";")[0].lower()
            return _MIME_SUFFIXES.get(media_type, _DEFAULT_SUFFIXES[reference.type])
        suffix = Path(unquote(parsed.path)).suffix.lower()
        if suffix in _ALLOWED_SUFFIXES[reference.type]:
            return suffix
        return _DEFAULT_SUFFIXES[reference.type]
