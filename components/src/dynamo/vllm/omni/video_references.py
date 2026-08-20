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
    VideoNvExt,
)

_REFERENCE_LIMITS = {
    "image": 30 * 1024 * 1024,
    "video": 50 * 1024 * 1024,
    "audio": 15 * 1024 * 1024,
}
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

    def cleanup(self) -> None:
        self.temporary_directory.cleanup()


class VideoReferenceMaterializer:
    """Validate, securely fetch, and stage typed video references."""

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

        self._validate_contract(references, request.nvext or VideoNvExt())
        temporary_directory = tempfile.TemporaryDirectory(
            prefix="dynamo_h3_references_"
        )
        grouped: dict[str, list[str]] = {}
        try:
            for index, reference in enumerate(references):
                path = await self._materialize_one(
                    reference,
                    index=index,
                    directory=Path(temporary_directory.name),
                )
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
    ) -> str:
        normalized = await validate_media_url(reference.source, self._url_policy)
        parsed = urlparse(normalized)
        if parsed.scheme == "file":
            path = Path(url2pathname(unquote(parsed.path)))
            self._validate_size(path.stat().st_size, reference.type)
            return str(path)

        if parsed.scheme == "data":
            content = await read_local_media_bytes(normalized, self._url_policy)
        else:
            limit = _REFERENCE_LIMITS[reference.type]
            try:
                content = await fetch_bytes(
                    normalized,
                    self._http_timeout,
                    policy=self._url_policy,
                    max_bytes=limit,
                )
            except HttpBodyTooLargeError as e:
                raise ValueError(
                    f"{reference.type} reference exceeds the "
                    f"{limit // (1024 * 1024)} MiB limit"
                ) from e
        if not content:
            raise ValueError(f"{reference.type} reference is empty")
        self._validate_size(len(content), reference.type)

        path = directory / f"{index:02d}{self._suffix(reference)}"
        await asyncio.to_thread(path.write_bytes, content)
        return str(path)

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

    @staticmethod
    def _validate_contract(
        references: list[VideoInputReference], nvext: VideoNvExt
    ) -> None:
        grouped = {
            kind: [reference for reference in references if reference.type == kind]
            for kind in ("image", "video", "audio")
        }
        task = nvext.task
        if task == "t2va":
            raise ValueError("t2va does not accept input_references")
        if task == "fl2va":
            if grouped["video"] or grouped["audio"] or not grouped["image"]:
                raise ValueError("fl2va accepts only one or two image references")
            if len(grouped["image"]) > 2:
                raise ValueError("fl2va accepts at most two image references")
            if nvext.frame_indices is not None and len(nvext.frame_indices) != len(
                grouped["image"]
            ):
                raise ValueError("fl2va requires one frame index per image reference")
        if task == "ref2va":
            if not grouped["image"] and not grouped["video"]:
                raise ValueError(
                    "ref2va requires at least one image or video reference"
                )
            if len(grouped["image"]) > 9:
                raise ValueError("ref2va accepts at most 9 image references")
            if len(grouped["video"]) > 3:
                raise ValueError("ref2va accepts at most 3 video references")
            if len(grouped["audio"]) > 3:
                raise ValueError("ref2va accepts at most 3 audio references")
        if len(references) > 12:
            raise ValueError("input_references accepts at most 12 references")

        start_times = nvext.start_time_seconds
        if isinstance(start_times, list) and len(start_times) != len(grouped["video"]):
            raise ValueError(
                "start_time_seconds requires one value per video reference"
            )
