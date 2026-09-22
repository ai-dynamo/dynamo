# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Extract image URLs and build sanitized decoder requests."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from dynamo.common.backend import GenerateRequest
from dynamo.llm.exceptions import InvalidArgument

_IMAGE_URL = "image_url"
_URL = "Url"

_RAW_MULTIMODAL_FIELDS = (
    "multi_modal_data",
    "multi_modal_uuids",
    "mm_processor_kwargs",
    "mm_routing_info",
    "media_io_kwargs",
)

_MULTIMODAL_EXTRA_ARG_FIELDS = (
    "mm_processor_kwargs",
    "mm_kwargs_shm",
    "mm_kwargs_nixl",
    "mm_hashes",
    "mm_hashes_by_modality",
    "mm_placeholders",
    "mm_placeholders_by_modality",
    "expanded_token_ids",
)


class ImageUrlExtractor:
    """Pull image URLs out of the frontend-preprocessed request shape."""

    def extract(self, request: Mapping[str, Any]) -> list[str]:
        """Return the list of image URLs in request order."""

        multimodal = request.get("multi_modal_data") or {}
        if not isinstance(multimodal, Mapping):
            raise InvalidArgument("multi_modal_data must be an object")

        unsupported = sorted(
            key for key, value in multimodal.items() if key != _IMAGE_URL and value
        )
        if unsupported:
            raise InvalidArgument(
                "external encoder supports image inputs only; "
                f"got unsupported multimodal data: {unsupported}"
            )

        image_items = multimodal.get(_IMAGE_URL) or []
        if not isinstance(image_items, list) or not image_items:
            raise InvalidArgument("external encoder requires at least one image")

        image_urls: list[str] = []
        for index, item in enumerate(image_items):
            if not isinstance(item, Mapping):
                raise InvalidArgument(f"image_url item {index} must be an object")
            image_url = item.get(_URL)
            if not isinstance(image_url, str) or not image_url:
                raise InvalidArgument(
                    f"image_url item {index} must contain a non-empty 'Url' string"
                )
            image_urls.append(image_url)
        return image_urls


class EncoderResultRequestBuilder:
    """Produce a decoder request from an encoder result and the original request."""

    def build(
        self,
        request_value: Mapping[str, Any],
        encoder_result: Mapping[str, Any],
        *,
        decoder_model_name: str,
    ) -> GenerateRequest:
        """Return a sanitized GenerateRequest with encoder_result set."""

        request = dict(request_value)
        if request.get("encoder_result") is not None:
            raise InvalidArgument("request already contains encoder_result")
        if request.get("prompt_embeds") is not None:
            raise InvalidArgument(
                "external encoder result cannot be combined with prompt_embeds"
            )

        request["encoder_result"] = dict(encoder_result)
        request["model"] = decoder_model_name
        for field_name in _RAW_MULTIMODAL_FIELDS:
            request.pop(field_name, None)

        extra_args = request.get("extra_args")
        if isinstance(extra_args, Mapping):
            copied_extra_args = dict(extra_args)
            for field_name in _MULTIMODAL_EXTRA_ARG_FIELDS:
                copied_extra_args.pop(field_name, None)
            request["extra_args"] = copied_extra_args

        return cast(GenerateRequest, request)
