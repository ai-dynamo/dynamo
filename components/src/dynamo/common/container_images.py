# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Published Dynamo container-image naming conventions."""

BACKEND_IMAGE_NAMES: dict[str, str] = {
    "vllm": "vllm-runtime",
    "sglang": "sglang-runtime",
    "trtllm": "tensorrtllm-runtime",
}

PLANNER_IMAGE_NAME = "dynamo-planner"


def _replace_image_name(image_ref: str, new_name: str) -> str:
    """Replace an image name while preserving its shared release identity."""
    slash_idx = image_ref.rfind("/")
    prefix = image_ref[: slash_idx + 1] if slash_idx >= 0 else ""
    suffix = image_ref[slash_idx + 1 :]
    name_and_tag, has_digest, _ = suffix.partition("@")
    colon_idx = name_and_tag.rfind(":")
    if colon_idx == len(name_and_tag) - 1 or (has_digest and colon_idx < 0):
        raise ValueError(
            f"Cannot derive {new_name!r} from image {image_ref!r}: "
            "a shared release tag is required; repository-specific digests "
            "cannot be transferred to sibling images"
        )
    tag = name_and_tag[colon_idx:] if colon_idx >= 0 else ""
    source_name = name_and_tag[:colon_idx] if colon_idx >= 0 else name_and_tag
    if source_name == new_name:
        return image_ref
    return f"{prefix}{new_name}{tag}"


def derive_planner_image(profiler_image: str) -> str:
    """Derive the planner image that belongs to a published Dynamo image."""
    return _replace_image_name(profiler_image, PLANNER_IMAGE_NAME)


def derive_backend_image(profiler_image: str, backend: str) -> str:
    """Derive one backend runtime image from a published Dynamo image."""
    backend_image_name = BACKEND_IMAGE_NAMES.get(backend)
    if backend_image_name is None:
        raise ValueError(
            f"Cannot derive backend image for unknown backend '{backend}'. "
            f"Supported backends: {list(BACKEND_IMAGE_NAMES.keys())}"
        )
    return _replace_image_name(profiler_image, backend_image_name)
