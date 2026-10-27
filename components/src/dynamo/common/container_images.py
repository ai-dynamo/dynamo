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
    """Replace the image name using the source image's shared release tag.

    OCI digests identify a manifest within one repository. They cannot be
    copied from a Dynamo image to a sibling planner or backend repository.
    """
    slash_idx = image_ref.rfind("/")
    prefix = image_ref[: slash_idx + 1] if slash_idx >= 0 else ""
    suffix = image_ref[slash_idx + 1 :]
    name_and_tag = suffix.partition("@")[0]
    colon_idx = name_and_tag.rfind(":")
    if colon_idx < 0 or colon_idx == len(name_and_tag) - 1:
        raise ValueError(
            f"Cannot derive {new_name!r} from image {image_ref!r}: "
            "a shared release tag is required; repository-specific digests "
            "cannot be transferred to sibling images"
        )
    return f"{prefix}{new_name}{name_and_tag[colon_idx:]}"


def derive_planner_image(source_image: str) -> str:
    """Derive the planner image that belongs to a published Dynamo image."""
    return _replace_image_name(source_image, PLANNER_IMAGE_NAME)


def derive_backend_image(source_image: str, backend: str) -> str:
    """Derive one backend runtime image from a published Dynamo image."""
    backend_image_name = BACKEND_IMAGE_NAMES.get(backend)
    if backend_image_name is None:
        raise ValueError(
            f"Cannot derive backend image for unknown backend '{backend}'. "
            f"Supported backends: {list(BACKEND_IMAGE_NAMES.keys())}"
        )
    return _replace_image_name(source_image, backend_image_name)
