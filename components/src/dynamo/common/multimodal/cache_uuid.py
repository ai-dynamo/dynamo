# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Policy and backend capability guard for client-provided multimodal cache UUIDs."""

from collections.abc import Mapping, Sequence

from dynamo.common.utils.env import env_bool

ALLOW_CLIENT_CACHE_UUIDS_ENV = "DYN_MM_ALLOW_CLIENT_CACHE_UUIDS"
UUID_ONLY_IMAGE_REJECTED_MESSAGE = (
    "Image parts that carry only a `uuid` are not accepted. Send the image URL "
    "or inline image data. To let clients reuse cached images by `uuid`, set "
    "DYN_MM_ALLOW_CLIENT_CACHE_UUIDS=1 on the vLLM workers."
)


def client_cache_uuids_allowed() -> bool:
    """Return whether client `uuid`s may be used as engine cache identities.

    Off by default: the engine cache identity is then derived from the media
    content, and image parts that carry only a `uuid` are rejected. Read per
    call so workers follow the current environment.
    """
    return env_bool(ALLOW_CLIENT_CACHE_UUIDS_ENV)


def reject_unsupported_multimodal_uuids(multi_modal_uuids: object) -> None:
    if multi_modal_uuids is None:
        return

    unsupported = "Cache UUIDs are supported only by the vLLM backend"
    if not isinstance(multi_modal_uuids, Mapping):
        raise ValueError(unsupported)

    for uuids in multi_modal_uuids.values():
        if not isinstance(uuids, Sequence) or isinstance(uuids, (str, bytes)):
            raise ValueError(unsupported)
        if any(uuid is not None for uuid in uuids):
            raise ValueError(unsupported)
