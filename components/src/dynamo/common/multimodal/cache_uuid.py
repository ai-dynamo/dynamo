# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Policy and backend capability guard for client-provided multimodal cache UUIDs."""

import os
from collections.abc import Mapping, Sequence

# Keep the name and message in sync with the Rust frontend
# (lib/llm/src/preprocessor/media/loader.rs::client_cache_uuids_allowed).
ALLOW_CLIENT_CACHE_UUIDS_ENV = "DYN_MM_ALLOW_CLIENT_CACHE_UUIDS"
UUID_ONLY_IMAGE_REJECTED_MESSAGE = (
    "Image parts that carry only a `uuid` are not accepted. Send the image URL "
    "or inline image data. To let clients reuse cached images by `uuid`, set "
    f"{ALLOW_CLIENT_CACHE_UUIDS_ENV}=1 on the frontend and the vLLM workers."
)


def client_cache_uuids_allowed() -> bool:
    """Return whether client `uuid`s may be used as engine cache identities.

    Off by default: the engine cache identity is then derived from the media
    content, and image parts that carry only a `uuid` are rejected. Read per
    call so both processes follow the current environment. As in Rust, any
    value other than ``1`` leaves it off, so a bad value never fails a request.
    """
    return os.environ.get(ALLOW_CLIENT_CACHE_UUIDS_ENV) == "1"


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
