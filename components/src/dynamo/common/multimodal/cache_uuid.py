# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backend capability guard for client-provided multimodal cache UUIDs."""

from collections.abc import Mapping, Sequence


def reject_unsupported_multimodal_uuids(
    multi_modal_uuids: Mapping[str, Sequence[str | None]] | None,
    *,
    backend: str,
) -> None:
    if not multi_modal_uuids or not any(
        uuid is not None for uuids in multi_modal_uuids.values() for uuid in uuids
    ):
        return

    raise ValueError(
        "Image UUID caching is supported only by the vLLM "
        f"backend; {backend} does not support the `uuid` field"
    )
