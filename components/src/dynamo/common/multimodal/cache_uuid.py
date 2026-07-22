# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backend capability guard for client-provided multimodal cache UUIDs."""


def reject_unsupported_multimodal_uuids(
    multi_modal_uuids: object,
    *,
    backend: str,
) -> None:
    if not multi_modal_uuids:
        return

    raise ValueError(
        "Image UUID caching is currently supported only by the vLLM "
        f"backend; {backend} does not support the `uuid` field"
    )
