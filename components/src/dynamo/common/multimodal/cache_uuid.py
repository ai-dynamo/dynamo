# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backend capability guards for client-provided multimodal request inputs."""

from collections.abc import Mapping, Sequence


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


def reject_unsupported_json_multimodal_data(multi_modal_data: object) -> None:
    if isinstance(multi_modal_data, Mapping) and any(
        isinstance(item, Mapping) and "Json" in item
        for items in multi_modal_data.values()
        if isinstance(items, list)
        for item in items
    ):
        raise ValueError("Custom JSON content is supported only by the vLLM backend")
