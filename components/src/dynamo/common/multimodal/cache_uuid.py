# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backend capability guard for client-provided multimodal cache UUIDs."""

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
