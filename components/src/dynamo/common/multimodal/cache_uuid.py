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


def reject_unsupported_backend_multimodal_data(
    backend_multi_modal_data: object,
) -> None:
    """Refuse a custom-modality payload on a backend that cannot install it.

    Only the vLLM backend installs these inputs as engine ``multi_modal_data``
    for a registered modality processor. Every other backend would run the
    request as plain text and answer as if the payload had not been sent.
    """
    if backend_multi_modal_data is not None:
        raise ValueError(
            "multi_modal_data on completion requests is supported only by the "
            "vLLM backend"
        )
