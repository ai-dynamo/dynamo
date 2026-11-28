# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Retain source ancestry when parallel media loads must report one failure."""

from collections.abc import Sequence

from dynamo.common.http import HttpError


def make_media_batch_error(
    message: str, failures: Sequence[BaseException]
) -> Exception:
    """Keep known fetch ancestry in mixed batches without assigning an HTTP status."""
    # The first cause may be a decode failure; later fetch diagnostics still
    # need their known ancestry when the frontend inspects the aggregate.
    if any(isinstance(failure, HttpError) for failure in failures):
        return HttpError(message)
    return Exception(message)
