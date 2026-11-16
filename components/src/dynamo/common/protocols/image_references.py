# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The list type of ``NvCreateImageRequest.input_reference``.

On the wire one reference is a string and two or more are an array, which
is what the Rust ``ImageReferences`` enum states in the schema. A worker
reads the field as a list, so the generator maps the field to this type
instead of the union the schema would give (see ``WORKER_TYPES`` in
``scripts/generate_media_protocols.py``). A string is read as a one-item
list, and one item is written back as a string, so a worker whose model has
the string field of an earlier release keeps reading it. An empty list is
rejected, as the frontend rejects it. A blank reference is the frontend's
rule: the schema cannot state it, and the worker's loader rejects it.
"""

from typing import Annotated

from pydantic import BeforeValidator, Field, PlainSerializer, StrictStr


def _as_list(value: object) -> object:
    return [value] if isinstance(value, str) else value


def _one_as_string(references: list[str]) -> str | list[str]:
    return references[0] if len(references) == 1 else references


ImageReferences = Annotated[
    list[StrictStr],
    Field(min_length=1),
    BeforeValidator(_as_list),
    PlainSerializer(_one_as_string, return_type=str | list[str]),
]
