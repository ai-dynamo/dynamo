# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Accept ``token_ids`` in every form the request plane can carry.

The frontend sends token ids as a sequence, or, with ``DYN_TOKEN_IDS_AS_BYTES``,
as one packed little-endian int32 buffer. Python handlers that need a
``list[int]`` call :func:`token_ids_to_list` at their entry point; handlers that
only need the prompt length call :func:`token_ids_len`; handlers that can consume
a buffer directly leave the value alone.
"""

import array
import sys
from collections.abc import Mapping, MutableMapping
from typing import Any, Optional, TypeVar, cast, overload

_INT32 = "i" if array.array("i").itemsize == 4 else "l"
_RequestT = TypeVar("_RequestT", bound=Mapping[str, Any])


@overload
def token_ids_to_list(value: None) -> None:
    ...


@overload
def token_ids_to_list(value: Any) -> list[int]:
    ...


def token_ids_to_list(value: Any) -> Optional[list[int]]:
    if value is None or isinstance(value, list):
        return value
    if isinstance(value, (bytes, bytearray, memoryview)):
        buf = memoryview(value).cast("B")
        if buf.nbytes % 4:
            raise ValueError(
                f"packed token_ids byte length {buf.nbytes} is not a multiple of 4"
            )
        ids = array.array(_INT32)
        ids.frombytes(buf)
        if sys.byteorder != "little":
            ids.byteswap()
        return ids.tolist()
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return tolist()
    return list(value)


def token_ids_len(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, (bytes, bytearray, memoryview)):
        return memoryview(value).nbytes // 4
    return len(value)


def normalize_request_token_ids(request: _RequestT) -> _RequestT:
    ids = request.get("token_ids")
    if ids is not None and not isinstance(ids, list):
        cast(MutableMapping[str, Any], request)["token_ids"] = token_ids_to_list(ids)
    return request
