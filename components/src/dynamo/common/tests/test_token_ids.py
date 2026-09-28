# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import array

import pytest

from dynamo.common.utils.token_ids import normalize_request_token_ids, token_ids_to_list

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]

IDS = [0, 1, 128000, 2**31 - 1, 7]


def _packed(ids):
    return b"".join(i.to_bytes(4, "little") for i in ids)


def test_list_and_none_pass_through_unchanged():
    ids = list(IDS)
    assert token_ids_to_list(ids) is ids
    assert token_ids_to_list(None) is None


@pytest.mark.parametrize("wrap", [bytes, bytearray, memoryview])
def test_packed_little_endian_int32_decodes(wrap):
    assert token_ids_to_list(wrap(_packed(IDS))) == IDS
    assert token_ids_to_list(wrap(b"")) == []


def test_odd_byte_length_is_rejected():
    with pytest.raises(ValueError):
        token_ids_to_list(_packed(IDS)[:-1])


def test_array_like_uses_tolist():
    assert token_ids_to_list(array.array("i", IDS)) == IDS


def test_normalize_rewrites_only_packed_values():
    packed = {"token_ids": _packed(IDS)}
    assert normalize_request_token_ids(packed)["token_ids"] == IDS
    plain = {"token_ids": list(IDS)}
    assert normalize_request_token_ids(plain)["token_ids"] is plain["token_ids"]
    assert normalize_request_token_ids({}) == {}
