# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for dynamo.triton.util: endpoint slugification, dtype mapping,
and Dynamo <-> NumPy tensor conversion helpers."""

import re

import numpy as np
import pytest

from dynamo.triton.util import (
    DYNAMO_TO_TRITON_DTYPE,
    TRITON_TO_DYNAMO_DTYPE,
    dynamo_tensor_to_numpy,
    endpoint_slug,
    numpy_to_dynamo_values,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.triton,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


# --- endpoint_slug ---------------------------------------------------------


def test_endpoint_slug_only_uses_dynamo_safe_chars():
    """Slug must be restricted to [a-z0-9_-]; anything else breaks NATS/discovery keys."""
    slug = endpoint_slug("My.Model/With Spaces,commas")
    assert re.fullmatch(r"[a-z0-9_-]+", slug), slug


def test_endpoint_slug_lowercases_uppercase():
    """Triton model names may be uppercase; slugs must not be, so they fit the
    Dynamo endpoint character set."""
    slug = endpoint_slug("MyModel")
    assert slug.startswith("mymodel_")


def test_endpoint_slug_replaces_disallowed_chars_with_underscore():
    """Every char outside [a-z0-9_-] is folded to '_' (per util.endpoint_slug regex)."""
    slug = endpoint_slug("nop_TYPE_BOOL_-1,-1")
    # commas -> underscore; the '-' is allowed and preserved
    prefix, _, _hash = slug.rpartition("_")
    assert prefix == "nop_type_bool_-1_-1"


def test_endpoint_slug_preserves_allowed_chars():
    """[a-z0-9_-] pass through untouched."""
    slug = endpoint_slug("abc-123_XYZ")
    prefix, _, _hash = slug.rpartition("_")
    assert prefix == "abc-123_xyz"


def test_endpoint_slug_is_deterministic():
    """Same input yields the same slug — required for stable registration keys."""
    assert endpoint_slug("identity") == endpoint_slug("identity")


def test_endpoint_slug_appends_8_char_hex_hash():
    """The suffix is a lowercase-hex sha256 prefix of length 8."""
    slug = endpoint_slug("identity")
    _, _, suffix = slug.rpartition("_")
    assert re.fullmatch(r"[0-9a-f]{8}", suffix), suffix


def test_endpoint_slug_differentiates_prefix_collisions():
    """Different model names must produce slugs where neither is a prefix of the
    other — the discovery store's boundary-less prefix scan would otherwise
    conflate them."""
    a = endpoint_slug("nop_-1_-1")
    b = endpoint_slug("nop_-1_-1_-1")
    assert a != b
    assert not a.startswith(b) and not b.startswith(a)


def test_endpoint_slug_differs_between_case_variants():
    """The hash is over the original (case-sensitive) name, so case-only
    variants produce distinct slugs even though their prefixes match."""
    lower = endpoint_slug("identity")
    mixed = endpoint_slug("Identity")
    assert lower != mixed


# --- dtype maps ------------------------------------------------------------


def test_triton_to_dynamo_dtype_covers_supported_types():
    """The map must cover the dtypes RequestHandler will encounter on outputs."""
    expected = {
        "BOOL",
        "UINT8",
        "UINT16",
        "UINT32",
        "UINT64",
        "INT8",
        "INT16",
        "INT32",
        "INT64",
        "FP16",
        "FP32",
        "FP64",
        "BYTES",
    }
    assert set(TRITON_TO_DYNAMO_DTYPE) == expected


def test_dynamo_to_triton_dtype_is_the_inverse():
    """DYNAMO_TO_TRITON_DTYPE must be the exact inverse of TRITON_TO_DYNAMO_DTYPE."""
    inverse = {v: k for k, v in TRITON_TO_DYNAMO_DTYPE.items()}
    assert DYNAMO_TO_TRITON_DTYPE == inverse


# --- dynamo_tensor_to_numpy ------------------------------------------------


def _tensor(name: str, data_type: str, shape: list[int], values: list) -> dict:
    return {
        "metadata": {"name": name, "shape": shape, "data_type": data_type},
        "data": {"data_type": data_type, "values": values},
    }


def test_dynamo_tensor_to_numpy_int32():
    """Int32 values become a NumPy int32 array with the declared shape."""
    arr = dynamo_tensor_to_numpy(_tensor("IN", "Int32", [3], [1, 2, 3]))
    np.testing.assert_array_equal(arr, np.array([1, 2, 3], np.int32))
    assert arr.dtype == np.int32


def test_dynamo_tensor_to_numpy_float32_preserves_shape():
    """The declared shape drives .reshape (values are flat on the wire)."""
    arr = dynamo_tensor_to_numpy(_tensor("IN", "Float32", [2, 2], [1.0, 2.0, 3.0, 4.0]))
    assert arr.dtype == np.float32
    assert arr.shape == (2, 2)
    np.testing.assert_array_equal(arr, np.array([[1.0, 2.0], [3.0, 4.0]], np.float32))


def test_dynamo_tensor_to_numpy_bytes_yields_object_array():
    """Bytes values arrive as lists of ints (per-string byte lists) and become
    an object-dtype array Triton accepts."""
    arr = dynamo_tensor_to_numpy(_tensor("IN", "Bytes", [1], [list(b"hello")]))
    assert arr.dtype == object
    assert arr.tolist() == [b"hello"]


def test_dynamo_tensor_to_numpy_bytes_multi_element():
    """Multiple byte strings are preserved, one per element."""
    arr = dynamo_tensor_to_numpy(_tensor("IN", "Bytes", [2], [list(b"a"), list(b"bb")]))
    assert arr.shape == (2,)
    assert arr.tolist() == [b"a", b"bb"]


def test_dynamo_tensor_to_numpy_rejects_unknown_dtype():
    """An unrecognized dtype string surfaces as ValueError, not a NumPy crash."""
    with pytest.raises(ValueError, match="Unsupported tensor data type"):
        dynamo_tensor_to_numpy(_tensor("IN", "NotADtype", [1], [0]))


# --- numpy_to_dynamo_values -----------------------------------------------


def test_numpy_to_dynamo_values_numeric_flattens_multidim():
    """Numeric arrays are returned as flat lists so the wire format stays 1D."""
    arr = np.array([[1, 2], [3, 4]], np.int32)
    assert numpy_to_dynamo_values(arr, "Int32") == [1, 2, 3, 4]


def test_numpy_to_dynamo_values_numeric_scalar_dtype_roundtrip():
    """Float dtypes come back as Python floats (numpy scalars serialize)."""
    values = numpy_to_dynamo_values(np.array([1.5, 2.5], np.float32), "Float32")
    assert values == [1.5, 2.5]


def test_numpy_to_dynamo_values_bytes_unwraps_bytes_to_list_of_ints():
    """Bytes tensors serialize each element as a list of byte-ints
    (matches Dynamo's over-the-wire format)."""
    arr = np.array([b"hi", b"bye"], dtype=object)
    values = numpy_to_dynamo_values(arr, "Bytes")
    assert values == [list(b"hi"), list(b"bye")]


def test_numpy_to_dynamo_values_bytes_flattens_multidim():
    """Multi-dim byte arrays flatten just like numeric arrays."""
    arr = np.array([[b"a"], [b"b"]], dtype=object)
    values = numpy_to_dynamo_values(arr, "Bytes")
    assert values == [list(b"a"), list(b"b")]
