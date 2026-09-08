# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib


def generate_hash_key(*args) -> str:
    """
    Generate a hashable key based on the provided arguments.

    The digest must stay byte-for-byte identical across releases: it is the
    encode worker's embedding-cache key, so any change silently invalidates
    every entry cached by an older worker during a rolling upgrade.

    Args:
        *args: A variable number of arguments to generate the key.

    Returns:
        A string representing the hashable key.
    """
    key = hashlib.sha256()
    for arg in args:
        key.update(str(arg).encode("utf-8"))
    return key.hexdigest()
