# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib


class EmbeddingCache:
    def __init__(self):
        # Initialize an empty dictionary to store key-value pairs
        self.cache = {}

    def generate_hash_key(self, *args):
        """
        Generate a hashable key based on the provided arguments.

        Args:
            *args: A variable number of arguments to generate the key.

        Returns:
            A string representing the hashable key.
        """
        key = hashlib.sha256()
        for arg in args:
            key.update(str(arg).encode("utf-8"))
        return key.hexdigest()

    def has_key(self, key):
        """
        Check if a key exists in the cache.

        Args:
            key: The key to check.

        Returns:
            True if the key exists in the cache, False otherwise.
        """
        return key in self.cache

    def set(self, key, value):
        """
        Store a key-value pair in the cache.

        Args:
            key: The key to store the value under.
            value: The value to store, expected to be a tuple.
        """
        self.cache[key] = value

    def get(self, key):
        """
        Retrieve the value associated with a key.

        Args:
            key: The key to look up.

        Returns:
            The value (tuple) associated with the key, or None if the key is not found.
        """
        return self.cache.get(key)
