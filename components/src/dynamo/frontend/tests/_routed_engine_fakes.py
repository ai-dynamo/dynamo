#  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

"""Shared fakes for RoutedEngine in processor unit tests."""


async def _async_iter(items):
    for item in items:
        yield item


class FakeRoutedItem:
    """Mimics a real routed-engine item with is_error/comments/data methods."""

    def __init__(self, data, is_error=False, comments=None):
        self._data = data
        self._is_error = is_error
        self._comments = comments or []

    def is_error(self):
        return self._is_error

    def comments(self):
        return self._comments

    def data(self):
        return self._data


class FakeRoutedEngine:
    def __init__(self, items=None):
        if items is None:
            items = [FakeRoutedItem({"token_ids": [101], "index": 0})]
        else:
            items = [
                item if isinstance(item, FakeRoutedItem) else FakeRoutedItem(item)
                for item in items
            ]
        self.items = items
        self.requests = []
        self.kwargs = []

    async def generate(self, preprocessed, **kwargs):
        self.requests.append(preprocessed)
        self.kwargs.append(kwargs)
        return _async_iter(self.items)
