#  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

"""Token-id prefix for one streaming choice.

Reasoning parsers test ``token in previous_token_ids`` on every chunk.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence


class TokenPrefixView(Sequence[int]):
    """Ids ``[0, stop)``. Holds its list, so a later reset does not change it."""

    __slots__ = ("_first", "_ids", "_stop")

    def __init__(self, ids: list[int], first: dict[int, int], stop: int) -> None:
        self._ids = ids
        self._first = first
        self._stop = stop

    def __len__(self) -> int:
        return self._stop

    def __iter__(self):
        ids = self._ids
        for index in range(self._stop):
            yield ids[index]

    def __contains__(self, item: object) -> bool:
        try:
            index = self._first.get(item)  # type: ignore[arg-type]
        except TypeError:
            return any(token == item for token in self._ids[: self._stop])
        return index is not None and index < self._stop

    def __getitem__(self, index: int | slice) -> int | list[int]:
        if isinstance(index, slice):
            start, stop, step = index.indices(self._stop)
            return self._ids[start:stop:step]
        if index < 0:
            index += self._stop
        if index < 0 or index >= self._stop:
            raise IndexError(index)
        return self._ids[index]

    def covers_full(self, ids: list[int]) -> bool:
        return self._ids is ids and self._stop == len(ids)


class TokenPrefix:
    """Growing token-id list with O(1) prefix membership."""

    __slots__ = ("_first", "_ids")

    def __init__(self) -> None:
        self._ids: list[int] = []
        self._first: dict[int, int] = {}

    def __len__(self) -> int:
        return len(self._ids)

    def view(self) -> TokenPrefixView:
        return TokenPrefixView(self._ids, self._first, len(self._ids))

    def append(self, delta: Sequence[int]) -> tuple[TokenPrefixView, TokenPrefixView]:
        """Append ``delta``. Return views of the prefix before and after."""
        previous = TokenPrefixView(self._ids, self._first, len(self._ids))
        ids = self._ids
        first = self._first
        for token in delta:
            index = len(ids)
            ids.append(token)
            try:
                if token not in first:
                    first[token] = index
            except TypeError:
                # Unhashable ids still live in the list; membership scans.
                pass
        current = TokenPrefixView(self._ids, self._first, len(self._ids))
        return previous, current

    def adopt(self, tokens: Iterable[int]) -> None:
        """Replace the prefix, unless ``tokens`` is already its full view."""
        if isinstance(tokens, TokenPrefixView) and tokens.covers_full(self._ids):
            return
        copied = list(tokens)
        self._ids = []
        self._first = {}
        self.append(copied)
