#  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

"""Token-id prefix for one streaming choice.

Reasoning parsers test ``token in previous_token_ids`` on every chunk.
DeepSeek v3/v3.1 tool parsers call ``count`` on the same sequences.
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Iterable, Sequence


class TokenPrefixView(Sequence[int]):
    """Ids ``[0, stop)``. Later appends and resets do not change this view."""

    __slots__ = ("_ids", "_positions", "_stop")

    def __init__(
        self, ids: list[int], positions: dict[int, list[int]], stop: int
    ) -> None:
        self._ids = ids
        self._positions = positions
        self._stop = stop

    def __len__(self) -> int:
        return self._stop

    def __iter__(self):
        ids = self._ids
        for index in range(self._stop):
            yield ids[index]

    def __contains__(self, item: object) -> bool:
        try:
            positions = self._positions.get(item)  # type: ignore[arg-type]
        except TypeError:
            return any(self._ids[i] == item for i in range(self._stop))
        return bool(positions) and positions[0] < self._stop

    def count(self, value: object) -> int:
        try:
            positions = self._positions.get(value)  # type: ignore[arg-type]
        except TypeError:
            return sum(1 for i in range(self._stop) if self._ids[i] == value)
        if not positions:
            return 0
        # Indexes at or after _stop belong to a later chunk.
        return bisect_right(positions, self._stop - 1)

    def __getitem__(self, index: int | slice) -> int | list[int]:
        if isinstance(index, slice):
            start, stop, step = index.indices(self._stop)
            # A negative step can leave stop at -1, which a second slice
            # would read against the backing list rather than this view.
            if step < 0:
                return [self._ids[i] for i in range(start, stop, step)]
            return self._ids[start:stop:step]
        if index < 0:
            index += self._stop
        if index < 0 or index >= self._stop:
            raise IndexError(index)
        return self._ids[index]

    def covers_full(self, ids: list[int]) -> bool:
        return self._ids is ids and self._stop == len(ids)


class TokenPrefix:
    """Growing token-id list. ``in`` uses the first index; ``count`` uses them all."""

    __slots__ = ("_ids", "_positions")

    def __init__(self) -> None:
        self._ids: list[int] = []
        self._positions: dict[int, list[int]] = {}

    def __len__(self) -> int:
        return len(self._ids)

    def view(self) -> TokenPrefixView:
        return TokenPrefixView(self._ids, self._positions, len(self._ids))

    def append(self, delta: Sequence[int]) -> tuple[TokenPrefixView, TokenPrefixView]:
        """Append ``delta``. Return views of the prefix before and after."""
        previous = TokenPrefixView(self._ids, self._positions, len(self._ids))
        ids = self._ids
        positions = self._positions
        for token in delta:
            index = len(ids)
            ids.append(token)
            try:
                bucket = positions.get(token)
                if bucket is None:
                    positions[token] = [index]
                else:
                    bucket.append(index)
            except TypeError:
                # Unhashable ids still live in the list; membership scans.
                pass
        current = TokenPrefixView(self._ids, self._positions, len(self._ids))
        return previous, current

    def adopt(self, tokens: Iterable[int]) -> None:
        """Replace the prefix, unless ``tokens`` is already its full view."""
        if isinstance(tokens, TokenPrefixView) and tokens.covers_full(self._ids):
            return
        copied = list(tokens)
        self._ids = []
        self._positions = {}
        self.append(copied)
