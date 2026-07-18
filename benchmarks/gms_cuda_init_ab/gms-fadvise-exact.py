# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evict exact checkpoint trees from the filesystem page cache."""

import os
import sys


def evict_tree(root: str) -> tuple[int, int, int]:
    files = 0
    total_bytes = 0
    errors = 0
    if not os.path.exists(root):
        print(f"root={root}\tmissing=true")
        return files, total_bytes, errors

    for directory, _, names in os.walk(root):
        for name in names:
            path = os.path.join(directory, name)
            try:
                stat = os.stat(path, follow_symlinks=False)
                if not os.path.isfile(path):
                    continue
                fd = os.open(path, os.O_RDONLY)
                try:
                    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                finally:
                    os.close(fd)
                files += 1
                total_bytes += stat.st_size
            except OSError as error:
                errors += 1
                print(f"error\tpath={path}\terror={error}", file=sys.stderr)

    print(
        f"root={root}\tfiles={files}\tbytes={total_bytes}\terrors={errors}",
        flush=True,
    )
    return files, total_bytes, errors


def main(arguments: list[str]) -> int:
    totals = [0, 0, 0]
    for argument in arguments:
        result = evict_tree(argument)
        totals = [left + right for left, right in zip(totals, result)]

    print(
        f"total\tfiles={totals[0]}\tbytes={totals[1]}\terrors={totals[2]}",
        flush=True,
    )
    return 1 if totals[2] else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
