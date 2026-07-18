# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evict exact checkpoint trees from the filesystem page cache."""

import json
import os
import stat
import sys


def evict_tree(root: str) -> dict[str, int | str]:
    files = 0
    total_bytes = 0
    errors = 0
    try:
        root_stat = os.lstat(root)
    except FileNotFoundError:
        return {
            "root": root,
            "status": "missing",
            "files": files,
            "bytes": total_bytes,
            "errors": 1,
        }
    except OSError as error:
        print(f"error\tpath={root}\terror={error}", file=sys.stderr)
        return {
            "root": root,
            "status": "error",
            "files": files,
            "bytes": total_bytes,
            "errors": 1,
        }

    if not stat.S_ISDIR(root_stat.st_mode):
        return {
            "root": root,
            "status": "not_directory",
            "files": files,
            "bytes": total_bytes,
            "errors": 1,
        }

    def walk_error(error: OSError) -> None:
        nonlocal errors
        errors += 1
        print(f"error\tpath={error.filename or root}\terror={error}", file=sys.stderr)

    for directory, _, names in os.walk(root, onerror=walk_error):
        for name in names:
            path = os.path.join(directory, name)
            try:
                file_stat = os.stat(path, follow_symlinks=False)
                if not stat.S_ISREG(file_stat.st_mode):
                    continue
                fd = os.open(path, os.O_RDONLY)
                try:
                    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                finally:
                    os.close(fd)
                files += 1
                total_bytes += file_stat.st_size
            except OSError as error:
                errors += 1
                print(f"error\tpath={path}\terror={error}", file=sys.stderr)

    status = "error" if errors else "ok" if files else "empty"
    return {
        "root": root,
        "status": status,
        "files": files,
        "bytes": total_bytes,
        "errors": errors,
    }


def main(arguments: list[str]) -> int:
    results = [evict_tree(argument) for argument in arguments]
    ok = bool(results) and all(result["status"] == "ok" for result in results)
    payload = {
        "ok": ok,
        "roots": results,
        "total": {
            "roots": len(results),
            "files": sum(int(result["files"]) for result in results),
            "bytes": sum(int(result["bytes"]) for result in results),
            "errors": sum(int(result["errors"]) for result in results),
        },
    }
    json.dump(payload, sys.stdout, separators=(",", ":"))
    print(flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
