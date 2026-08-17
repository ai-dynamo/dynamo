#!/usr/bin/env python3
"""Publish a read-only local checkpoint cache from an authoritative PVC tree."""

import argparse
import hashlib
import json
import os
import pathlib
import shutil
import stat
import tempfile


def fsync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def inventory(root):
    files = []
    for path in sorted(root.rglob("*")):
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            if stat.S_ISDIR(info.st_mode):
                continue
            raise RuntimeError(f"cache source contains non-regular file: {path}")
        files.append((path.relative_to(root).as_posix(), info.st_size))
    if not files:
        raise RuntimeError("cache source has no regular files")
    encoded = "".join(f"{name}\0{size}\n" for name, size in files).encode()
    return files, "sha256:" + hashlib.sha256(encoded).hexdigest()


def copy_file(source, target, expected_size):
    src_fd = os.open(source, os.O_RDONLY | os.O_NOFOLLOW)
    dst_fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    digest = hashlib.sha256()
    copied = 0
    try:
        while True:
            block = os.read(src_fd, 4 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
            copied += len(block)
            view = memoryview(block)
            while view:
                written = os.write(dst_fd, view)
                view = view[written:]
        if copied != expected_size:
            raise RuntimeError(f"source changed while copying {source}")
        os.fsync(dst_fd)
    finally:
        os.close(src_fd)
        os.close(dst_fd)
    os.chmod(target, 0o400)
    return "sha256:" + digest.hexdigest()


def publish(source, cache_root, compatibility):
    source = source.resolve(strict=True)
    files, checkpoint_digest = inventory(source)
    cache_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    final = cache_root / checkpoint_digest.removeprefix("sha256:")
    if final.exists():
        manifest = final / "manifest.json"
        if final.is_dir() and (final / "READY").is_file() and manifest.is_file():
            existing = json.loads(manifest.read_text())
            if existing.get("checkpoint_digest") == checkpoint_digest:
                return final
        raise RuntimeError(f"existing cache entry is not valid: {final}")
    staging = pathlib.Path(tempfile.mkdtemp(prefix=".staging-", dir=cache_root))
    try:
        images = staging / "images"
        images.mkdir(mode=0o700)
        manifest_files = {}
        for relative, size in files:
            target = images / relative
            target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            manifest_files[relative] = {"size": size, "sha256": copy_file(source / relative, target, size)}
        metadata = {"checkpoint_digest": checkpoint_digest, "files": manifest_files, **compatibility}
        manifest = staging / "manifest.json"
        with open(manifest, "x", encoding="utf-8") as out:
            json.dump(metadata, out, sort_keys=True, separators=(",", ":"))
            out.write("\n")
            out.flush()
            os.fsync(out.fileno())
        os.chmod(manifest, 0o400)
        fsync_dir(images)
        fsync_dir(staging)
        ready = staging / "READY"
        ready_fd = os.open(ready, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
        os.fsync(ready_fd)
        os.close(ready_fd)
        fsync_dir(staging)
        os.rename(staging, final)
        fsync_dir(cache_root)
        return final
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=pathlib.Path)
    parser.add_argument("--cache-root", required=True, type=pathlib.Path)
    parser.add_argument("--compatibility-json", required=True)
    args = parser.parse_args()
    compatibility = json.loads(args.compatibility_json)
    required = {"model_digest", "container_image_digest", "criu_commit", "cuda_version", "driver_version", "gpu_model", "vllm_version", "tensor_parallel_size"}
    if required - compatibility.keys():
        raise SystemExit("missing required compatibility fields")
    print(publish(args.source, args.cache_root, compatibility))


if __name__ == "__main__":
    main()
