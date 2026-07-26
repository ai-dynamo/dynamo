# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""S3 object store for GMS weight snapshots.

A GMS snapshot is a directory (manifest.json, gms_metadata.json, and the raw
shard_*.bin files). Mirroring that tree to any S3-compatible object store lets
workers restore weights from the bucket instead of requiring a shared
filesystem or re-striped local NVMe, and aligns with the parallel restore path
that saturates storage bandwidth. No vendor-specific features are used.

TODO(future): downloads land in host memory via fsspec before being applied to
GPU memory. Wire up an RDMA / GPUDirect Storage (GDS) restore path — routing
object reads through the existing NIXL-GDS transfer backend (see
snapshot/transfer.py) — to stream shards directly into GPU memory and bypass
the host bounce buffer.

Credentials and endpoint follow the same fsspec conventions used elsewhere in
the repo (see dynamo.common.storage):

    export FSSPEC_S3_ENDPOINT_URL=https://s3.example.com
    export FSSPEC_S3_KEY=...        # or AWS_ACCESS_KEY_ID
    export FSSPEC_S3_SECRET=...     # or AWS_SECRET_ACCESS_KEY

Object storage requires the optional dependency ``fsspec[s3]``.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

logger = logging.getLogger(__name__)

# Uploaded last so that exists() only reports a complete, restorable artifact.
MANIFEST_FILENAME = "manifest.json"

_DEFAULT_CONCURRENCY = 8


def make_key_prefix(prefix: str, snapshot_id: str, version: str = "1") -> str:
    """Build an object key prefix "<prefix/>?<snapshot_id>/versions/<version>".

    Mirrors the Go snapshot agent's layout so a single bucket can host both
    container checkpoints and GMS weight snapshots under stable, versioned keys.
    """
    parts: List[str] = []
    cleaned_prefix = (prefix or "").strip().strip("/")
    if cleaned_prefix:
        parts.append(cleaned_prefix)
    parts.append((snapshot_id or "").strip("/"))
    parts.append("versions")
    parts.append((version or "1").strip() or "1")
    return "/".join(parts)


class S3ArtifactStore:
    """Mirror a GMS snapshot directory to/from an S3-compatible object store."""

    def __init__(
        self,
        bucket: str,
        *,
        endpoint_url: Optional[str] = None,
        key: Optional[str] = None,
        secret: Optional[str] = None,
        token: Optional[str] = None,
        region: Optional[str] = None,
        use_ssl: bool = True,
        force_path_style: bool = False,
        concurrency: int = _DEFAULT_CONCURRENCY,
    ) -> None:
        bucket = (bucket or "").strip().strip("/")
        if not bucket:
            raise ValueError("bucket is required")
        try:
            import fsspec
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise RuntimeError(
                "object storage requires the optional dependency 'fsspec[s3]'"
            ) from exc

        client_kwargs = {}
        if endpoint_url:
            client_kwargs["endpoint_url"] = endpoint_url
        if region:
            client_kwargs["region_name"] = region

        # Default to virtual-hosted (bucket DNS) addressing; opt in to path-style
        # for endpoints without bucket DNS support.
        addressing_style = "path" if force_path_style else "virtual"
        config_kwargs = {"s3": {"addressing_style": addressing_style}}

        self._bucket = bucket
        self._concurrency = concurrency if concurrency > 0 else _DEFAULT_CONCURRENCY
        self._fs = fsspec.filesystem(
            "s3",
            key=key,
            secret=secret,
            token=token,
            use_ssl=use_ssl,
            client_kwargs=client_kwargs or None,
            config_kwargs=config_kwargs,
        )

    def _remote(self, key: str) -> str:
        return f"{self._bucket}/{key}"

    @staticmethod
    def _normalize_prefix(key_prefix: str) -> str:
        return (key_prefix or "").strip().strip("/")

    def upload_tree(self, local_dir: str, key_prefix: str) -> None:
        """Upload every regular file under ``local_dir`` to ``key_prefix``.

        The manifest is uploaded last so a concurrent restore that observes it is
        guaranteed the full artifact has landed.
        """
        prefix = self._normalize_prefix(key_prefix)
        rel_files: List[str] = []
        manifest_seen = False
        for root, _dirs, names in os.walk(local_dir):
            for name in names:
                abs_path = os.path.join(root, name)
                if not os.path.isfile(abs_path):
                    continue
                rel = os.path.relpath(abs_path, local_dir)
                if rel.replace(os.sep, "/") == MANIFEST_FILENAME:
                    manifest_seen = True
                    continue
                rel_files.append(rel)
        if not manifest_seen:
            raise FileNotFoundError(
                f"snapshot dir {local_dir} has no {MANIFEST_FILENAME}"
            )

        # Clear any previous artifact at this prefix first: drops the old manifest
        # (so exists() stays false for the whole upload window) and removes
        # orphaned objects from a prior snapshot with more/different shards.
        self.remove(prefix)

        def _put(rel: str) -> None:
            key = (
                f"{prefix}/{rel.replace(os.sep, '/')}"
                if prefix
                else rel.replace(os.sep, "/")
            )
            self._fs.put_file(os.path.join(local_dir, rel), self._remote(key))

        if rel_files:
            with ThreadPoolExecutor(max_workers=self._concurrency) as pool:
                list(pool.map(_put, rel_files))

        # Manifest last: the completion marker.
        _put(MANIFEST_FILENAME)
        logger.info(
            "Uploaded GMS snapshot to s3://%s/%s (%d files)",
            self._bucket,
            prefix,
            len(rel_files) + 1,
        )

    def download_tree(self, key_prefix: str, local_dir: str) -> None:
        """Download every object under ``key_prefix`` into ``local_dir``."""
        prefix = self._normalize_prefix(key_prefix)
        os.makedirs(local_dir, exist_ok=True)
        remote_prefix = self._remote(prefix)
        keys = self._fs.find(remote_prefix)
        if not keys:
            raise FileNotFoundError(f"no objects under s3://{self._bucket}/{prefix}")

        bucket_prefix = f"{self._bucket}/{prefix}".rstrip("/")

        def _get(key: str) -> None:
            rel = key[len(bucket_prefix) :].lstrip("/")
            if not rel:
                return
            dest = self._safe_dest(local_dir, rel)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            self._fs.get_file(key, dest)

        with ThreadPoolExecutor(max_workers=self._concurrency) as pool:
            list(pool.map(_get, keys))
        logger.info(
            "Downloaded GMS snapshot from s3://%s/%s (%d objects)",
            self._bucket,
            prefix,
            len(keys),
        )

    def exists(self, key_prefix: str) -> bool:
        """Report whether a complete artifact (its manifest) exists."""
        prefix = self._normalize_prefix(key_prefix)
        key = f"{prefix}/{MANIFEST_FILENAME}" if prefix else MANIFEST_FILENAME
        return bool(self._fs.exists(self._remote(key)))

    def remove(self, key_prefix: str) -> None:
        """Delete every object under ``key_prefix``. Missing prefixes are ignored."""
        prefix = self._normalize_prefix(key_prefix)
        remote_prefix = self._remote(prefix)
        try:
            self._fs.rm(remote_prefix, recursive=True)
        except FileNotFoundError:
            return

    @staticmethod
    def _safe_dest(local_dir: str, rel: str) -> str:
        """Resolve ``rel`` under ``local_dir``, rejecting path traversal."""
        clean_dir = os.path.abspath(local_dir)
        dest = os.path.abspath(os.path.join(clean_dir, rel))
        if dest != clean_dir and not dest.startswith(clean_dir + os.sep):
            raise ValueError(f"object key {rel!r} escapes restore dir {clean_dir}")
        return dest


def store_from_env(
    bucket: Optional[str] = None,
    *,
    concurrency: int = _DEFAULT_CONCURRENCY,
) -> Optional[S3ArtifactStore]:
    """Build an S3ArtifactStore from environment, or None if not configured.

    Honors the fsspec S3 variables (FSSPEC_S3_*) and the standard AWS_* names.
    Returns None when no bucket is configured, signaling the caller to use the
    local disk path.
    """
    bucket = bucket or os.environ.get("GMS_SNAPSHOT_S3_BUCKET")
    if not bucket:
        return None
    endpoint = os.environ.get("FSSPEC_S3_ENDPOINT_URL") or os.environ.get(
        "AWS_ENDPOINT_URL"
    )
    key = os.environ.get("FSSPEC_S3_KEY") or os.environ.get("AWS_ACCESS_KEY_ID")
    secret = os.environ.get("FSSPEC_S3_SECRET") or os.environ.get(
        "AWS_SECRET_ACCESS_KEY"
    )
    token = os.environ.get("AWS_SESSION_TOKEN")
    region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
    use_ssl = os.environ.get("GMS_SNAPSHOT_S3_USE_SSL", "1") not in (
        "0",
        "false",
        "False",
    )
    force_path_style = os.environ.get("GMS_SNAPSHOT_S3_FORCE_PATH_STYLE", "0") in (
        "1",
        "true",
        "True",
    )
    return S3ArtifactStore(
        bucket,
        endpoint_url=endpoint,
        key=key,
        secret=secret,
        token=token,
        region=region,
        use_ssl=use_ssl,
        force_path_style=force_path_style,
        concurrency=concurrency,
    )
