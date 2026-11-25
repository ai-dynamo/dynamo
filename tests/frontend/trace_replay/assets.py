# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fetch only immutable, hash-verified frontend assets, independently of Dynamo."""

from __future__ import annotations

import fcntl
import os
import re
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlsplit

import requests

from .contract import ReplayError, digest, file_digest, validate_profile


@dataclass(frozen=True)
class AssetBundle:
    model_dir: Path
    encoder: Path
    sha256: dict[str, str]


class _AssetSession(requests.Session):
    """Honor explicit network settings without implicit netrc credentials."""

    def __init__(self) -> None:
        super().__init__()
        self.trust_env = False

    def send(
        self, request: requests.PreparedRequest, **kwargs: Any
    ) -> requests.Response:
        # resolve_redirects calls send again: recompute NO_PROXY and CA settings
        # for every hop. Session's standard rebuild_auth strips cross-host auth.
        target = urlsplit(request.url or "")
        if (
            target.scheme != "https"
            or target.username is not None
            or target.password is not None
        ):
            raise ReplayError("asset download requires HTTPS without URL credentials")
        kwargs["proxies"] = requests.utils.get_environ_proxies(request.url or "")
        kwargs["verify"] = (
            os.environ.get("REQUESTS_CA_BUNDLE")
            or os.environ.get("CURL_CA_BUNDLE")
            or True
        )
        return super().send(request, **kwargs)


def _files(profile: dict[str, Any], reference: dict[str, Any] | None) -> dict[str, str]:
    validate_profile(profile)
    model = profile["model"]
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", model["repo_id"]):
        raise ReplayError("asset repository must be a Hugging Face owner/name")
    files = dict(model["assets"])
    if reference is not None:
        name = reference.get("path")
        if (
            any(reference.get(key) != model[key] for key in ("repo_id", "revision"))
            or not isinstance(name, str)
            or not name
            or any(
                not re.fullmatch(r"[A-Za-z0-9_.-]+", part) or part in {".", ".."}
                for part in name.split("/")
            )
            or not re.fullmatch(r"[0-9a-f]{64}", str(reference.get("sha256", "")))
            or name in files
        ):
            raise ReplayError(
                "asset encoder must be a pinned relative file in the model repository"
            )
        files[name] = reference["sha256"]
    return files


def cache_identity(profile: dict[str, Any], reference: dict[str, Any]) -> str:
    """Identify downloaded bytes without local paths, aliases or runtime settings."""
    files = _files(profile, reference)
    return digest(
        {
            "repo_id": profile["model"]["repo_id"],
            "revision": profile["model"]["revision"],
            "files": files,
        }
    )


def _existing(destination: Path, expected: str, name: str) -> bool:
    # Follow valid HF snapshot symlinks, but never treat dangling links as misses.
    if not destination.exists() and not destination.is_symlink():
        return False
    if not destination.is_file() or file_digest(destination) != expected:
        raise ReplayError("existing model asset has the wrong hash: " + name)
    return True


@contextmanager
def _publication_lock(directory: Path) -> Iterator[None]:
    descriptor = os.open(
        directory / ".frontend-replay-assets.lock", os.O_RDWR | os.O_CREAT, 0o600
    )
    try:
        deadline = time.monotonic() + 10
        while True:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise ReplayError(
                        "asset publication is busy; retry with the same cache"
                    )
                time.sleep(0.05)
        try:
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def _publish(temporary: Path, destination: Path, expected: str, name: str) -> None:
    # Advisory ownership works on bind mounts and filesystems without hardlinks.
    # A second helper must recheck the winner while holding the same lock.
    with _publication_lock(destination.parent):
        if not _existing(destination, expected, name):
            os.replace(temporary, destination)


def ensure_assets(
    profile: dict[str, Any], model_dir: Path, *, reference: dict[str, Any] | None = None
) -> dict[str, str]:
    """Prepare files in place; reuse matching files and reject corrupt evidence."""
    files = _files(profile, reference)
    token = os.environ.get("HF_TOKEN", "")
    if token and any(not 0x21 <= ord(character) <= 0x7E for character in token):
        raise ReplayError("HF_TOKEN must contain visible ASCII header characters")
    model = profile["model"]
    with _AssetSession() as session:
        for name, expected in files.items():
            destination = model_dir / name
            if _existing(destination, expected, name):
                continue
            destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=".asset-", dir=destination.parent
            )
            temporary = Path(temporary_name)
            url = f"https://huggingface.co/{model['repo_id']}/resolve/{model['revision']}/{name}"
            headers = {"Authorization": "Bearer " + token} if token else {}
            try:
                with os.fdopen(descriptor, "wb") as stream:
                    try:
                        with session.get(
                            url, headers=headers, stream=True, timeout=(10, 120)
                        ) as response:
                            response.raise_for_status()
                            for block in response.iter_content(1024 * 1024):
                                stream.write(block)
                    except requests.RequestException as exc:
                        raise ReplayError(
                            f"asset download failed: {name} ({type(exc).__name__})"
                        ) from None
                if file_digest(temporary) != expected:
                    raise ReplayError(
                        "downloaded model asset has the wrong hash: " + name
                    )
                _publish(temporary, destination, expected, name)
            finally:
                temporary.unlink(missing_ok=True)
    return files


def resolve_cached_assets(
    profile: dict[str, Any], reference: dict[str, Any], cache_root: Path
) -> AssetBundle:
    """Return one verified immutable model/encoder cache entry."""
    model_dir = cache_root / cache_identity(profile, reference)
    hashes = ensure_assets(profile, model_dir, reference=reference)
    return AssetBundle(
        model_dir=model_dir, encoder=model_dir / reference["path"], sha256=hashes
    )
