# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Minimal Python wrapper around Rust LoRA core with extension points for custom sources.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Protocol

from dynamo._core import PyLoRACache, PyLoRADownloader


class LoRASourceProtocol(Protocol):
    """
    Protocol for custom Python LoRA sources.
    Users can implement this to add custom sources.
    """

    async def download(self, lora_uri: str, dest_path: Path) -> Path:
        """Download LoRA to dest_path, return actual path"""
        ...

    async def exists(self, lora_uri: str) -> bool:
        """Check if LoRA exists in this source"""
        ...


class LoRAManager:
    """
    Minimal Python wrapper around Rust core with extension points.

    The manager uses the Rust-based PyLoRADownloader for S3 and local file sources,
    and allows registering custom Python sources for other protocols.
    """

    def __init__(self, cache_path: Optional[Path] = None):
        """
        Initialize LoRA manager.

        Args:
            cache_path: Optional custom cache path. If not provided, uses DYN_LORA_PATH env var.
        """
        # Use Rust cache
        if cache_path:
            self.cache = PyLoRACache(str(cache_path))
        else:
            self.cache = PyLoRACache.from_env()

        # Setup downloader with Rust sources (local + S3)
        self.downloader = PyLoRADownloader.create_default()

        # Extension point: custom sources
        self.custom_sources: Dict[str, LoRASourceProtocol] = {}

    def register_custom_source(self, name: str, source: LoRASourceProtocol):
        """
        Extension point: Register custom Python source.

        Args:
            name: Name for the custom source
            source: LoRA source implementing LoRASourceProtocol
        """
        self.custom_sources[name] = source

    async def download_lora(
        self, lora_uri: str, source_hint: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Download LoRA if needed, return local path.

        Args:
            lora_uri: Source URI (file://, s3://, or custom scheme)
            source_hint: Optional hint to use specific custom source

        Returns:
            Dictionary with:
                - status: "success" or "error"
                - local_path: Local path to LoRA (if successful)
                - message: Error message (if error)
        """
        try:
            # Try custom source first if hint provided
            if source_hint and source_hint in self.custom_sources:
                source = self.custom_sources[source_hint]
                cache_key = self._uri_to_cache_key(lora_uri)
                dest_path = Path(self.cache.get_cache_path(cache_key))
                local_path = await source.download(lora_uri, dest_path)
            else:
                # Use Rust downloader (handles file:// and s3://)
                local_path = Path(self.downloader.download_if_needed(lora_uri))

            return {"status": "success", "local_path": str(local_path)}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def is_cached(self, lora_uri: str) -> bool:
        """
        Check if LoRA is already cached locally.

        Args:
            lora_uri: Source URI

        Returns:
            True if cached, False otherwise
        """
        cache_key = self._uri_to_cache_key(lora_uri)
        return self.cache.is_cached(cache_key)

    def _uri_to_cache_key(self, uri: str) -> str:
        """
        Convert URI to cache key.
        For file:// URIs, this is just for consistency - local files aren't cached.
        """
        # For s3://bucket/path/to/lora -> path/to/lora
        # For file:// URIs -> not used (files aren't cached)
        if uri.startswith("s3://") or uri.startswith("gs://"):
            # Extract path component
            parts = uri.split("/", 3)
            if len(parts) > 3:
                return parts[3]
        elif uri.startswith("http://") or uri.startswith("https://"):
            # Use path component
            from urllib.parse import urlparse

            parsed = urlparse(uri)
            return parsed.path.lstrip("/")

        # Fallback: sanitize the URI
        return uri.replace("://", "_").replace("/", "_").replace("\\", "_")
