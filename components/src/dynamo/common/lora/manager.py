# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Minimal Python wrapper around Rust LoRA core with extension points for custom sources.
"""

import asyncio
from pathlib import Path
from typing import Any, Dict, Optional, Protocol

from dynamo.llm import LoRADownloader


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

    The manager uses the Rust-based LoRADownloader for S3 and local file sources,
    and allows registering custom Python sources for other protocols.
    """

    def __init__(self, cache_path: Optional[Path] = None):
        """
        Initialize LoRA manager.

        Args:
            cache_path: Optional custom cache path. If not provided, uses DYN_LORA_PATH env var.
        """
        # Single unified Rust interface handles both downloading and caching
        cache_str = str(cache_path) if cache_path else None
        self._downloader = LoRADownloader(cache_str)

        # Extension point: custom sources
        self._custom_sources: Dict[str, LoRASourceProtocol] = {}

    def register_custom_source(self, scheme: str, source: LoRASourceProtocol):
        """
        Register a custom Python source for a URI scheme.

        Args:
            scheme: URI scheme without "://" (e.g., "hf" for hf:// URIs)
            source: LoRA source implementing LoRASourceProtocol
        """
        self._custom_sources[scheme] = source

    async def download_lora(self, lora_uri: str) -> Dict[str, Any]:
        """
        Download LoRA if needed, return local path.

        The source is inferred from the URI scheme:
        - file:// -> Local filesystem (Rust)
        - s3:// -> S3 (Rust)
        - Custom schemes -> Registered Python sources

        Args:
            lora_uri: Source URI (file://, s3://, or custom scheme)

        Returns:
            Dictionary with:
                - status: "success" or "error"
                - local_path: Local path to LoRA (if successful)
                - message: Error message (if error)
        """
        try:
            # Extract scheme from URI
            scheme = lora_uri.split("://")[0] if "://" in lora_uri else None

            # Check for custom Python source matching the scheme
            if scheme and scheme in self._custom_sources:
                source = self._custom_sources[scheme]

                if not await source.exists(lora_uri):
                    return {
                        "status": "error",
                        "message": f"LoRA not found at {lora_uri}",
                    }

                cache_key = self._uri_to_cache_key(lora_uri)
                dest_path = Path(self._downloader.get_cache_path(cache_key))
                local_path = await source.download(lora_uri, dest_path)
            else:
                # Use Rust downloader (handles file:// and s3://)
                local_path = Path(await self._downloader.download_if_needed(lora_uri))

            return {"status": "success", "local_path": str(local_path)}
        except asyncio.CancelledError:
            raise
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def is_cached(self, lora_uri: str) -> bool:
        """Check if LoRA is already cached locally."""
        cache_key = LoRADownloader.uri_to_cache_key(lora_uri)
        return self._downloader.is_cached(cache_key)

    def _uri_to_cache_key(self, uri: str) -> str:
        """Convert URI to cache key. Delegates to Rust for consistency."""
        return LoRADownloader.uri_to_cache_key(uri)
