# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for dynamo.common.storage module."""

from unittest.mock import MagicMock, patch

import pytest

from dynamo.common.storage import get_fs, get_media_url, upload_to_fs

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


class TestGetFs:
    """Tests for get_fs() filesystem initialization."""

    def test_local_file_url(self, tmp_path):
        """Test file:// URL returns DirFileSystem with correct path."""
        media_dir = tmp_path / "test_media"
        fs = get_fs(f"file://{media_dir}")
        assert fs.path == str(media_dir)
        protocol = fs.fs.protocol
        if isinstance(protocol, (list, tuple)):
            protocol = protocol[0]
        assert protocol == "file"

    def test_local_file_url_auto_mkdir(self, tmp_path):
        """Test file:// URL enables auto_mkdir on underlying filesystem."""
        media_dir = tmp_path / "test_media"
        fs = get_fs(f"file://{media_dir}")
        assert fs.fs.auto_mkdir is True

    def test_no_protocol_defaults_to_file(self, tmp_path):
        """Test URL without protocol defaults to file."""
        media_dir = tmp_path / "test_media"
        fs = get_fs(str(media_dir))
        protocol = fs.fs.protocol
        if isinstance(protocol, (list, tuple)):
            protocol = protocol[0]
        assert protocol == "file"

    def test_s3_url_protocol(self):
        """Test s3:// URL extracts correct protocol and bucket path."""
        with patch("dynamo.common.storage.fsspec.filesystem") as mock_fsspec, patch(
            "dynamo.common.storage.DirFileSystem"
        ) as mock_dirfs:
            mock_inner_fs = MagicMock(protocol="s3")
            mock_fsspec.return_value = mock_inner_fs
            get_fs("s3://my-bucket/prefix")
            mock_fsspec.assert_called_once_with("s3")
            mock_dirfs.assert_called_once_with(
                fs=mock_inner_fs, path="my-bucket/prefix"
            )

    def test_gs_url_protocol(self):
        """Test gs:// URL extracts correct protocol and path."""
        with patch("dynamo.common.storage.fsspec.filesystem") as mock_fsspec, patch(
            "dynamo.common.storage.DirFileSystem"
        ) as mock_dirfs:
            mock_inner_fs = MagicMock(protocol="gs")
            mock_fsspec.return_value = mock_inner_fs
            get_fs("gs://my-gcs-bucket/data")
            mock_fsspec.assert_called_once_with("gs")
            mock_dirfs.assert_called_once_with(
                fs=mock_inner_fs, path="my-gcs-bucket/data"
            )


class TestGetMediaUrl:
    """Tests for get_media_url() URL construction."""

    def _make_fs(self, protocol="file", path="/tmp/media"):  # noqa: S108
        """Create a mock DirFileSystem."""
        fs = MagicMock()
        fs.fs.protocol = protocol
        fs.path = path
        return fs

    def test_base_url_rewrite(self):
        """Test that base_url takes precedence over protocol fallback."""
        fs = self._make_fs()
        url = get_media_url(
            fs, "videos/req-123.mp4", base_url="https://cdn.example.com/media"
        )
        assert url == "https://cdn.example.com/media/videos/req-123.mp4"

    def test_base_url_trailing_slash_stripped(self):
        """Test that trailing slash on base_url is normalized."""
        fs = self._make_fs()
        url = get_media_url(fs, "images/test.png", base_url="https://cdn.example.com/")
        assert url == "https://cdn.example.com/images/test.png"

    def test_protocol_fallback_file(self):
        """Test URL construction from file:// protocol when no base_url."""
        fs = self._make_fs(protocol="file", path="/tmp/dynamo_media")  # noqa: S108
        url = get_media_url(fs, "videos/req-123.mp4")
        assert url == "file:///tmp/dynamo_media/videos/req-123.mp4"

    def test_protocol_fallback_s3(self):
        """Test URL construction from s3:// protocol when no base_url."""
        fs = self._make_fs(protocol="s3", path="my-bucket/prefix")
        url = get_media_url(fs, "images/img.png")
        assert url == "s3://my-bucket/prefix/images/img.png"

    def test_tuple_protocol_uses_first(self):
        """Test that tuple protocol (e.g., ('s3', 's3a')) uses first element."""
        fs = self._make_fs()
        fs.fs.protocol = ("s3", "s3a")
        fs.path = "my-bucket"
        url = get_media_url(fs, "file.mp4")
        assert url == "s3://my-bucket/file.mp4"

    def test_list_protocol_uses_first(self):
        """Test that list protocol uses first element."""
        fs = self._make_fs()
        fs.fs.protocol = ["gs", "gcs"]
        fs.path = "my-gcs-bucket"
        url = get_media_url(fs, "file.mp4")
        assert url == "gs://my-gcs-bucket/file.mp4"

    def test_base_url_none_falls_back_to_protocol(self):
        """Test that None base_url triggers protocol fallback."""
        fs = self._make_fs()
        url = get_media_url(fs, "test.png", base_url=None)
        assert url == "file:///tmp/media/test.png"

    def test_base_url_empty_string_falls_back_to_protocol(self):
        """Test that empty string base_url triggers protocol fallback."""
        fs = self._make_fs()
        url = get_media_url(fs, "test.png", base_url="")
        assert url == "file:///tmp/media/test.png"


class TestUploadToFs:
    """Tests for upload_to_fs() async upload + URL construction."""

    def _make_fs(self, protocol="file", path="/tmp/media"):  # noqa: S108
        """Create a mock DirFileSystem."""
        fs = MagicMock()
        fs.fs.protocol = protocol
        fs.path = path
        fs.pipe = MagicMock()
        return fs

    @pytest.mark.asyncio
    async def test_calls_pipe_with_correct_args(self):
        """Test that fs.pipe is called with storage_path and data."""
        fs = self._make_fs()
        data = b"test image bytes"

        await upload_to_fs(fs, "images/test.png", data)

        fs.pipe.assert_called_once_with("images/test.png", data)

    @pytest.mark.asyncio
    async def test_returns_url_with_base_url(self):
        """Test that returned URL uses base_url when provided."""
        fs = self._make_fs()
        url = await upload_to_fs(
            fs, "videos/req-123.mp4", b"video", base_url="https://cdn.example.com"
        )
        assert url == "https://cdn.example.com/videos/req-123.mp4"

    @pytest.mark.asyncio
    async def test_returns_url_with_protocol_fallback(self):
        """Test that returned URL uses protocol when no base_url."""
        fs = self._make_fs(protocol="s3", path="my-bucket/prefix")
        url = await upload_to_fs(fs, "images/img.png", b"image")
        assert url == "s3://my-bucket/prefix/images/img.png"

    @pytest.mark.asyncio
    async def test_pipe_called_before_url_returned(self):
        """Test that pipe() is called (data uploaded) before URL is returned."""
        fs = self._make_fs()
        call_order = []
        original_pipe = fs.pipe

        def tracking_pipe(*args, **kwargs):
            call_order.append("pipe")
            return original_pipe(*args, **kwargs)

        fs.pipe = tracking_pipe

        url = await upload_to_fs(fs, "test.png", b"data")
        call_order.append("url_returned")

        assert call_order == ["pipe", "url_returned"]
        assert url
