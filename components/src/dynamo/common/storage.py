# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Dynamo Common Storage Module

Filesystem Spec (fsspec) is a project to provide a unified pythonic interface to local, remote and embedded file systems and bytes storage.

https://filesystem-spec.readthedocs.io/en/latest/index.html#who-uses-fsspec

Configuration for the storage:

Local Filesystem:
    1. fs_url MUST contain a root path - path must be accessible and writable

S3:
    1. If you want to use S3 please install additional dependencies: fsspec[s3]
    2. fs_url MUST contain a bucket name
    3. Configure credentials     https://s3fs.readthedocs.io/en/latest/?badge=latest#credentials

        a) AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, and AWS_SESSION_TOKEN environment variables

        b) configuration files such as ~/.aws/credentials

        c) for nodes on EC2, the IAM metadata provider

        d) for S3 compatible storage, you can use the following environment variables:
            # export FSSPEC_S3_ENDPOINT_URL=https://...
            # export FSSPEC_S3_KEY='miniokey...'
            # export FSSPEC_S3_SECRET='asecretkey...'



"""
import fsspec
from fsspec.implementations.dirfs import DirFileSystem


def get_fs(fs_url: str) -> DirFileSystem:
    """
    Initialize fsspec filesystem for the given URL.

    Args:
        fs_url: The URL of the filesystem to initialize. e.g. s3://bucket, gs://bucket, file:///local/path

    Returns:
        The initialized DirFileSystem wrapper for the filesystem.

        fs.fs.protocol to get the protocol of the filesystem
        fs.path to get the bucket or root path

        path to the object in the filesystem - f"{fs.fs.protocol}://{fs.path}/{path}"
    """
    # Extract protocol from URL (s3://, gs://, az://, file://)
    fs_url_parts = fs_url.split("://")
    protocol = fs_url_parts[0] if "://" in fs_url else "file"

    # ... or bucket name
    root_path = fs_url_parts[1] if len(fs_url_parts) > 1 else "/"

    fs_opts = {}
    if protocol in "file":
        # create directory for local filesystem
        fs_opts = {"auto_mkdir": True}

    return DirFileSystem(fs=fsspec.filesystem(protocol, **fs_opts), path=root_path)
