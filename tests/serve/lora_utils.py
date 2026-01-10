# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
MinIO Service and LoRA Test Utilities.

Provides infrastructure for LoRA adapter testing with S3-compatible storage.
Works in both CI (pre-started MinIO) and local development (auto-starts Docker).
"""

import logging
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import boto3
import requests
from botocore.client import Config
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

# LoRA testing constants
MINIO_ENDPOINT = "http://localhost:9000"
MINIO_ACCESS_KEY = "minioadmin"
MINIO_SECRET_KEY = "minioadmin"
MINIO_BUCKET = "my-loras"
DEFAULT_LORA_REPO = "codelion/Qwen3-0.6B-accuracy-recovery-lora"
DEFAULT_LORA_NAME = "codelion/Qwen3-0.6B-accuracy-recovery-lora"


@dataclass
class MinioLoraConfig:
    """Configuration for MinIO and LoRA setup."""

    endpoint: str = MINIO_ENDPOINT
    access_key: str = MINIO_ACCESS_KEY
    secret_key: str = MINIO_SECRET_KEY
    bucket: str = MINIO_BUCKET
    lora_repo: str = DEFAULT_LORA_REPO
    lora_name: str = DEFAULT_LORA_NAME
    data_dir: Optional[str] = None

    def get_s3_uri(self) -> str:
        """Get the S3 URI for the LoRA adapter."""
        return f"s3://{self.bucket}/{self.lora_name}"

    def get_env_vars(self) -> dict:
        """Get environment variables for AWS/MinIO access."""
        return {
            "AWS_ENDPOINT": self.endpoint,
            "AWS_ACCESS_KEY_ID": self.access_key,
            "AWS_SECRET_ACCESS_KEY": self.secret_key,
            "AWS_REGION": "us-east-1",
            "AWS_ALLOW_HTTP": "true",
            "DYN_LORA_ENABLED": "true",
            "DYN_LORA_PATH": "/tmp/dynamo_loras_minio_test",
        }


class MinioService:
    """
    Manages MinIO service lifecycle for tests.

    Follows a "connect or create" pattern:
    - First checks if MinIO is already running (CI or manual)
    - If not, starts a Docker container (local development)
    - Only cleans up containers it created
    """

    CONTAINER_NAME = "dynamo-minio-test"

    def __init__(self, config: MinioLoraConfig):
        self.config = config
        self._logger = logging.getLogger(self.__class__.__name__)
        self._temp_download_dir: Optional[str] = None
        self._s3_client = None
        self._owns_container: bool = False

    def _get_s3_client(self):
        """Get or create boto3 S3 client for MinIO."""
        if self._s3_client is None:
            self._s3_client = boto3.client(
                "s3",
                endpoint_url=self.config.endpoint,
                aws_access_key_id=self.config.access_key,
                aws_secret_access_key=self.config.secret_key,
                config=Config(signature_version="s3v4"),
                region_name="us-east-1",
            )
        return self._s3_client

    def _is_healthy(self) -> bool:
        """Check if MinIO is running and healthy."""
        health_url = f"{self.config.endpoint}/minio/health/live"
        try:
            response = requests.get(health_url, timeout=2)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def _is_docker_available(self) -> bool:
        """Check if Docker daemon is accessible."""
        try:
            result = subprocess.run(["docker", "info"], capture_output=True, timeout=5)
            return result.returncode == 0
        except (subprocess.SubprocessError, FileNotFoundError):
            return False

    def start(self) -> None:
        """
        Connect to MinIO service, starting a container if necessary.

        Raises:
            RuntimeError: If MinIO cannot be started or connected to.
        """
        self._logger.info("Connecting to MinIO...")

        # Check if MinIO is already running
        if self._is_healthy():
            self._logger.info("Connected to existing MinIO instance")
            self._owns_container = False
            return

        # Try to start Docker container
        if not self._is_docker_available():
            raise RuntimeError(
                "MinIO is not available and Docker is not accessible.\n"
                "Start MinIO manually:\n"
                "  docker run -d -p 9000:9000 -p 9001:9001 "
                "-e MINIO_ROOT_USER=minioadmin -e MINIO_ROOT_PASSWORD=minioadmin "
                f"--name {self.CONTAINER_NAME} "
                "quay.io/minio/minio server /data --console-address ':9001'"
            )

        self._start_container()
        self._owns_container = True
        self._logger.info("MinIO container started successfully")

    def _start_container(self) -> None:
        """Start MinIO Docker container."""
        # Clean up any existing container
        subprocess.run(
            ["docker", "rm", "-f", self.CONTAINER_NAME],
            capture_output=True,
        )

        # Create data directory
        if not self.config.data_dir:
            self.config.data_dir = tempfile.mkdtemp(prefix="minio_test_")

        cmd = [
            "docker",
            "run",
            "-d",
            "--name",
            self.CONTAINER_NAME,
            "-p",
            "9000:9000",
            "-p",
            "9001:9001",
            "-e",
            f"MINIO_ROOT_USER={self.config.access_key}",
            "-e",
            f"MINIO_ROOT_PASSWORD={self.config.secret_key}",
            "-v",
            f"{self.config.data_dir}:/data",
            "quay.io/minio/minio",
            "server",
            "/data",
            "--console-address",
            ":9001",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to start MinIO: {result.stderr}")

        self._wait_for_ready()

    def _wait_for_ready(self, timeout: int = 30) -> None:
        """Wait for MinIO to be ready."""
        start_time = time.time()

        while time.time() - start_time < timeout:
            if self._is_healthy():
                return
            time.sleep(1)

        raise RuntimeError(f"MinIO did not become ready within {timeout}s")

    def stop(self) -> None:
        """Stop MinIO container if this instance started it."""
        if not self._owns_container:
            self._logger.debug("Not stopping MinIO (not owned by this instance)")
            return

        self._logger.info("Stopping MinIO container...")
        subprocess.run(
            ["docker", "rm", "-f", self.CONTAINER_NAME],
            capture_output=True,
        )
        self._owns_container = False

    def create_bucket(self) -> None:
        """Create the S3 bucket if it doesn't exist."""
        s3_client = self._get_s3_client()

        try:
            s3_client.head_bucket(Bucket=self.config.bucket)
            self._logger.info(f"Bucket already exists: {self.config.bucket}")
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code in ("404", "NoSuchBucket"):
                self._logger.info(f"Creating bucket: {self.config.bucket}")
                try:
                    s3_client.create_bucket(Bucket=self.config.bucket)
                except ClientError as create_error:
                    raise RuntimeError(
                        f"Failed to create bucket: {create_error}"
                    ) from create_error
            else:
                raise RuntimeError(f"Failed to check bucket: {e}") from e

    def download_lora(self) -> str:
        """Download LoRA from Hugging Face Hub, returns temp directory path."""
        self._temp_download_dir = tempfile.mkdtemp(prefix="lora_download_")
        self._logger.info(
            f"Downloading LoRA {self.config.lora_repo} to {self._temp_download_dir}"
        )

        result = subprocess.run(
            [
                "huggingface-cli",
                "download",
                self.config.lora_repo,
                "--local-dir",
                self._temp_download_dir,
                "--local-dir-use-symlinks",
                "False",
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(f"Failed to download LoRA: {result.stderr}")

        # Clean up cache directory
        cache_dir = os.path.join(self._temp_download_dir, ".cache")
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)

        return self._temp_download_dir

    def upload_lora(self, local_path: str) -> None:
        """Upload LoRA to MinIO using boto3."""
        self._logger.info(
            f"Uploading LoRA to s3://{self.config.bucket}/{self.config.lora_name}"
        )

        s3_client = self._get_s3_client()
        local_path = Path(local_path)

        for file_path in local_path.rglob("*"):
            if not file_path.is_file():
                continue
            if ".git" in file_path.parts:
                continue

            relative_path = file_path.relative_to(local_path).as_posix()
            s3_key = f"{self.config.lora_name}/{relative_path}"

            try:
                s3_client.upload_file(str(file_path), self.config.bucket, s3_key)
            except ClientError as e:
                raise RuntimeError(f"Failed to upload {file_path}: {e}") from e

        self._logger.info("LoRA upload completed")

    def cleanup_download(self) -> None:
        """Clean up temporary download directory only."""
        if self._temp_download_dir and os.path.exists(self._temp_download_dir):
            shutil.rmtree(self._temp_download_dir)
            self._temp_download_dir = None

    def cleanup_temp(self) -> None:
        """Clean up all temporary directories including MinIO data dir."""
        self.cleanup_download()

        if self.config.data_dir and os.path.exists(self.config.data_dir):
            shutil.rmtree(self.config.data_dir, ignore_errors=True)


def load_lora_adapter(
    system_port: int, lora_name: str, s3_uri: str, timeout: int = 60
) -> None:
    """Load a LoRA adapter via the system API."""
    url = f"http://localhost:{system_port}/v1/loras"
    payload = {"lora_name": lora_name, "source": {"uri": s3_uri}}

    logger.info(f"Loading LoRA adapter: {lora_name} from {s3_uri}")

    response = requests.post(url, json=payload, timeout=timeout)
    if response.status_code != 200:
        raise RuntimeError(
            f"Failed to load LoRA adapter: {response.status_code} - {response.text}"
        )

    logger.info(f"LoRA adapter loaded successfully: {response.json()}")
