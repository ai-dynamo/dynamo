# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


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
    """Configuration for MinIO and LoRA setup"""

    endpoint: str = MINIO_ENDPOINT
    access_key: str = MINIO_ACCESS_KEY
    secret_key: str = MINIO_SECRET_KEY
    bucket: str = MINIO_BUCKET
    lora_repo: str = DEFAULT_LORA_REPO
    lora_name: str = DEFAULT_LORA_NAME
    data_dir: Optional[str] = None

    def get_s3_uri(self) -> str:
        """Get the S3 URI for the LoRA adapter"""
        return f"s3://{self.bucket}/{self.lora_name}"

    def get_env_vars(self) -> dict:
        """Get environment variables for AWS/MinIO access"""
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
    """Connects to MinIO service for tests.

    In CI, MinIO is pre-started by the workflow (no Docker access needed).
    For local development, MinIO can be started manually or via docker-compose.
    """

    CONTAINER_NAME = "dynamo-minio-test"

    def __init__(self, config: MinioLoraConfig):
        self.config = config
        self._logger = logging.getLogger(self.__class__.__name__)
        self._temp_download_dir: Optional[str] = None
        self._s3_client = None

    def _get_s3_client(self):
        """Get or create boto3 S3 client for MinIO"""
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

    def start(self) -> None:
        """Connect to MinIO service (started by CI workflow or manually)"""
        self._logger.info("Connecting to MinIO service...")

        # Check if MinIO is available (pre-started by CI or running locally)
        if not self._is_minio_ready():
            # For local development, try to start MinIO if docker is available
            if self._can_use_docker():
                self._start_local_minio()
            else:
                raise RuntimeError(
                    "MinIO is not available. In CI, it should be pre-started by the workflow. "
                    "For local development, start MinIO manually: "
                    "docker run -d -p 9000:9000 -p 9001:9001 "
                    "-e MINIO_ROOT_USER=minioadmin -e MINIO_ROOT_PASSWORD=minioadmin "
                    "quay.io/minio/minio server /data --console-address ':9001'"
                )

        self._logger.info("MinIO service is ready")

    def _is_minio_ready(self) -> bool:
        """Check if MinIO is already running and ready"""
        health_url = f"{self.config.endpoint}/minio/health/live"
        try:
            response = requests.get(health_url, timeout=2)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def _can_use_docker(self) -> bool:
        """Check if docker is available (for local development)"""
        try:
            result = subprocess.run(["docker", "info"], capture_output=True, timeout=5)
            return result.returncode == 0
        except (subprocess.SubprocessError, FileNotFoundError):
            return False

    def _start_local_minio(self) -> None:
        """Start MinIO locally for development (not used in CI)"""
        self._logger.info("Starting local MinIO container for development...")

        # Create data directory
        if not self.config.data_dir:
            self.config.data_dir = tempfile.mkdtemp(prefix="minio_test_")

        # Stop existing container if running
        subprocess.run(
            ["docker", "rm", "-f", self.CONTAINER_NAME],
            capture_output=True,
        )

        # Start MinIO container
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

        # Wait for MinIO to be ready
        self._wait_for_ready()

    def _wait_for_ready(self, timeout: int = 30) -> None:
        """Wait for MinIO to be ready"""
        health_url = f"{self.config.endpoint}/minio/health/live"
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                response = requests.get(health_url, timeout=2)
                if response.status_code == 200:
                    return
            except requests.RequestException:
                pass
            time.sleep(1)

        raise RuntimeError(f"MinIO did not become ready within {timeout}s")

    def stop(self) -> None:
        """Stop MinIO container (only if started locally for development)"""
        # In CI, MinIO is managed by the workflow - don't try to stop it
        if os.environ.get("MINIO_AVAILABLE") == "true":
            self._logger.info("MinIO is managed by CI workflow, skipping stop")
            return

        self._logger.info("Stopping local MinIO container...")
        subprocess.run(
            ["docker", "rm", "-f", self.CONTAINER_NAME],
            capture_output=True,
        )

    def create_bucket(self) -> None:
        """Create the S3 bucket using boto3"""
        s3_client = self._get_s3_client()

        # Check if bucket exists
        try:
            s3_client.head_bucket(Bucket=self.config.bucket)
            self._logger.info(f"Bucket already exists: {self.config.bucket}")
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code in ("404", "NoSuchBucket"):
                # Create bucket
                self._logger.info(f"Creating bucket: {self.config.bucket}")
                try:
                    s3_client.create_bucket(Bucket=self.config.bucket)
                    self._logger.info(f"Bucket created: {self.config.bucket}")
                except ClientError as create_error:
                    raise RuntimeError(
                        f"Failed to create bucket: {create_error}"
                    ) from create_error
            else:
                raise RuntimeError(f"Failed to check bucket: {e}") from e

    def download_lora(self) -> str:
        """Download LoRA from Hugging Face Hub, returns temp directory path"""
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
        """Upload LoRA to MinIO using boto3"""
        self._logger.info(
            f"Uploading LoRA to s3://{self.config.bucket}/{self.config.lora_name}"
        )

        s3_client = self._get_s3_client()
        local_path = Path(local_path)

        # Walk through the directory and upload all files
        for file_path in local_path.rglob("*"):
            if file_path.is_file():
                # Skip git files
                if ".git" in str(file_path):
                    continue

                # Calculate the S3 key (relative path from local_path)
                # Use as_posix() to ensure forward slashes in S3 key
                relative_path = file_path.relative_to(local_path).as_posix()
                s3_key = f"{self.config.lora_name}/{relative_path}"

                self._logger.debug(f"Uploading {file_path} to {s3_key}")
                try:
                    s3_client.upload_file(str(file_path), self.config.bucket, s3_key)
                except ClientError as e:
                    raise RuntimeError(f"Failed to upload {file_path}: {e}") from e

        self._logger.info("LoRA upload completed")

    def cleanup_download(self) -> None:
        """Clean up temporary download directory only"""
        if self._temp_download_dir and os.path.exists(self._temp_download_dir):
            shutil.rmtree(self._temp_download_dir)
            self._temp_download_dir = None

    def cleanup_temp(self) -> None:
        """Clean up all temporary directories including MinIO data dir"""
        self.cleanup_download()

        if self.config.data_dir and os.path.exists(self.config.data_dir):
            shutil.rmtree(self.config.data_dir, ignore_errors=True)


def load_lora_adapter(
    system_port: int, lora_name: str, s3_uri: str, timeout: int = 60
) -> None:
    """Load a LoRA adapter via the system API"""
    url = f"http://localhost:{system_port}/v1/loras"
    payload = {"lora_name": lora_name, "source": {"uri": s3_uri}}

    logger.info(f"Loading LoRA adapter: {lora_name} from {s3_uri}")

    response = requests.post(url, json=payload, timeout=timeout)
    if response.status_code != 200:
        raise RuntimeError(
            f"Failed to load LoRA adapter: {response.status_code} - {response.text}"
        )

    logger.info(f"LoRA adapter loaded successfully: {response.json()}")
