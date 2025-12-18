# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import logging
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from typing import Optional

import requests

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
    """Manages MinIO Docker container lifecycle for tests"""

    CONTAINER_NAME = "dynamo-minio-test"

    def __init__(self, config: MinioLoraConfig):
        self.config = config
        self._logger = logging.getLogger(self.__class__.__name__)
        self._temp_download_dir: Optional[str] = None

    def start(self) -> None:
        """Start MinIO container"""
        self._logger.info("Starting MinIO container...")

        # Create data directory
        if self.config.data_dir:
            data_dir = self.config.data_dir
        else:
            data_dir = tempfile.mkdtemp(prefix="minio_test_")
        self.config.data_dir = data_dir

        # Stop existing container if running
        self.stop()

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
            f"{data_dir}:/data",
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
        self._logger.info("MinIO started successfully")

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
        """Stop and remove MinIO container"""
        self._logger.info("Stopping MinIO container...")

        # Stop container
        subprocess.run(
            ["docker", "stop", self.CONTAINER_NAME],
            capture_output=True,
        )

        # Remove container
        subprocess.run(
            ["docker", "rm", self.CONTAINER_NAME],
            capture_output=True,
        )

    def create_bucket(self) -> None:
        """Create the S3 bucket using AWS CLI"""
        env = os.environ.copy()
        env.update(
            {
                "AWS_ACCESS_KEY_ID": self.config.access_key,
                "AWS_SECRET_ACCESS_KEY": self.config.secret_key,
            }
        )

        # Check if bucket exists
        result = subprocess.run(
            [
                "aws",
                "--endpoint-url",
                self.config.endpoint,
                "s3",
                "ls",
                f"s3://{self.config.bucket}",
            ],
            capture_output=True,
            text=True,
            env=env,
        )

        if result.returncode != 0:
            # Create bucket
            self._logger.info(f"Creating bucket: {self.config.bucket}")
            result = subprocess.run(
                [
                    "aws",
                    "--endpoint-url",
                    self.config.endpoint,
                    "s3",
                    "mb",
                    f"s3://{self.config.bucket}",
                ],
                capture_output=True,
                text=True,
                env=env,
            )
            if result.returncode != 0:
                raise RuntimeError(f"Failed to create bucket: {result.stderr}")

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
        """Upload LoRA to MinIO"""
        self._logger.info(
            f"Uploading LoRA to s3://{self.config.bucket}/{self.config.lora_name}"
        )

        env = os.environ.copy()
        env.update(
            {
                "AWS_ACCESS_KEY_ID": self.config.access_key,
                "AWS_SECRET_ACCESS_KEY": self.config.secret_key,
            }
        )

        result = subprocess.run(
            [
                "aws",
                "--endpoint-url",
                self.config.endpoint,
                "s3",
                "sync",
                local_path,
                f"s3://{self.config.bucket}/{self.config.lora_name}",
                "--exclude",
                "*.git*",
            ],
            capture_output=True,
            text=True,
            env=env,
        )

        if result.returncode != 0:
            raise RuntimeError(f"Failed to upload LoRA: {result.stderr}")

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
