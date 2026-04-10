# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FastVideo-specific health check configuration."""

from dynamo.health_check import HealthCheckPayload


class FastVideoHealthCheckPayload(HealthCheckPayload):
    """Minimal `/v1/videos` payload used for FastVideo readiness checks."""

    def __init__(self, model_path: str) -> None:
        self.default_payload = {
            "prompt": "test",
            "model": model_path,
            "seconds": 1,
            "size": "256x256",
            "response_format": "b64_json",
            "nvext": {
                "fps": 8,
                "num_frames": 8,
                "num_inference_steps": 1,
                "guidance_scale": 5.0,
            },
        }
        super().__init__()
