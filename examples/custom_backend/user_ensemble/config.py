# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Application configuration for the integrated-encoder user ensemble."""

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_ENCODER_CLASS = (
    "examples.custom_encoder.hitchhikers_vision_encoder.HitchhikersVisionEncoder"
)
