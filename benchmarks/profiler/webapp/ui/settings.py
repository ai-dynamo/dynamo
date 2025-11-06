# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
UI components for settings panels in the Dynamo SLA Profiler webapp.

This module provides functions to build the settings UI sections:
- Model and backend configuration
- Hardware configuration (GPUs, cost)
- SLA parameters (ISL, OSL, TTFT, ITL)
"""

import gradio as gr
from aiconfigurator.sdk.common import SupportedModels

from benchmarks.profiler.webapp.core.constants import (
    BACKEND_VERSIONS,
    DEFAULT_CONFIG_YAML,
    GPU_SYSTEMS,
    INFERENCE_BACKENDS,
    MAX_GPU_OPTIONS,
    MIN_GPU_OPTIONS,
)


def create_model_settings():
    """
    Create the model and backend settings UI.

    Returns:
        Dictionary of Gradio components
    """
    with gr.Group():
        with gr.Row():
            supported_models = list(SupportedModels.keys())
            aic_model_name = gr.Dropdown(
                label="Model",
                choices=supported_models,
                value=supported_models[0],
                info="Model to profile",
            )

            backend = gr.Dropdown(
                label="Backend",
                choices=INFERENCE_BACKENDS,
                value="trtllm",
                info="Inference backend",
            )

        config_yaml = gr.Textbox(
            label="Config (YAML)",
            placeholder=DEFAULT_CONFIG_YAML,
            lines=5,
            info="DynamoGraphDeployment YAML configuration",
        )

        use_aic = gr.Checkbox(
            label="Use AI Configurator",
            value=True,
            info="Use AI Configurator to estimate performance",
        )

        with gr.Row():
            aic_backend = gr.Dropdown(
                label="AI Configurator Backend",
                choices=INFERENCE_BACKENDS,
                value="trtllm",
                info="Backend for AI Configurator estimation",
                visible=True,
            )

            aic_backend_version = gr.Dropdown(
                label="AI Configurator Backend Version",
                choices=BACKEND_VERSIONS["trtllm"],
                value="0.20.0",
                info="Backend version for AI Configurator",
                allow_custom_value=True,
                visible=True,
            )

    return {
        "aic_model_name": aic_model_name,
        "backend": backend,
        "config_yaml": config_yaml,
        "use_aic": use_aic,
        "aic_backend": aic_backend,
        "aic_backend_version": aic_backend_version,
    }


def create_hardware_settings():
    """
    Create the hardware configuration UI.

    Returns:
        Dictionary of Gradio components
    """
    with gr.Group():
        with gr.Row():
            aic_system = gr.Dropdown(
                label="System",
                choices=GPU_SYSTEMS,
                value="H200_SXM",
                info="Target GPU system",
            )

            gpu_cost_per_hour = gr.Number(
                label="Cost per GPU Hour ($)",
                value=3.0,
                info="Cost per GPU per hour in dollars",
            )

        with gr.Row():
            min_num_gpus_per_engine = gr.Dropdown(
                label="Min GPUs per Engine",
                choices=MIN_GPU_OPTIONS,
                value=1,
                info="Minimum number of GPUs (TP size)",
            )

            max_num_gpus_per_engine = gr.Dropdown(
                label="Max GPUs per Engine",
                choices=MAX_GPU_OPTIONS,
                value=4,
                info="Maximum number of GPUs (TP size)",
            )

        num_gpus_per_node = gr.Number(
            label="GPUs per Node",
            value=8,
            info="Number of GPUs per node (for MoE models)",
        )

    return {
        "aic_system": aic_system,
        "gpu_cost_per_hour": gpu_cost_per_hour,
        "min_num_gpus_per_engine": min_num_gpus_per_engine,
        "max_num_gpus_per_engine": max_num_gpus_per_engine,
        "num_gpus_per_node": num_gpus_per_node,
    }


def create_sla_settings():
    """
    Create the SLA configuration UI.

    Returns:
        Dictionary of Gradio components
    """
    with gr.Group():
        with gr.Row():
            isl = gr.Number(
                label="Input Sequence Length (ISL)",
                value=5000,
                precision=0,
                info="Target input sequence length",
            )

            osl = gr.Number(
                label="Output Sequence Length (OSL)",
                value=50,
                precision=0,
                info="Target output sequence length",
            )

        with gr.Row():
            max_context_length = gr.Number(
                label="Max Context Length",
                value=8192,
                precision=0,
                info="Maximum context length supported by the model",
            )

            ttft = gr.Number(
                label="Target TTFT (ms)",
                value=50.0,
                info="Target Time To First Token in milliseconds",
            )

        itl = gr.Number(
            label="Target ITL (ms)",
            value=10.0,
            info="Target Inter Token Latency in milliseconds",
        )

    return {
        "isl": isl,
        "osl": osl,
        "max_context_length": max_context_length,
        "ttft": ttft,
        "itl": itl,
    }
