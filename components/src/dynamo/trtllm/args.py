# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Argument parsing and typed config for Dynamo TRT-LLM."""

import argparse
import logging
import os
import sys
from typing import Any, Dict, Optional, Sequence

from dynamo.common.config_dump import register_encoder
from dynamo.common.configuration.groups.runtime_args import (
    DynamoRuntimeArgGroup,
    DynamoRuntimeConfig,
)
from dynamo.common.utils.runtime import parse_endpoint
from dynamo.trtllm.backend_args import DynamoTrtllmArgGroup, DynamoTrtllmConfig
from dynamo.trtllm.constants import DisaggregationMode, Modality

DEFAULT_ENDPOINT_COMPONENT = "tensorrt_llm"
DEFAULT_PREFILL_COMPONENT = "prefill"
DEFAULT_ENCODE_COMPONENT = "tensorrt_llm_encode"
DEFAULT_DIFFUSION_COMPONENT = "diffusion"
DEFAULT_ENDPOINT_NAME = "generate"
VALID_TRTLLM_CONNECTORS = {"none", "kvbm"}


class Config(DynamoRuntimeConfig, DynamoTrtllmConfig):
    component: str
    use_kv_events: bool

    def validate(self) -> None:
        DynamoRuntimeConfig.validate(self)
        DynamoTrtllmConfig.validate(self)
        # Derive use_kv_events from publish_events_and_metrics
        self.use_kv_events = self.publish_events_and_metrics

        # fix the connector as trtllm accepts only one connector and it should be in VALID_TRTLLM_CONNECTORS
        # while the runtime args accepts a list of connectors
        if self.connector:
            if len(self.connector) > 1:
                raise ValueError(
                    "TRT-LLM supports at most one connector entry. Use `--connector none` or `--connector kvbm`."
                )
            elif self.connector[0] not in VALID_TRTLLM_CONNECTORS:
                source = (
                    f"DYN_CONNECTOR environment variable ('{os.environ['DYN_CONNECTOR']}')"
                    if "DYN_CONNECTOR" in os.environ
                    else f"shared runtime default ('{self.connector[0]}')"
                )
                logging.warning(
                    f"TRT-LLM does not support connector '{self.connector[0]}' (set via {source}). "
                    f"Supported connectors: {VALID_TRTLLM_CONNECTORS}. Falling back to 'none'."
                )
                self.connector = ["none"]

    def has_connector(self, connector_name: str) -> bool:
        return (
            self.connector is not None
            and len(self.connector) > 0
            and connector_name == self.connector[0]
        )


@register_encoder(Config)
def _preprocess_for_encode_config(config: Config) -> Dict[str, Any]:
    return config.__dict__


def parse_args(argv: Optional[Sequence[str]] = None) -> Config:
    """Parse command-line arguments for the TensorRT-LLM backend."""
    cli_args = list(argv) if argv is not None else sys.argv[1:]

    parser = argparse.ArgumentParser(
        description="Dynamo TensorRT-LLM worker configuration",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    DynamoRuntimeArgGroup().add_arguments(parser)
    DynamoTrtllmArgGroup().add_arguments(parser)

    parsed_args = parser.parse_args(cli_args)
    config = Config.from_cli_args(parsed_args)
    config.validate()

    # TODO: move this to common configuration.
    if config.custom_jinja_template:
        expanded_template_path = os.path.expanduser(
            os.path.expandvars(config.custom_jinja_template)
        )
        if not os.path.isfile(expanded_template_path):
            raise FileNotFoundError(
                f"Custom Jinja template file not found: {expanded_template_path}"
            )
        config.custom_jinja_template = expanded_template_path
    else:
        config.custom_jinja_template = None

    endpoint = config.endpoint or _default_endpoint(
        namespace=config.namespace,
        modality=config.modality,
        disaggregation_mode=config.disaggregation_mode,
    )
    parsed_namespace, parsed_component_name, parsed_endpoint_name = parse_endpoint(
        endpoint
    )
    config.namespace = parsed_namespace
    config.component = parsed_component_name
    config.endpoint = parsed_endpoint_name

    return config


def _default_endpoint(
    namespace: str, modality: Modality, disaggregation_mode: DisaggregationMode
) -> str:
    if modality == Modality.VIDEO_DIFFUSION:
        component_name = DEFAULT_DIFFUSION_COMPONENT
    elif disaggregation_mode == DisaggregationMode.ENCODE:
        component_name = DEFAULT_ENCODE_COMPONENT
    elif disaggregation_mode == DisaggregationMode.PREFILL:
        component_name = DEFAULT_PREFILL_COMPONENT
    else:
        component_name = DEFAULT_ENDPOINT_COMPONENT
    return f"dyn://{namespace}.{component_name}.{DEFAULT_ENDPOINT_NAME}"
