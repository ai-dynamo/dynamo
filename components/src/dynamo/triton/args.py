# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse

from dynamo.common.configuration.arg_group import ArgGroup
from dynamo.common.configuration.config_base import ConfigBase
from dynamo.common.configuration.utils import add_argument

_DISCOVER_BACKEND_CHOICES = ["kubernetes", "etcd", "file", "mem"]
_REQUEST_PLANE_CHOICES = ["tcp", "nats"]


class DynamoArgGroup(ArgGroup):
    """Dynamo Runtime configuration options."""

    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser) -> None:
        if not isinstance(parser, argparse.ArgumentParser):
            raise TypeError("parser must be an instance of argparse.ArgumentParser")

        # -- Dynamo Runtime Options -------------------------------------------------
        dynamo_group = parser.add_argument_group("Dynamo Runtime Options")
        add_argument(
            dynamo_group,
            flag_name="--namespace",
            env_var="DYN_NAMESPACE",
            default="dynamo",
            help="Dynamo namespace.",
        )
        add_argument(
            dynamo_group,
            flag_name="--discovery-backend",
            env_var="DYN_DISCOVERY_BACKEND",
            default="etcd",
            choices=_DISCOVER_BACKEND_CHOICES,
            help="Service discovery backend: kubernetes (K8s API), etcd (distributed KV), file (local filesystem), mem (in-memory).",
        )
        add_argument(
            dynamo_group,
            flag_name="--request-plane",
            env_var="DYN_REQUEST_PLANE",
            default="tcp",
            choices=_REQUEST_PLANE_CHOICES,
            help="How requests are distributed from routers to workers. "
            "'tcp' is fastest.",
        )


class Config(ConfigBase):
    """Configuration for Dynamo Runtime generic options."""

    discovery_backend: str
    request_plane: str
    namespace: str

    def validate(self) -> None:
        if hasattr(super(), "validate"):
            super().validate()

        if (
            not self.discovery_backend
            or self.discovery_backend not in _DISCOVER_BACKEND_CHOICES
        ):
            raise ValueError(
                "--discovery-backend is required and must be one of "
                f"'{''', '''.join(_DISCOVER_BACKEND_CHOICES)}' "
                "(or set DYN_DISCOVERY_BACKEND)."
            )
        if not self.request_plane or self.request_plane not in _REQUEST_PLANE_CHOICES:
            raise ValueError(
                "--request-plane is required and must be one of "
                f"'{''', '''.join(_REQUEST_PLANE_CHOICES)}' "
                "(or set DYN_REQUEST_PLANE)."
            )
        if not self.namespace:
            raise ValueError("--namespace is required (or set DYN_NAMESPACE).")
