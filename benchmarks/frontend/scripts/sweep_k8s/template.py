# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Deploy YAML template rendering and application.

Supports ${VARIABLE} placeholders using Python's string.Template.
This enables arbitrary backend deployments (mocker, vLLM, TensorRT-LLM, etc.)
without hardcoding DGD structures.
"""

from __future__ import annotations

import string
from pathlib import Path
from typing import Dict

from sweep_core.models import DeployDimension, SweepConfig
from sweep_k8s.kubectl import apply_yaml

# Tokenizer backend mapping for template substitution
TOKENIZER_TEMPLATE_MAP = {
    "hf": "default",
    "default": "default",
    "fast": "fast",
    "fastokens": "fast",
}

DEFAULT_HF_TOKEN_SECRET_NAME = "hf-token-secret"


def _indent_block(text: str, spaces: int) -> str:
    prefix = " " * spaces
    return "\n".join(f"{prefix}{line}" if line else "" for line in text.splitlines())


def _build_image_pull_secrets_block(image_pull_secret: str) -> str:
    if not image_pull_secret:
        return ""
    return _indent_block(
        f"""imagePullSecrets:
  - name: {image_pull_secret}""",
        8,
    )


def build_substitution_dict(
    deploy: DeployDimension,
    config: SweepConfig,
) -> Dict[str, str]:
    """Build a variable substitution dictionary for template rendering.

    Combines deploy dimensions, sweep config, and k8s config into a flat
    dictionary suitable for string.Template substitution.
    """
    k8s = config.k8s
    hf_token_secret_name = DEFAULT_HF_TOKEN_SECRET_NAME
    variables: Dict[str, str] = {
        # Deploy dimensions
        "DYN_TOKENIZER_BACKEND": TOKENIZER_TEMPLATE_MAP.get(
            deploy.tokenizer, deploy.tokenizer
        ),
        "NUM_WORKERS": str(deploy.workers),
        # Model info
        "MODEL": config.model,
        "MODEL_NAME": config.model_name,
        "MODEL_PATH": config.model,
        # Image
        "IMAGE": k8s.image,
        # K8s config
        "NAMESPACE": k8s.namespace,
        "DGD_NAME": k8s.dgd_name,
        "FRONTEND_PORT": str(k8s.frontend_port),
        "WORKER_REPLICAS": str(k8s.worker_replicas),
        "FRONTEND_REPLICAS": str(k8s.frontend_replicas),
        "SPEEDUP_RATIO": str(config.speedup_ratio),
        "REQUEST_PLANE": k8s.request_plane,
        "EVENT_PLANE": k8s.event_plane,
        "ROUTER_MODE": k8s.router_mode,
        "HF_TOKEN_SECRET_NAME": hf_token_secret_name,
        "FRONTEND_IMAGE_PULL_SECRETS_BLOCK": _build_image_pull_secrets_block(
            k8s.image_pull_secret
        ),
        "WORKER_IMAGE_PULL_SECRETS_BLOCK": _build_image_pull_secrets_block(
            k8s.image_pull_secret
        ),
    }

    # Add any env_overrides from the deploy dimension
    variables.update(deploy.env_overrides)

    return variables


def render_template(template_path: Path, variables: Dict[str, str]) -> str:
    """Read a deploy YAML template and substitute ${VAR} placeholders.

    Uses safe_substitute so missing variables are left as-is rather than
    raising KeyError. This is important because DGD templates may contain
    ${VARIABLE} references that are resolved by the k8s operator at runtime.
    """
    raw = template_path.read_text()
    tmpl = string.Template(raw)
    return tmpl.safe_substitute(variables)


def apply_rendered_template(
    template_path: Path,
    deploy: DeployDimension,
    config: SweepConfig,
) -> None:
    """Render a deploy template and apply it via kubectl."""
    variables = build_substitution_dict(deploy, config)
    rendered = render_template(template_path, variables)
    print(f"  Applying rendered template: {template_path.name}")
    apply_yaml(rendered, config.k8s.namespace)
