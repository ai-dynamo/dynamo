# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared Kubernetes client initialization helpers."""

import logging
import os

from kubernetes_asyncio import client, config

logger = logging.getLogger(__name__)


async def init_kubernetes_clients(
    need_custom_api: bool = False,
    need_apps_api: bool = False,
):
    """Initialize Kubernetes API clients.

    Uses KUBECONFIG env var -> in-cluster config -> default kubeconfig fallback.

    Args:
        need_custom_api: If True, also return CustomObjectsApi (for CRD operations)
        need_apps_api: If True, also return AppsV1Api (for StatefulSet operations)

    Returns:
        Tuple of (core_api, batch_api, custom_api_or_None, apps_api_or_None, in_cluster)
    """
    in_cluster = False
    kubeconfig_path = os.environ.get("KUBECONFIG")

    if kubeconfig_path and os.path.exists(kubeconfig_path):
        logger.info(f"Loading kubeconfig from KUBECONFIG: {kubeconfig_path}")
        await config.load_kube_config(config_file=kubeconfig_path)
        logger.info("Successfully loaded kubeconfig from KUBECONFIG")
    else:
        try:
            logger.info("Attempting in-cluster kubernetes config")
            config.load_incluster_config()
            in_cluster = True
            logger.info("Successfully loaded in-cluster kubernetes config")
        except Exception as e:
            logger.warning(
                f"In-cluster config failed ({type(e).__name__}: {e}), "
                f"falling back to default kubeconfig (~/.kube/config)"
            )
            await config.load_kube_config()
            logger.info("Successfully loaded default kubeconfig")

    k8s_client = client.ApiClient()
    core_api = client.CoreV1Api(k8s_client)
    batch_api = client.BatchV1Api(k8s_client)
    custom_api = client.CustomObjectsApi(k8s_client) if need_custom_api else None
    apps_api = client.AppsV1Api() if need_apps_api else None

    return core_api, batch_api, custom_api, apps_api, in_cluster
