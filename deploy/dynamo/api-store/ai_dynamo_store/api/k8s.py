#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#  http://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from typing import Any, Dict

from kubernetes import client, config


def create_custom_resource(
    group: str, version: str, namespace: str, plural: str, body: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Create a custom resource in Kubernetes.

    Args:
        group: API group
        version: API version
        namespace: Target namespace
        plural: Resource plural name
        body: Resource definition

    Returns:
        Created resource
    """
    try:
        config.load_incluster_config()
    except config.config_exception.ConfigException:
        config.load_kube_config()

    api = client.CustomObjectsApi()
    return api.create_namespaced_custom_object(
        group=group, version=version, namespace=namespace, plural=plural, body=body
    )


def create_dynamo_deployment(
    name: str, namespace: str, dynamo_nim: str, labels: Dict[str, str]
) -> Dict[str, Any]:
    """
    Create a DynamoDeployment custom resource.

    Args:
        name: Deployment name
        namespace: Target namespace
        dynamo_nim: Bento name and version (format: name:version)
        labels: Resource labels

    Returns:
        Created deployment
    """
    body = {
        "apiVersion": "nvidia.com/v1alpha1",
        "kind": "DynamoDeployment",
        "metadata": {"name": name, "namespace": namespace, "labels": labels},
        "spec": {"dynamoNim": dynamo_nim, "services": {}},
    }

    return create_custom_resource(
        group="nvidia.com",
        version="v1alpha1",
        namespace=namespace,
        plural="dynamodeployments",
        body=body,
    )
