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

from datetime import datetime
from typing import Any, Dict, Optional

from kubernetes import config as k8s_config


class DynamoDeployment:
    def __init__(
        self,
        name: str,
        cluster: str,
        admin_console: str,
        created_at: str,
        created_by: str,
        _schema: str = "v1",
        ingress_base_url: Optional[str] = None,
    ):
        self.name = name
        self.cluster = cluster
        self.admin_console = admin_console
        self.created_at = created_at
        self.created_by = created_by
        self._schema = _schema
        self.ingress_url = (
            f"{ingress_base_url}/api/v2/deployments/{name}?cluster={cluster}"
            if ingress_base_url
            else None
        )

    @classmethod
    def create_deployment(
        cls, deployment_name: str, namespace: str, config: Any
    ) -> "DynamoDeployment":
        # Load kube config and get username
        k8s_config.load_kube_config()
        username = (
            k8s_config.list_kube_config_contexts()[1]
            .get("context", {})
            .get("user", "unknown")
        )

        return cls(
            name=deployment_name,
            cluster=namespace,
            admin_console=f"kubectl get dynamodeployment {deployment_name} -n {namespace}",
            created_at=datetime.now().isoformat(),
            created_by=username,
            _schema="v1alpha1",
            ingress_base_url=config.get("ingress_base_url"),
        )

    def get_crd_payload(
        self,
        bento: str,
        scaling_min: int,
        scaling_max: int,
        instance_type: str,
        env_vars: list,
        secret: list,
    ) -> Dict[str, Any]:
        # Ensure bento is in name:tag format
        if ":" not in bento:
            bento = f"{bento}:latest"

        payload = {
            "apiVersion": "nvidia.com/v1alpha1",
            "kind": "DynamoDeployment",
            "metadata": {
                "name": self.name,
                "namespace": self.cluster,
                "labels": {
                    "app.kubernetes.io/name": "dynamo-kubernetes-operator",
                    "app.kubernetes.io/managed-by": "dynamo-cli",
                },
            },
            "spec": {
                "dynamoNim": bento,
                "services": {
                    "main": {
                        "spec": {
                            "dynamoNim": bento,
                            "serviceName": self.name,
                            "autoscaling": {
                                "minReplicas": scaling_min or 1,
                                "maxReplicas": scaling_max or 5,
                            },
                            "resources": {
                                "requests": {
                                    "cpu": "4",
                                    "memory": "16Gi",
                                    "gpu": instance_type or "1",
                                },
                                "limits": {
                                    "cpu": "8",
                                    "memory": "32Gi",
                                    "gpu": instance_type or "1",
                                },
                            },
                            "envs": env_vars,
                            "ingress": {
                                "enabled": True,
                                "hostPrefix": self.name,
                            },
                            "readinessProbe": {
                                "httpGet": {"path": "/healthz", "port": 8080},
                                "initialDelaySeconds": 5,
                                "periodSeconds": 10,
                            },
                        }
                    }
                },
            },
        }

        return payload
