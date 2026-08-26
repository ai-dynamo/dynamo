# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

output "cluster_name" {
  description = "EKS cluster name."
  value       = module.eks.cluster_name
}

output "aws_region" {
  description = "AWS Region containing the EKS cluster."
  value       = var.aws_region
}

output "dynamo_namespace" {
  description = "Kubernetes namespace containing the Dynamo platform and DGD."
  value       = var.dynamo_namespace
}

output "dgd_name" {
  description = "DynamoGraphDeployment name."
  value       = var.dgd_name
}

output "model_name" {
  description = "Model identifier exposed by Mocker."
  value       = var.model_name
}

output "workload_availability_zone" {
  description = "The only Availability Zone used by the CPU node group and Mocker pods."
  value       = local.workload_availability_zone
}

output "configure_kubectl" {
  description = "Command that adds the EKS cluster to the local kubeconfig."
  value       = "aws eks update-kubeconfig --region ${var.aws_region} --name ${module.eks.cluster_name}"
}

output "verify_single_az_placement" {
  description = "Command that shows the node and zone for every pod in the DGD."
  value       = "kubectl get pods -n ${var.dynamo_namespace} -l nvidia.com/dynamo-graph-deployment-name=${var.dgd_name} -o 'custom-columns=NAME:.metadata.name,NODE:.spec.nodeName,ZONE:.spec.nodeSelector.topology\\.kubernetes\\.io/zone'"
}

output "port_forward_frontend" {
  description = "Command that forwards the Dynamo frontend to localhost:8000."
  value       = "kubectl port-forward -n ${var.dynamo_namespace} svc/${var.dgd_name}-frontend 8000:8000"
}
