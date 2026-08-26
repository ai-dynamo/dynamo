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

variable "aws_region" {
  description = "AWS Region in which to create the VPC and EKS cluster."
  type        = string
  default     = "us-west-2"
}

variable "cluster_name" {
  description = "Name of the EKS cluster."
  type        = string
  default     = "dynamo-mocker-cpu"

  validation {
    condition     = can(regex("^[0-9A-Za-z][0-9A-Za-z_-]{0,99}$", var.cluster_name))
    error_message = "cluster_name must be 1-100 characters and contain only letters, numbers, underscores, and hyphens."
  }
}

variable "kubernetes_version" {
  description = "EKS Kubernetes minor version."
  type        = string
  default     = "1.35"
}

variable "vpc_cidr" {
  description = "CIDR block for the VPC. The template derives six non-overlapping subnets from it."
  type        = string
  default     = "10.42.0.0/16"

  validation {
    condition = (
      can(cidrnetmask(var.vpc_cidr)) &&
      can(cidrsubnet(var.vpc_cidr, 4, 5)) &&
      can(
        tonumber(split("/", var.vpc_cidr)[1]) >= 16 &&
        tonumber(split("/", var.vpc_cidr)[1]) <= 20
      )
    )
    error_message = "vpc_cidr must be an IPv4 VPC CIDR with a /16 through /20 prefix so the derived subnets are no smaller than /24."
  }
}

variable "workload_availability_zone" {
  description = "Availability Zone for every CPU node and Dynamo Mocker pod. Null selects the first EKS-eligible AZ in the Region."
  type        = string
  default     = null
  nullable    = true
}

variable "cluster_endpoint_public_access_cidrs" {
  description = "CIDR blocks allowed to reach the public EKS API endpoint. Use your administrator public IP with a /32 mask."
  type        = list(string)

  validation {
    condition = (
      length(var.cluster_endpoint_public_access_cidrs) > 0 &&
      alltrue([for cidr in var.cluster_endpoint_public_access_cidrs : can(cidrnetmask(cidr))]) &&
      !contains(var.cluster_endpoint_public_access_cidrs, "0.0.0.0/0")
    )
    error_message = "Provide at least one valid CIDR block for EKS API access; 0.0.0.0/0 is not allowed."
  }
}

variable "node_instance_types" {
  description = "Ordered EC2 instance type choices for the CPU managed node group."
  type        = list(string)
  default     = ["m7i.xlarge", "m6i.xlarge", "m5.xlarge"]

  validation {
    condition     = length(var.node_instance_types) > 0
    error_message = "node_instance_types must contain at least one EC2 instance type."
  }
}

variable "node_group_scaling" {
  description = "Minimum, desired, and maximum size of the single-AZ CPU managed node group."
  type = object({
    min_size     = number
    desired_size = number
    max_size     = number
  })
  default = {
    min_size     = 1
    desired_size = 2
    max_size     = 4
  }

  validation {
    condition = (
      var.node_group_scaling.min_size >= 0 &&
      var.node_group_scaling.desired_size >= 1 &&
      var.node_group_scaling.desired_size >= var.node_group_scaling.min_size &&
      var.node_group_scaling.max_size >= var.node_group_scaling.desired_size
    )
    error_message = "node_group_scaling must satisfy 0 <= min_size <= desired_size <= max_size and desired_size >= 1."
  }
}

variable "dynamo_version" {
  description = "Released Dynamo platform chart and planner image tag."
  type        = string
  default     = "1.4.0"
}

variable "dynamo_namespace" {
  description = "Kubernetes namespace for the Dynamo platform and Mocker deployment."
  type        = string
  default     = "dynamo-system"
}

variable "dgd_name" {
  description = "Name of the DynamoGraphDeployment."
  type        = string
  default     = "mocker-agg"

  validation {
    condition     = can(regex("^[a-z0-9]([-a-z0-9]{0,51}[a-z0-9])?$", var.dgd_name))
    error_message = "dgd_name must be a valid Kubernetes DNS label with at most 53 characters so it is also a valid Helm release name."
  }
}

variable "mocker_replicas" {
  description = "Number of Mocker pods. Each pod starts one simulated worker."
  type        = number
  default     = 3

  validation {
    condition     = var.mocker_replicas >= 1 && floor(var.mocker_replicas) == var.mocker_replicas
    error_message = "mocker_replicas must be a positive integer."
  }
}

variable "model_name" {
  description = "Public model identifier used for tokenizer metadata and the OpenAI-compatible model name."
  type        = string
  default     = "Qwen/Qwen3-0.6B"
}

variable "mocker_speedup_ratio" {
  description = "Multiplier applied by Mocker to the simulated serving speed."
  type        = number
  default     = 1.0

  validation {
    condition     = var.mocker_speedup_ratio > 0
    error_message = "mocker_speedup_ratio must be greater than zero."
  }
}

variable "tags" {
  description = "Tags applied to AWS resources created by the template."
  type        = map(string)
  default = {
    ManagedBy = "Terraform"
    Project   = "dynamo-mocker-cpu"
  }
}
