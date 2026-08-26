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

module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "6.6.1"

  name = var.cluster_name
  cidr = var.vpc_cidr

  azs = local.cluster_availability_zones

  # The first subnet in every list is in the selected workload AZ. The single
  # NAT gateway is therefore colocated with the CPU node group.
  private_subnets = [
    cidrsubnet(var.vpc_cidr, 4, 0),
    cidrsubnet(var.vpc_cidr, 4, 1),
  ]
  public_subnets = [
    cidrsubnet(var.vpc_cidr, 4, 2),
    cidrsubnet(var.vpc_cidr, 4, 3),
  ]
  intra_subnets = [
    cidrsubnet(var.vpc_cidr, 4, 4),
    cidrsubnet(var.vpc_cidr, 4, 5),
  ]

  enable_dns_hostnames = true
  enable_dns_support   = true

  enable_nat_gateway = true
  single_nat_gateway = true

  public_subnet_tags = {
    "kubernetes.io/role/elb" = "1"
  }

  private_subnet_tags = {
    "kubernetes.io/cluster/${var.cluster_name}" = "shared"
    "kubernetes.io/role/internal-elb"           = "1"
  }
}
