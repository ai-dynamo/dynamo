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

module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "21.25.0"

  name               = var.cluster_name
  kubernetes_version = var.kubernetes_version

  vpc_id                   = module.vpc.vpc_id
  control_plane_subnet_ids = module.vpc.intra_subnets
  subnet_ids               = module.vpc.private_subnets

  endpoint_private_access                  = true
  endpoint_public_access                   = true
  endpoint_public_access_cidrs             = var.cluster_endpoint_public_access_cidrs
  authentication_mode                      = "API_AND_CONFIG_MAP"
  enable_irsa                              = false
  create_kms_key                           = false
  encryption_config                        = null
  enable_cluster_creator_admin_permissions = true

  addons = {
    coredns    = {}
    kube-proxy = {}
    vpc-cni = {
      before_compute = true
    }
  }

  eks_managed_node_groups = {
    mocker_cpu = {
      name = "mocker-cpu"

      # Supplying exactly one subnet constrains every managed node to the
      # selected workload AZ. The DGD also has an explicit zone selector.
      subnet_ids = [module.vpc.private_subnets[0]]

      ami_type       = "AL2023_x86_64_STANDARD"
      capacity_type  = "ON_DEMAND"
      instance_types = var.node_instance_types

      min_size     = var.node_group_scaling.min_size
      desired_size = var.node_group_scaling.desired_size
      max_size     = var.node_group_scaling.max_size

      labels = local.node_labels

      block_device_mappings = {
        xvda = {
          device_name = "/dev/xvda"
          ebs = {
            delete_on_termination = true
            encrypted             = true
            volume_size           = 50
            volume_type           = "gp3"
          }
        }
      }

      node_repair_config = {
        enabled = true
      }

      update_config = {
        max_unavailable = 1
      }
    }
  }
}
