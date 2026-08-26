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

data "aws_availability_zones" "available" {
  state = "available"

  # Enabled Local and Wavelength Zones can otherwise appear in this data
  # source, but EKS cluster subnets must use standard Availability Zones.
  filter {
    name   = "opt-in-status"
    values = ["opt-in-not-required"]
  }

  lifecycle {
    postcondition {
      condition = length([
        for index, name in self.names : name
        if !contains(local.eks_unsupported_zone_ids, self.zone_ids[index])
      ]) >= 2
      error_message = "Amazon EKS requires at least two eligible Availability Zones in the selected Region."
    }

    postcondition {
      condition = (
        var.workload_availability_zone == null ||
        contains([
          for index, name in self.names : name
          if !contains(local.eks_unsupported_zone_ids, self.zone_ids[index])
        ], var.workload_availability_zone)
      )
      error_message = "workload_availability_zone must be an available, EKS-eligible AZ in aws_region."
    }
  }
}

locals {
  # Amazon EKS does not allow cluster subnets in these AZ IDs.
  eks_unsupported_zone_ids = toset(["use1-az3", "usw1-az2", "cac1-az3"])

  eks_eligible_availability_zones = [
    for index, name in data.aws_availability_zones.available.names : name
    if !contains(local.eks_unsupported_zone_ids, data.aws_availability_zones.available.zone_ids[index])
  ]

  workload_availability_zone = coalesce(
    var.workload_availability_zone,
    try(local.eks_eligible_availability_zones[0], "unavailable"),
  )

  secondary_availability_zone = try([for name in local.eks_eligible_availability_zones : name
    if name != local.workload_availability_zone
  ][0], local.workload_availability_zone)

  cluster_availability_zones = [
    local.workload_availability_zone,
    local.secondary_availability_zone,
  ]

  node_labels = {
    "dynamo.nvidia.com/purpose" = "mocker-cpu"
  }

  mocker_node_selector = merge(local.node_labels, {
    "topology.kubernetes.io/zone" = local.workload_availability_zone
  })

  planner_image = "nvcr.io/nvidia/ai-dynamo/dynamo-planner:${var.dynamo_version}"
  platform_chart_url = (
    "https://helm.ngc.nvidia.com/nvidia/ai-dynamo/charts/dynamo-platform-${var.dynamo_version}.tgz"
  )
}
