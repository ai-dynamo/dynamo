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

resource "helm_release" "dynamo_platform" {
  name             = "dynamo-platform"
  namespace        = var.dynamo_namespace
  create_namespace = true
  chart            = local.platform_chart_url

  atomic          = true
  cleanup_on_fail = true
  timeout         = 900
  wait            = true
  wait_for_jobs   = true

  # Kubernetes-native discovery, TCP requests, and ZMQ events do not require
  # the bundled etcd or NATS subcharts for this deployment.
  values = [yamlencode({
    global = {
      etcd = {
        install = false
      }
      nats = {
        install = false
      }
    }
  })]

  depends_on = [module.eks]
}

resource "helm_release" "dynamo_mocker" {
  name      = var.dgd_name
  namespace = var.dynamo_namespace
  chart     = "${path.module}/helm/dynamo-mocker"

  atomic          = true
  cleanup_on_fail = true
  lint            = true
  timeout         = 600
  wait            = true

  values = [yamlencode({
    deployment = {
      name = var.dgd_name
    }
    image = local.planner_image
    model = {
      name = var.model_name
    }
    mocker = {
      replicas      = var.mocker_replicas
      speedup_ratio = var.mocker_speedup_ratio
    }
    placement = {
      nodeSelector = local.mocker_node_selector
    }
  })]

  depends_on = [helm_release.dynamo_platform]
}
