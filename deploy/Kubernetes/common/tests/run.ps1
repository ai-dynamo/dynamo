# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

set-strictmode -version latest

. "$(& git rev-parse --show-toplevel)/deploy/Kubernetes/_build/helm-test.ps1"

$tests = @(
  @{
    name = 'basic'
    expected = 0
    matches = @(
      @{
        indent = 6
        lines = @(
          'labels:'
          '  app: test'
          '  app.kubernetes.io/component: test_dynamo_chart'
          '  app.kubernetes.io/instance: test'
          '  app.kubernetes.io/name: test_dynamo_chart'
          '  app.kubernetes.io/part-of: dynamo'
          '  app.kubernetes.io/version: "1.0.0"'
          '  app.kubernetes.io/managed-by: Helm'
          '  helm.sh/chart: "dynamo_component"'
          '  helm.sh/version: "1.0.0"'
        )
      }
      @{
        indent = 8
        lines = @(
          '- containerPort: 8000'
          '  name: health'
        )
      }
      @{
        indent = 8
        lines = @(
          '- containerPort: 9345'
          '  name: request'
        )
      }
      @{
        indent = 8
        lines = @(
          '- containerPort: 443'
          '  name: api'
        )
      }
      @{
        indent = 8
        lines = @(
          '- containerPort: 9347'
          '  name: metrics'
        )
      }
      @{
        indent = 10
        lines = @(
          'limits:'
          '  cpu: 4'
          '  ephemeral-storage: 1Gi'
          '  nvidia.com/gpu: 1'
          '  memory: 16Gi'
        )
      }
      @{
        indent = 10
        lines = @(
          'requests:'
          '  cpu: 4'
          '  ephemeral-storage: 1Gi'
          '  nvidia.com/gpu: 1'
          '  memory: 16Gi'
        )
      }
    )
    options = @()
    values = @(
      'basic.yaml'
    )
  }
  @{
    name = "resource_gpu"
    expected = 0
    matches = @(
      @{
        indent = 14
        lines = @(
          '- key: nvidia.com/gpu'
          '  operator: Exists'
        )
      }
      @{
        indent = 14
        lines = @(
          '- key: nvidia.com/gpu.product'
          '  operator: In'
          '  values:'
          '  - a10g'
        )
      }
      @{
        indent = 10
        lines = @(
          'limits:'
          '  cpu: 4'
          '  ephemeral-storage: 1Gi'
          '  nvidia.com/gpu: 2'
          '  memory: 16Gi'
        )
      }
      @{
        indent = 10
        lines = @(
          'requests:'
          '  cpu: 4'
          '  ephemeral-storage: 1Gi'
          '  nvidia.com/gpu: 2'
          '  memory: 16Gi'
        )
      }
    )
    options = @()
    values = @(
      'basic.yaml'
      'resource_gpu.yaml'
    )
  }
  @{
    name = 'invalid_values'
    expected = 1
    matches = @(
      'Error: values don''t meet the specifications of the schema\(s\) in the following chart\(s\):'
      @{
        indent = 0
        lines = @(
          '- kubernetes.checks.liveness.successThreshold: Must validate one and only one schema (oneOf)'
          '- kubernetes.checks.liveness.successThreshold: Must be greater than or equal to 1'
        )
      }
      @{
        indent = 0
        lines = @(
          '- kubernetes.checks.liveness.failureThreshold: Must validate one and only one schema (oneOf)'
          '- kubernetes.checks.liveness.failureThreshold: Must be greater than or equal to 1'
        )
      }
      @{
        indent = 0
        lines = @(
          '- kubernetes.checks.liveness.initialDelaySeconds: Must validate one and only one schema (oneOf)'
          '- kubernetes.checks.liveness.initialDelaySeconds: Invalid type. Expected: integer, given: number'
        )
      }
      @{
        indent = 0
        lines = @(
          '- kubernetes.checks.liveness.periodSeconds: Must validate one and only one schema (oneOf)'
          '- kubernetes.checks.liveness.periodSeconds: Invalid type. Expected: integer, given: string'
        )
      }
      @{
        indent = 0
        lines = @(
          '- ports.health: Must validate one and only one schema (oneOf)'
          '- ports.health: Must be less than or equal to 65535'
        )
      }
      @{
        indent = 0
        lines = @(
        '- ports.metrics: Must validate one and only one schema (oneOf)'
        '- ports.metrics: Invalid type. Expected: integer, given: string'
        )
      }
      @{
        indent = 0
        lines = @(
        '- ports.request: Must validate one and only one schema (oneOf)'
        '- ports.request: Must be greater than or equal to 1025'
        )
      }
      @{
        indent = 0
        lines = @(
          '- resources.cpu: Must validate one and only one schema (oneOf)'
          '- resources.cpu: Must be greater than or equal to 1'
        )
      }
    )
    options = @()
    values = @(
      'basic.yaml'
      'invalid_values.yaml'
    )
  }
)

$config = initialize_test $args $tests

# Being w/ the state of not having passed.
$is_pass = $false

try {
  $is_pass = $(test_helm_chart $config)
}
catch {
  if (get_is_debug) {
    throw $_
  }

  fatal_exit "$_"
}

# Clean up any NVBUILD environment variables left behind by the build.
cleanup_after

if (-not $is_pass) {
  exit -1
}

exit 0
