/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import type { DeploymentConfig } from "../lib/config";

const VPC_CONFIG = {
  availabilityZones: ["us-west-2a", "us-west-2b"],
  cidrBlock: "10.42.0.0/16",
  name: "dynamo-mocker-cpu-test",
  region: "us-west-2",
} as const;

export const VALID_CONFIG: DeploymentConfig = {
  clusterConfig: {
    bastionConfig: {
      availabilityZone: "us-west-2a",
      iamRoleName: "dynamo-mocker-bastion-test",
      instanceType: "t3.small",
      keyPairName: "test-admin-key",
      name: "dynamo-mocker-bastion",
      region: "us-west-2",
      vpcConfig: VPC_CONFIG,
    },
    clusterName: "dynamo-mocker-cpu",
    region: "us-west-2",
    vpcConfig: VPC_CONFIG,
  },
  dynamoPlatformConfig: {
    dynamoNamespace: "dynamo-system",
    dynamoVersion: "1.4.0",
  },
  dgdPools: [
    {
      name: "cpu-a",
      availabilityZone: "us-west-2a",
      subnetSlot: 1,
      nodeGroupConfig: {
        instanceTypes: ["m7i.xlarge", "m6i.xlarge", "m5.xlarge"],
        scaling: {
          minSize: 1,
          desiredSize: 2,
          maxSize: 4,
        },
      },
    },
  ],
  dgds: [
    {
      dgdName: "mocker-a",
      poolNames: ["cpu-a"],
      mockerReplicas: 3,
      mockerSpeedupRatio: 1,
      modelName: "Qwen/Qwen3-0.6B",
    },
  ],
};

export const MULTI_POOL_CONFIG: DeploymentConfig = {
  ...VALID_CONFIG,
  dgdPools: [
    VALID_CONFIG.dgdPools[0],
    {
      name: "cpu-b",
      availabilityZone: "us-west-2a",
      subnetSlot: 2,
      nodeGroupConfig: {
        instanceTypes: ["c7i.xlarge"],
        scaling: {
          minSize: 1,
          desiredSize: 1,
          maxSize: 2,
        },
      },
    },
  ],
  dgds: [
    VALID_CONFIG.dgds[0],
    {
      dgdName: "mocker-b",
      poolNames: ["cpu-b"],
      mockerReplicas: 2,
      mockerSpeedupRatio: 2,
      modelName: "Qwen/Qwen3-1.7B",
    },
    {
      dgdName: "mocker-flex",
      poolNames: ["cpu-b", "cpu-a"],
      mockerReplicas: 4,
      mockerSpeedupRatio: 3,
      modelName: "Qwen/Qwen3-4B",
    },
  ],
};
