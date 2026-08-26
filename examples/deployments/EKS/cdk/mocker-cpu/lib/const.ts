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

import type {
  BastionConfig,
  ClusterConfig,
  DeploymentConfig,
  DgdConfig,
  DgdPoolConfig,
  DynamoPlatformConfig,
  NodeGroupConfig,
  VpcConfig,
} from "./config";
import { US_EAST_2_BASTION_KEY_PAIR } from "./keypairs";

const US_EAST_2 = "us-east-2";
const US_EAST_2A = "us-east-2a";
const US_EAST_2B = "us-east-2b";

const M5_XLARGE = "m5.xlarge";
const T3_SMALL = "t3.small";

export const MOCKER_VPC_US_EAST_2 = {
  name: "mocker-vpc-us-east-2",
  availabilityZones: [US_EAST_2A, US_EAST_2B],
  region: US_EAST_2,
  cidrBlock: "10.42.0.0/16",
} satisfies VpcConfig;

export const BASTION_CONFIG_OHIO = {
  name: "mocker-experiment-bastion",
  iamRoleName: "mocker-bastion-role",
  instanceType: T3_SMALL,
  availabilityZone: US_EAST_2A,
  keyPairName: US_EAST_2_BASTION_KEY_PAIR,
  vpcConfig: MOCKER_VPC_US_EAST_2,
  region: US_EAST_2,
} satisfies BastionConfig;

export const MOCKER_US_EAST_2_CONFIG = {
  region: US_EAST_2,
  clusterName: "sachal-mocker-cluster",
  vpcConfig: MOCKER_VPC_US_EAST_2,
  bastionConfig: BASTION_CONFIG_OHIO,
} satisfies ClusterConfig;

export const MOCKER_CPU_NODE_GROUP = {
  instanceTypes: [M5_XLARGE],
  scaling: {
    minSize: 2,
    maxSize: 2,
    desiredSize: 2,
  },
} satisfies NodeGroupConfig;

export const DYNAMO_PLATFORM_1_4_0 = {
  dynamoVersion: "1.4.0",
  dynamoNamespace: "dynamo",
} satisfies DynamoPlatformConfig;

export const MOCKER_OHIO_1_DGD = {
  dgdName: "mocker-ohio-1",
  poolNames: ["mocker-cpu-ohio-1"],
  mockerReplicas: 2,
  modelName: "Qwen/Qwen3-32B",
  mockerSpeedupRatio: 1,
} satisfies DgdConfig;

export const MOCKER_OHIO_1_POOL = {
  name: "mocker-cpu-ohio-1",
  availabilityZone: US_EAST_2A,
  subnetSlot: 1,
  nodeGroupConfig: MOCKER_CPU_NODE_GROUP,
} satisfies DgdPoolConfig;

export const US_EAST2_DEPLOYMENT = {
  clusterConfig: MOCKER_US_EAST_2_CONFIG,
  dynamoPlatformConfig: DYNAMO_PLATFORM_1_4_0,
  dgdPools: [MOCKER_OHIO_1_POOL],
  dgds: [MOCKER_OHIO_1_DGD],
} satisfies DeploymentConfig;
