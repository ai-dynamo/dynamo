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
  PeeringRegionConfig,
  VpcConfig,
} from "./config";
import {
  US_EAST_2_BASTION_KEY_PAIR,
  US_WEST_2_BASTION_KEY_PAIR,
} from "./keypairs";
import { KARMADA_VERSION } from "./multi-cluster";

const US_EAST_2 = "us-east-2";
const US_EAST_2A = "us-east-2a";
const US_EAST_2B = "us-east-2b";
const US_WEST_2 = "us-west-2";
const US_WEST_2A = "us-west-2a";
const US_WEST_2B = "us-west-2b";

const M5_XLARGE = "m5.xlarge";
const T3_SMALL = "t3.small";

// Local POC images built from Dynamo a47ad40894 and published to both Regions.
const POC_ECR_OHIO =
  "197002356433.dkr.ecr.us-east-2.amazonaws.com/global-router-poc";
const POC_ECR_WEST =
  "197002356433.dkr.ecr.us-west-2.amazonaws.com/global-router-poc";
const POC_PLANNER_DIGEST =
  "sha256:23b02f9e4b0b981e6f9118fceb6afcfb08e05261cde2701b7a7967f2ff27925c";
const POC_RELAY_DIGEST =
  "sha256:f4199e6f5c2fcd8d058216eb0cb37837eee0e51629aff8553a0d0aad178ec0b2";

export const MOCKER_VPC_US_EAST_2 = {
  name: "mocker-vpc-us-east-2",
  availabilityZones: [US_EAST_2A, US_EAST_2B],
  region: US_EAST_2,
  cidrBlock: "10.42.0.0/16",
} satisfies VpcConfig;

export const BASTION_CONFIG_OHIO = {
  name: "mocker-experiment-bastion",
  amiId: "ami-070db52ff58e36a5a",
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

export const MOCKER_VPC_US_WEST_2 = {
  name: "mocker-vpc-us-west-2",
  availabilityZones: [US_WEST_2A, US_WEST_2B],
  region: US_WEST_2,
  cidrBlock: "10.43.0.0/16",
} satisfies VpcConfig;

export const BASTION_CONFIG_WEST = {
  name: "mocker-west-experiment-bastion",
  amiId: "ami-07a288b554ded85f8",
  iamRoleName: "mocker-west-bastion-role",
  instanceType: T3_SMALL,
  availabilityZone: US_WEST_2A,
  keyPairName: US_WEST_2_BASTION_KEY_PAIR,
  vpcConfig: MOCKER_VPC_US_WEST_2,
  region: US_WEST_2,
} satisfies BastionConfig;

export const MOCKER_US_WEST_2_CONFIG = {
  region: US_WEST_2,
  clusterName: "sachal-mocker-west-cluster",
  vpcConfig: MOCKER_VPC_US_WEST_2,
  bastionConfig: BASTION_CONFIG_WEST,
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
  workerRuntimeNamespace: "dynamo-mocker-ohio-1-27f7037e",
  frontendRouterMode: "kv",
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
  mockerImage: `${POC_ECR_OHIO}@${POC_PLANNER_DIGEST}`,
  mockerRuntimeVersionOverride: "1.6.0",
  kvDcRelayImage: `${POC_ECR_OHIO}@${POC_RELAY_DIGEST}`,
  clusterConfig: MOCKER_US_EAST_2_CONFIG,
  privateFrontendSourceCidrs: [
    MOCKER_VPC_US_EAST_2.cidrBlock,
    MOCKER_VPC_US_WEST_2.cidrBlock,
  ],
  privateRelaySourceCidrs: [
    MOCKER_VPC_US_EAST_2.cidrBlock,
    MOCKER_VPC_US_WEST_2.cidrBlock,
  ],
  dynamoPlatformConfig: DYNAMO_PLATFORM_1_4_0,
  dgdPools: [MOCKER_OHIO_1_POOL],
  dgds: [MOCKER_OHIO_1_DGD],
  // Ohio is the Karmada hub. Member names match the Router's site IDs.
  multiCluster: {
    memberName: "aws-ohio",
    peerCidrs: [MOCKER_VPC_US_WEST_2.cidrBlock],
    karmadaControlPlane: {
      version: KARMADA_VERSION,
      apiSourceCidrs: [
        MOCKER_VPC_US_EAST_2.cidrBlock,
        MOCKER_VPC_US_WEST_2.cidrBlock,
      ],
    },
  },
} satisfies DeploymentConfig;

export const MOCKER_WEST_1_DGD = {
  dgdName: "mocker-west-1",
  workerRuntimeNamespace: "dynamo-mocker-west-1-eb50ba2c",
  frontendRouterMode: "kv",
  poolNames: ["mocker-cpu-west-1"],
  mockerReplicas: 2,
  modelName: "Qwen/Qwen3-32B",
  mockerSpeedupRatio: 1,
} satisfies DgdConfig;

export const MOCKER_WEST_1_POOL = {
  name: "mocker-cpu-west-1",
  availabilityZone: US_WEST_2A,
  subnetSlot: 1,
  nodeGroupConfig: MOCKER_CPU_NODE_GROUP,
} satisfies DgdPoolConfig;

export const US_WEST2_DEPLOYMENT = {
  mockerImage: `${POC_ECR_WEST}@${POC_PLANNER_DIGEST}`,
  mockerRuntimeVersionOverride: "1.6.0",
  kvDcRelayImage: `${POC_ECR_WEST}@${POC_RELAY_DIGEST}`,
  clusterConfig: MOCKER_US_WEST_2_CONFIG,
  privateFrontendSourceCidrs: [
    MOCKER_VPC_US_EAST_2.cidrBlock,
    MOCKER_VPC_US_WEST_2.cidrBlock,
  ],
  privateRelaySourceCidrs: [
    MOCKER_VPC_US_EAST_2.cidrBlock,
    MOCKER_VPC_US_WEST_2.cidrBlock,
  ],
  dynamoPlatformConfig: DYNAMO_PLATFORM_1_4_0,
  dgdPools: [MOCKER_WEST_1_POOL],
  dgds: [MOCKER_WEST_1_DGD],
  multiCluster: {
    memberName: "aws-west",
    peerCidrs: [MOCKER_VPC_US_EAST_2.cidrBlock],
  },
} satisfies DeploymentConfig;

export const DEPLOYMENTS = {
  ohio: US_EAST2_DEPLOYMENT,
  west: US_WEST2_DEPLOYMENT,
} satisfies Record<string, DeploymentConfig>;

// From the regional CloudFormation stacks in account 197002356433 on 2026-09-29.
// Public route tables added on 2026-10-20. Refresh these IDs if either regional VPC is rebuilt.
export const POC_PEERING_OHIO = {
  region: US_EAST_2,
  cidrBlock: MOCKER_VPC_US_EAST_2.cidrBlock,
  vpcId: "vpc-06c34093b7a5acb5f",
  privateRouteTableIds: [
    "rtb-0b1892e84e10b25d4", // clusterSubnet1
    "rtb-065e446706956ba1e", // clusterSubnet2
    "rtb-0e8ad0043f450774a", // dgd-pool-slot-1Subnet1
    "rtb-0aaeffd32412af270", // dgd-pool-slot-1Subnet2
  ],
  // publicSubnet1 and publicSubnet2: the only route tables with an internet gateway route.
  publicRouteTableIds: ["rtb-087da2b329a497b8b", "rtb-0c9ccaee37e35b64a"],
} satisfies PeeringRegionConfig;

export const POC_PEERING_WEST = {
  region: US_WEST_2,
  cidrBlock: MOCKER_VPC_US_WEST_2.cidrBlock,
  vpcId: "vpc-0b22a247e4f8e8d7c",
  privateRouteTableIds: [
    "rtb-08af3bdd3bd2080b6", // clusterSubnet1
    "rtb-0d4ebeb19aa31a74c", // clusterSubnet2
    "rtb-0755738960722e63b", // dgd-pool-slot-1Subnet1
    "rtb-0c00d24e9c4e92bef", // dgd-pool-slot-1Subnet2
  ],
  // publicSubnet1 and publicSubnet2: the only route tables with an internet gateway route.
  publicRouteTableIds: ["rtb-0903883bb4ad61a14", "rtb-0b8371fc3cd38a7f8"],
} satisfies PeeringRegionConfig;

export type DeploymentName = keyof typeof DEPLOYMENTS;

export function selectDeployment(name: unknown): DeploymentConfig {
  if (typeof name !== "string" || !Object.hasOwn(DEPLOYMENTS, name)) {
    throw new Error(
      `Unknown deployment ${String(name)}. Choose one of: ${Object.keys(DEPLOYMENTS).join(", ")}.`,
    );
  }

  return DEPLOYMENTS[name as DeploymentName];
}
