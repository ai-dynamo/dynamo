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

import { Stack, aws_ec2 as ec2 } from "aws-cdk-lib";
import type { Construct } from "constructs";

import type { PeeringRegionConfig } from "./config";

/**
 * Add inter-Region peering and private routes in dedicated stacks. Keeping the
 * routes outside the VPC stacks avoids a cross-Region CloudFormation cycle.
 */
export function addInterRegionPeering(
  scope: Construct,
  ohioVpc: PeeringRegionConfig,
  westVpc: PeeringRegionConfig,
  account?: string,
): void {
  if (ohioVpc.region === westVpc.region) {
    throw new Error("Inter-Region VPC peering requires distinct Regions.");
  }
  if (ohioVpc.cidrBlock === westVpc.cidrBlock) {
    throw new Error("Inter-Region VPC peering requires distinct VPC CIDRs.");
  }
  if (ohioVpc.vpcId === westVpc.vpcId) {
    throw new Error("Inter-Region VPC peering requires distinct VPC IDs.");
  }

  const peeringRoutesStack = new Stack(scope, "OhioWestPeeringRoutes", {
    crossRegionReferences: true,
    description: "Ohio to West VPC peering and Ohio private routes",
    env: { account, region: ohioVpc.region },
  });
  const peering = new ec2.CfnVPCPeeringConnection(
    peeringRoutesStack,
    "OhioWestVpcPeering",
    {
      peerRegion: westVpc.region,
      peerVpcId: westVpc.vpcId,
      tags: [
        { key: "Name", value: "mocker-ohio-west-private-peering" },
        { key: "Project", value: "dynamo-mocker-cpu" },
      ],
      vpcId: ohioVpc.vpcId,
    },
  );
  addPrivateRoutes(
    peeringRoutesStack,
    ohioVpc.privateRouteTableIds,
    westVpc.cidrBlock,
    peering.attrId,
    "OhioToWest",
  );

  const westRoutesStack = new Stack(scope, "WestPrivateRoutes", {
    crossRegionReferences: true,
    description: "West private routes to the Ohio VPC over peering",
    env: { account, region: westVpc.region },
  });
  addPrivateRoutes(
    westRoutesStack,
    westVpc.privateRouteTableIds,
    ohioVpc.cidrBlock,
    peering.attrId,
    "WestToOhio",
  );
}

function addPrivateRoutes(
  stack: Stack,
  routeTableIds: readonly string[],
  destinationCidr: string,
  peeringId: string,
  idPrefix: string,
): void {
  for (const [index, routeTableId] of routeTableIds.entries()) {
    new ec2.CfnRoute(stack, `${idPrefix}PrivateRoute${index + 1}`, {
      destinationCidrBlock: destinationCidr,
      routeTableId,
      vpcPeeringConnectionId: peeringId,
    });
  }
}
