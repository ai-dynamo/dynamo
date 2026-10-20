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
 * Add inter-Region peering and its routes in dedicated stacks. Keeping the
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
  for (const vpc of [ohioVpc, westVpc]) {
    const publicIds = vpc.publicRouteTableIds ?? [];
    if (publicIds.some((id) => vpc.privateRouteTableIds.includes(id))) {
      throw new Error(
        `VPC ${vpc.vpcId} lists a route table as both private and public.`,
      );
    }
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
  addRoutes(
    peeringRoutesStack,
    ohioVpc.privateRouteTableIds,
    westVpc.cidrBlock,
    peering.attrId,
    "OhioToWestPrivate",
  );
  addRoutes(
    peeringRoutesStack,
    ohioVpc.publicRouteTableIds ?? [],
    westVpc.cidrBlock,
    peering.attrId,
    "OhioToWestPublic",
  );

  const westRoutesStack = new Stack(scope, "WestPrivateRoutes", {
    crossRegionReferences: true,
    description: "West private routes to the Ohio VPC over peering",
    env: { account, region: westVpc.region },
  });
  addRoutes(
    westRoutesStack,
    westVpc.privateRouteTableIds,
    ohioVpc.cidrBlock,
    peering.attrId,
    "WestToOhioPrivate",
  );
  addRoutes(
    westRoutesStack,
    westVpc.publicRouteTableIds ?? [],
    ohioVpc.cidrBlock,
    peering.attrId,
    "WestToOhioPublic",
  );
}

/** Logical IDs are `<prefix>Route<n>`, so existing private routes keep theirs. */
function addRoutes(
  stack: Stack,
  routeTableIds: readonly string[],
  destinationCidr: string,
  peeringId: string,
  idPrefix: string,
): void {
  for (const [index, routeTableId] of routeTableIds.entries()) {
    new ec2.CfnRoute(stack, `${idPrefix}Route${index + 1}`, {
      destinationCidrBlock: destinationCidr,
      routeTableId,
      vpcPeeringConnectionId: peeringId,
    });
  }
}
