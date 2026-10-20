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

import assert from "node:assert/strict";
import { test } from "node:test";

import { App, Stack } from "aws-cdk-lib";
import { Template } from "aws-cdk-lib/assertions";

import { POC_PEERING_OHIO, POC_PEERING_WEST } from "../lib/const";
import { addInterRegionPeering } from "../lib/inter-region-peering";

test("both mode creates one Ohio peering connection and private routes both ways", () => {
  const app = new App();
  addInterRegionPeering(
    app,
    POC_PEERING_OHIO,
    POC_PEERING_WEST,
    "111111111111",
  );

  const stacks = app.node.children.filter((child) => child instanceof Stack);
  assert.equal(stacks.length, 2);
  const peeringStack = app.node.tryFindChild("OhioWestPeeringRoutes") as Stack;
  const westRoutesStack = app.node.tryFindChild("WestPrivateRoutes") as Stack;
  const peeringTemplate = Template.fromStack(peeringStack);
  const westRoutesTemplate = Template.fromStack(westRoutesStack);
  peeringTemplate.resourceCountIs("AWS::EC2::VPCPeeringConnection", 1);
  westRoutesTemplate.resourceCountIs("AWS::EC2::VPCPeeringConnection", 0);
  peeringTemplate.hasResourceProperties("AWS::EC2::VPCPeeringConnection", {
    PeerRegion: "us-west-2",
    VpcId: POC_PEERING_OHIO.vpcId,
    PeerVpcId: POC_PEERING_WEST.vpcId,
    Tags: [
      { Key: "Name", Value: "mocker-ohio-west-private-peering" },
      { Key: "Project", Value: "dynamo-mocker-cpu" },
    ],
  });
  assertPrivateRoutes(peeringTemplate, "10.43.0.0/16", POC_PEERING_OHIO.privateRouteTableIds);
  assertPrivateRoutes(westRoutesTemplate, "10.42.0.0/16", POC_PEERING_WEST.privateRouteTableIds);
});

function assertPrivateRoutes(
  template: Template,
  destinationCidr: string,
  expectedRouteTableIds: readonly string[],
): void {
  const routes = Object.values(
    template.findResources("AWS::EC2::Route"),
  ) as Array<{ Properties: Record<string, unknown> }>;
  const peeringRoutes = routes.filter(
    ({ Properties }) => Properties.DestinationCidrBlock === destinationCidr,
  );
  assert.equal(peeringRoutes.length, expectedRouteTableIds.length);
  assert.deepEqual(
    peeringRoutes.map(({ Properties }) => Properties.RouteTableId).sort(),
    [...expectedRouteTableIds].sort(),
  );
  for (const { Properties } of peeringRoutes) {
    assert.ok(Properties.VpcPeeringConnectionId);
    assert.ok(Properties.RouteTableId);
  }
}
