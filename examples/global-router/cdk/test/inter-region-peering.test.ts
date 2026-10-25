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

test("both mode creates one Ohio peering connection and routes both ways", () => {
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
  assertPeeringRoutes(peeringTemplate, "10.43.0.0/16", [
    ...POC_PEERING_OHIO.privateRouteTableIds,
    ...POC_PEERING_OHIO.publicRouteTableIds,
  ]);
  assertPeeringRoutes(westRoutesTemplate, "10.42.0.0/16", [
    ...POC_PEERING_WEST.privateRouteTableIds,
    ...POC_PEERING_WEST.publicRouteTableIds,
  ]);
  // Every route, public ones included, uses this peering connection; West
  // reads its ID across Regions.
  for (const route of Object.values(peeringTemplate.findResources("AWS::EC2::Route"))) {
    assert.deepEqual(route.Properties.VpcPeeringConnectionId, {
      "Fn::GetAtt": ["OhioWestVpcPeering", "Id"],
    });
  }
  for (const route of Object.values(westRoutesTemplate.findResources("AWS::EC2::Route"))) {
    assert.match(JSON.stringify(route.Properties.VpcPeeringConnectionId), /OhioWestVpcPeeringId/);
  }
});

test("keeps private route IDs and adds public routes only when configured", () => {
  const app = new App();
  addInterRegionPeering(
    app,
    { ...POC_PEERING_OHIO, publicRouteTableIds: ["rtb-0hiopublic1", "rtb-0hiopublic2"] },
    { ...POC_PEERING_WEST, publicRouteTableIds: [] },
    "111111111111",
  );
  const peeringStack = app.node.tryFindChild("OhioWestPeeringRoutes") as Stack;
  const westRoutesStack = app.node.tryFindChild("WestPrivateRoutes") as Stack;
  const ohioRoutes = Template.fromStack(peeringStack).findResources("AWS::EC2::Route");
  const westRoutes = Template.fromStack(westRoutesStack).findResources("AWS::EC2::Route");

  // Logical IDs of deployed routes must not change, or CloudFormation replaces them.
  assert.ok("OhioToWestPrivateRoute1" in ohioRoutes);
  assert.ok("WestToOhioPrivateRoute1" in westRoutes);
  assert.equal(
    (ohioRoutes.OhioToWestPublicRoute2 as { Properties: Record<string, unknown> })
      .Properties.RouteTableId,
    "rtb-0hiopublic2",
  );
  assert.equal(
    Object.keys(ohioRoutes).length,
    POC_PEERING_OHIO.privateRouteTableIds.length + 2,
  );
  assert.equal(
    Object.keys(westRoutes).length,
    POC_PEERING_WEST.privateRouteTableIds.length,
  );
});

test("rejects a route table listed as both private and public", () => {
  assert.throws(
    () =>
      addInterRegionPeering(
        new App(),
        POC_PEERING_OHIO,
        {
          ...POC_PEERING_WEST,
          publicRouteTableIds: [POC_PEERING_WEST.privateRouteTableIds[0]],
        },
        "111111111111",
      ),
    /both private and public/,
  );
});

function assertPeeringRoutes(
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
