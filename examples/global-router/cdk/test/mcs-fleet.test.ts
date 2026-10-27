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
import { App } from "aws-cdk-lib";
import { Template } from "aws-cdk-lib/assertions";
import {
  McsFleet,
  McsMemberStack,
  McsPeeringsStack,
  McsRoutesStack,
  validateMcsFleet,
} from "../lib/mcs-fleet";

const names = ["ohio", "west", "tokyo", "saopaulo"];
const regions = ["us-east-2", "us-west-2", "ap-northeast-1", "sa-east-1"];
const fleet: McsFleet = {
  members: names.map((id, i) => ({
    id,
    siteId: `aws-${id}`,
    region: regions[i],
    clusterName: `example-${id}`,
    vpcId: `vpc-0000000000000000${i}`,
    cidr: `10.${42 + i}.0.0/16`,
    routeTableIds: [`rtb-0000000000000000${i}`, `rtb-0000000000000001${i}`],
    clusterSecurityGroupId: `sg-0000000000000000${i}`,
    kubectlServiceToken: `arn:aws:lambda:${regions[i]}:111111111111:function:example-kubectl`,
  })),
  links: names.flatMap((left, i) =>
    names
      .slice(i + 1)
      .map((right, j) => ({
        left,
        right,
        peeringId: `pcx-000000000000000${i}${j}`,
        external: left === "ohio" && right === "west",
      })),
  ),
};
const env = { account: "111111111111", region: "us-east-2" };

test("four members require six direct links and nonoverlapping private pod networks", () => {
  validateMcsFleet(fleet);
  assert.throws(
    () => validateMcsFleet({ ...fleet, links: fleet.links.slice(1) }),
    /every member pair/,
  );
  assert.throws(
    () =>
      validateMcsFleet({ ...fleet, links: [...fleet.links, fleet.links[0]] }),
    /duplicate/,
  );
  assert.throws(
    () =>
      validateMcsFleet({
        ...fleet,
        members: fleet.members.map((m, i) =>
          i === 1 ? { ...m, cidr: fleet.members[0].cidr } : m,
        ),
      }),
    /overlap/,
  );
  assert.throws(
    () =>
      validateMcsFleet({
        ...fleet,
        members: fleet.members.map((m, i) =>
          i === 1 ? { ...m, cidr: "0.0.0.0/0" } : m,
        ),
      }),
    /private VPC/,
  );
});
test("mesh leaves existing Ohio-West connection and route ownership untouched", () => {
  const props = { env, fleet, member: fleet.members[0] };
  const peering = Template.fromStack(
    new McsPeeringsStack(new App(), "Peerings", props),
  );
  peering.resourceCountIs("AWS::EC2::VPCPeeringConnection", 2);
  const routes = Template.fromStack(
    new McsRoutesStack(new App(), "Routes", props),
  );
  routes.resourceCountIs("AWS::EC2::Route", 4);
  const values = Object.values(routes.findResources("AWS::EC2::Route"));
  assert.deepEqual(
    new Set(values.map((v) => v.Properties.DestinationCidrBlock)),
    new Set(["10.44.0.0/16", "10.45.0.0/16"]),
  );
  assert.throws(
    () =>
      new McsRoutesStack(new App(), "MissingId", {
        ...props,
        fleet: {
          ...fleet,
          links: fleet.links.map((l) => ({ ...l, peeringId: undefined })),
        },
      }),
    /populate peering IDs/,
  );
});
test("member setup never owns cluster, node groups, or the running hub", () => {
  const stack = new McsMemberStack(new App(), "Member", {
    env,
    fleet,
    member: {
      ...fleet.members[0],
      hubApiService: { name: "karmada-apiserver", namespace: "karmada-system" },
    },
  });
  const t = Template.fromStack(stack);
  for (const type of [
    "AWS::EKS::Cluster",
    "AWS::EKS::Nodegroup",
    "AWS::EC2::Instance",
    "AWS::IAM::Role",
    "Custom::AWSCDK-EKS-HelmChart",
  ])
    t.resourceCountIs(type, 0);
  t.resourceCountIs("AWS::EC2::SecurityGroupIngress", 6);
  const patches = Object.values(
    t.findResources("Custom::AWSCDK-EKS-KubernetesPatch"),
  );
  assert.equal(patches.length, 2);
  assert.ok(
    JSON.stringify(patches).includes("AWS_VPC_K8S_CNI_EXCLUDE_SNAT_CIDRS"),
  );
  assert.ok(
    JSON.stringify(patches).includes("10.43.0.0/16,10.44.0.0/16,10.45.0.0/16"),
  );
  for (const value of Object.values(t.toJSON().Resources) as {
    Type: string;
    DeletionPolicy?: string;
  }[])
    if (value.Type !== "AWS::CDK::Metadata")
      assert.equal(value.DeletionPolicy, "Retain");
  assert.equal(stack.terminationProtection, true);
});
