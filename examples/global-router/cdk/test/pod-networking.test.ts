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

import { cidrsOverlap } from "../lib/foundation/cidr";
import type { DeploymentConfig } from "../testing/legacy/config";
import type { PeerNetworkingConfig } from "../lib/foundation/config";
import {
  MOCKER_VPC_US_EAST_2,
  MOCKER_VPC_US_WEST_2,
  US_EAST2_DEPLOYMENT,
  US_WEST2_DEPLOYMENT,
} from "../testing/legacy/const";
import { DynamoMockerCpuStack } from "../testing/legacy/dynamo-mocker-cpu-stack";
import { VALID_CONFIG } from "./fixtures";

interface Resource {
  readonly DeletionPolicy?: string;
  readonly DependsOn?: readonly string[];
  readonly Properties: Record<string, unknown>;
}

const MEMBER: PeerNetworkingConfig = {
  peerCidrs: ["10.43.0.0/16"],
};

function synthesize(peerNetworking?: PeerNetworkingConfig): Template {
  const config: DeploymentConfig = { ...VALID_CONFIG, peerNetworking };
  const app = new App();
  const stack = new DynamoMockerCpuStack(app, "TestStack", {
    config,
    env: { account: "111111111111", region: config.clusterConfig.region },
  });
  return Template.fromStack(stack);
}

function resources(template: Template, type: string): Resource[] {
  return Object.values(template.findResources(type)) as Resource[];
}

function peerIngress(template: Template, description = "Pod traffic"): Resource[] {
  return resources(template, "AWS::EC2::SecurityGroupIngress").filter(
    (rule) => String(rule.Properties.Description).startsWith(description),
  );
}

test("a single-cluster deployment gets no peer networking resources", () => {
  const template = synthesize();
  assert.equal(peerIngress(template).length, 0);
  template.resourceCountIs("Custom::AWSCDK-EKS-KubernetesPatch", 0);
  assert.doesNotMatch(JSON.stringify(template.toJSON()), /karmada|multicluster\.x-k8s\.io|ServiceExport|ServiceImport/i);
});

test("peer networking admits pod traffic and preserves source addresses without discovery controllers", () => {
  const template = synthesize(MEMBER);

  const rules = peerIngress(template);
  assert.equal(rules.length, 1);
  assert.equal(rules[0].Properties.CidrIp, "10.43.0.0/16");
  assert.equal(rules[0].Properties.IpProtocol, "tcp");
  assert.deepEqual([rules[0].Properties.FromPort, rules[0].Properties.ToPort], [0, 65535]);
  // Pods use their node's security groups. The node launch templates set none,
  // so EKS gives the nodes the cluster security group, which this rule is on.
  const [clusterId] = Object.keys(template.findResources("AWS::EKS::Cluster"));
  assert.deepEqual(rules[0].Properties.GroupId, {
    "Fn::GetAtt": [clusterId, "ClusterSecurityGroupId"],
  });
  const nodegroups = resources(template, "AWS::EKS::Nodegroup");
  assert.ok(nodegroups.length > 0);
  for (const nodegroup of nodegroups) {
    const { Id } = nodegroup.Properties.LaunchTemplate as { Id: { Ref: string } };
    const data = template.toJSON().Resources[Id.Ref].Properties.LaunchTemplateData;
    for (const key of ["SecurityGroupIds", "SecurityGroups", "NetworkInterfaces"]) {
      assert.equal(data[key], undefined, `node launch template sets ${key}`);
    }
  }
  // Path MTU Discovery over the peering needs ICMP "fragmentation needed".
  const icmp = peerIngress(template, "Path MTU Discovery");
  assert.equal(icmp.length, 1);
  assert.equal(icmp[0].Properties.CidrIp, "10.43.0.0/16");
  assert.equal(icmp[0].Properties.IpProtocol, "icmp");
  assert.deepEqual([icmp[0].Properties.FromPort, icmp[0].Properties.ToPort], [3, 4]);
  assert.deepEqual(icmp[0].Properties.GroupId, rules[0].Properties.GroupId);

  const [patch] = resources(template, "Custom::AWSCDK-EKS-KubernetesPatch");
  assert.equal(patch.Properties.ResourceName, "daemonset/aws-node");
  assert.equal(patch.Properties.ResourceNamespace, "kube-system");
  // A JSON merge patch would replace the whole container list.
  assert.equal(patch.Properties.PatchType, "strategic");
  const env = (patch: unknown) =>
    (patch as { spec: { template: { spec: { containers: Array<{ name: string; env: unknown[] }> } } } })
      .spec.template.spec.containers[0];
  const applied = env(JSON.parse(String(patch.Properties.ApplyPatchJson)));
  assert.equal(applied.name, "aws-node");
  assert.deepEqual(applied.env, [
    { name: "AWS_VPC_K8S_CNI_EXCLUDE_SNAT_CIDRS", value: "10.43.0.0/16" },
  ]);
  assert.deepEqual(env(JSON.parse(String(patch.Properties.RestorePatchJson))).env, [
    { $patch: "delete", name: "AWS_VPC_K8S_CNI_EXCLUDE_SNAT_CIDRS" },
  ]);

  assert.doesNotMatch(JSON.stringify(template.toJSON()), /karmada|multicluster\.x-k8s\.io|ServiceExport|ServiceImport/i);
});

test("rejects invalid peer networking settings", () => {
  const cases: Array<[string, PeerNetworkingConfig, RegExp]> = [
    ["bad peer range", { ...MEMBER, peerCidrs: ["10.43.0.0"] }, /VPC ranges/],
    ["peer wider than a VPC", { ...MEMBER, peerCidrs: ["10.0.0.0/8"] }, /VPC ranges/],
    ["public peer", { ...MEMBER, peerCidrs: ["8.8.0.0/16"] }, /VPC ranges/],
    ["unaligned peer", { ...MEMBER, peerCidrs: ["10.43.1.0/16"] }, /VPC ranges/],
    // The VPC CNI skips a zero-padded range, so its traffic would still be NATed.
    ["zero-padded peer", { ...MEMBER, peerCidrs: ["010.043.000.000/16"] }, /VPC ranges/],
    // VALID_CONFIG's own VPC is 10.42.0.0/16.
    ["own VPC as a peer", { ...MEMBER, peerCidrs: ["10.42.0.0/16"] }, /own VPC/],
    ["peer inside the own VPC", { ...MEMBER, peerCidrs: ["10.42.128.0/17"] }, /own VPC/],
  ];
  for (const [name, peerNetworking, message] of cases) {
    assert.throws(() => synthesize(peerNetworking), message, name);
  }
});

test("Ohio and West each allow direct traffic from the other VPC", () => {
  assert.deepEqual(US_EAST2_DEPLOYMENT.peerNetworking.peerCidrs, [MOCKER_VPC_US_WEST_2.cidrBlock]);
  assert.deepEqual(US_WEST2_DEPLOYMENT.peerNetworking.peerCidrs, [MOCKER_VPC_US_EAST_2.cidrBlock]);
});

test("peer ranges overlap only when they share an address", () => {
  assert.equal(cidrsOverlap("10.42.0.0/16", "10.42.128.0/17"), true);
  assert.equal(cidrsOverlap("10.0.0.0/8", "10.42.0.0/16"), true);
  assert.equal(cidrsOverlap("10.42.0.0/16", "10.43.0.0/16"), false);
  assert.throws(() => cidrsOverlap("10.42.0.0", "10.43.0.0/16"), /not an IPv4 CIDR/);
});
