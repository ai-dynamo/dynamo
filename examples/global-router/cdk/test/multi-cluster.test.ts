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

import { cidrsOverlap } from "../lib/cidr";
import type { DeploymentConfig, MultiClusterConfig } from "../lib/config";
import {
  MOCKER_VPC_US_EAST_2,
  MOCKER_VPC_US_WEST_2,
  US_EAST2_DEPLOYMENT,
  US_WEST2_DEPLOYMENT,
} from "../lib/const";
import { DynamoMockerCpuStack } from "../lib/dynamo-mocker-cpu-stack";
import {
  KARMADA_VERSION,
  karmadaChartUrl,
  withKarmadaControlPlane,
} from "../lib/multi-cluster";
import { VALID_CONFIG } from "./fixtures";

interface Resource {
  readonly DeletionPolicy?: string;
  readonly DependsOn?: readonly string[];
  readonly Properties: Record<string, unknown>;
}

const MEMBER: MultiClusterConfig = {
  memberName: "test-west",
  peerCidrs: ["10.43.0.0/16"],
};

const HUB: MultiClusterConfig = {
  memberName: "test-hub",
  peerCidrs: ["10.43.0.0/16"],
  karmadaControlPlane: {
    version: "1.19.0",
    apiSourceCidrs: ["10.42.0.0/16", "10.43.0.0/16"],
  },
};

function synthesize(multiCluster?: MultiClusterConfig): Template {
  const config: DeploymentConfig = { ...VALID_CONFIG, multiCluster };
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

function karmadaChart(template: Template): Resource | undefined {
  return resources(template, "Custom::AWSCDK-EKS-HelmChart").find(
    (chart) => chart.Properties.Release === "karmada",
  );
}

function mcsCrdManifests(template: Template): Resource[] {
  return resources(template, "Custom::AWSCDK-EKS-KubernetesResource").filter(
    (manifest) =>
      JSON.stringify(manifest.Properties.Manifest).includes(
        "serviceexports.multicluster.x-k8s.io",
      ),
  );
}

function peerIngress(template: Template, description = "MCS pod traffic"): Resource[] {
  return resources(template, "AWS::EC2::SecurityGroupIngress").filter(
    (rule) => String(rule.Properties.Description).startsWith(description),
  );
}

/** Every `tag` value anywhere in the chart values. */
function imageTags(values: unknown): string[] {
  if (Array.isArray(values)) {
    return values.flatMap(imageTags);
  }
  if (values === null || typeof values !== "object") {
    return [];
  }
  return Object.entries(values).flatMap(([key, value]) =>
    key === "tag" && typeof value === "string" ? [value] : imageTags(value),
  );
}

test("a single-cluster deployment gets no Karmada or MCS resources", () => {
  const template = synthesize();
  assert.equal(karmadaChart(template), undefined);
  assert.equal(mcsCrdManifests(template).length, 0);
  assert.equal(peerIngress(template).length, 0);
  template.resourceCountIs("Custom::AWSCDK-EKS-KubernetesPatch", 0);
  assert.equal(template.findOutputs("KarmadaMemberName").KarmadaMemberName, undefined);
});

test("a member admits peer pod traffic, keeps pod addresses and installs the MCS CRDs", () => {
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

  const [crds] = mcsCrdManifests(template);
  assert.equal(crds.Properties.Overwrite, true);
  // Adopted CRDs may predate the stack; deleting one deletes every export.
  assert.equal(crds.DeletionPolicy, "Retain");
  const names = (JSON.parse(String(crds.Properties.Manifest)) as Array<{
    kind: string;
    metadata: { name: string };
  }>).map((crd) => `${crd.kind}/${crd.metadata.name}`);
  assert.deepEqual(names, [
    "CustomResourceDefinition/serviceexports.multicluster.x-k8s.io",
    "CustomResourceDefinition/serviceimports.multicluster.x-k8s.io",
  ]);

  assert.equal(karmadaChart(template), undefined);
  template.hasOutput("KarmadaMemberName", { Value: "test-west" });
  assert.equal(
    template.findOutputs("KarmadaApiEndpointCommand").KarmadaApiEndpointCommand,
    undefined,
  );
});

test("the hub runs a pinned Karmada control plane behind an internal load balancer", () => {
  const template = synthesize(HUB);
  const chart = karmadaChart(template);
  assert.ok(chart);
  assert.equal(chart.Properties.Chart, karmadaChartUrl("1.19.0"));
  assert.equal(
    chart.Properties.Chart,
    "https://github.com/karmada-io/karmada/releases/download/v1.19.0/karmada-chart-v1.19.0.tgz",
  );
  assert.equal(chart.Properties.Namespace, "karmada-system");
  assert.equal(chart.Properties.CreateNamespace, true);
  assert.equal(chart.Properties.Wait, true);
  // Helm must time out, and roll back, inside the kubectl Lambda's 15 minutes.
  assert.equal(chart.Properties.Timeout, "600s");
  // The chart's uninstall hook deletes the MCS CRDs and its namespace.
  assert.equal(chart.DeletionPolicy, "Retain");

  const values = JSON.parse(String(chart.Properties.Values)) as {
    installMode: string;
    cfssl: { image: { tag: string } };
    kubectl: { image: { tag?: string; digest: string } };
    staticResourceJob: { ttlSecondsAfterFinished: number };
    certs: { auto: { hosts: string[] } };
    apiServer: { serviceType: string; serviceAnnotations: Record<string, string> };
    controllerManager: { featureGates: Record<string, boolean> };
  };
  assert.equal(values.installMode, "host");
  // Helm 4 counts a deleted Job as never ready, so the setup Job must
  // outlive Helm's wait.
  assert.ok(values.staticResourceJob.ttlSecondsAfterFinished > 600);
  // A propagated ServiceImport brings its derived Service only with this on.
  assert.equal(values.controllerManager.featureGates.PropagateDeps, true);
  assert.equal(values.cfssl.image.tag, "v1.6.5");
  assert.match(values.kubectl.image.digest, /^sha256:[0-9a-f]{64}$/);
  assert.equal(values.kubectl.image.tag, undefined);
  assert.equal(values.apiServer.serviceType, "LoadBalancer");
  assert.deepEqual(values.apiServer.serviceAnnotations, {
    "service.beta.kubernetes.io/aws-load-balancer-type": "external",
    "service.beta.kubernetes.io/aws-load-balancer-nlb-target-type": "ip",
    "service.beta.kubernetes.io/aws-load-balancer-scheme": "internal",
    "service.beta.kubernetes.io/load-balancer-source-ranges": "10.42.0.0/16,10.43.0.0/16",
  });
  // The load balancer's name must be valid for the API certificate, and
  // the chart's own names must stay.
  assert.ok(values.certs.auto.hosts.includes("*.elb.us-west-2.amazonaws.com"));
  assert.ok(values.certs.auto.hosts.includes("*.{{ .Release.Namespace }}.svc"));
  assert.ok(values.certs.auto.hosts.includes("localhost"));
  for (const tag of imageTags(values)) {
    assert.ok(tag !== "latest", "every image the chart would pull as latest is pinned");
  }

  // Waits for the node groups and the load balancer controller.
  const charts = Object.entries(template.findResources("Custom::AWSCDK-EKS-HelmChart"));
  const controller = charts.find(
    ([, resource]) => resource.Properties.Release === "aws-load-balancer-controller",
  );
  assert.ok(controller);
  assert.ok(chart.DependsOn?.includes(controller[0]));
  const nodegroupIds = Object.keys(template.findResources("AWS::EKS::Nodegroup"));
  assert.ok(nodegroupIds.length > 0);
  for (const nodegroupId of nodegroupIds) {
    assert.ok(chart.DependsOn?.includes(nodegroupId), `waits for ${nodegroupId}`);
  }
  template.hasOutput("KarmadaMemberName", { Value: "test-hub" });
  template.hasOutput("KarmadaApiEndpointCommand", {
    Value:
      "kubectl get service -n karmada-system karmada-apiserver " +
      "-o jsonpath='{.status.loadBalancer.ingress[0].hostname}'",
  });
  template.hasOutput("KarmadaKubeconfigCommand", {
    Value:
      "kubectl get secret -n karmada-system karmada-kubeconfig " +
      "-o jsonpath='{.data.kubeconfig}' | base64 -d > karmada-apiserver.config",
  });
});

test("rejects invalid multi-cluster settings", () => {
  const cases: Array<[string, MultiClusterConfig, RegExp]> = [
    ["upper-case member name", { ...MEMBER, memberName: "West" }, /lowercase DNS label/],
    ["long member name", { ...MEMBER, memberName: "w".repeat(49) }, /at most 48/],
    ["bad peer range", { ...MEMBER, peerCidrs: ["10.43.0.0"] }, /VPC ranges/],
    ["peer wider than a VPC", { ...MEMBER, peerCidrs: ["10.0.0.0/8"] }, /VPC ranges/],
    ["public peer", { ...MEMBER, peerCidrs: ["8.8.0.0/16"] }, /VPC ranges/],
    ["unaligned peer", { ...MEMBER, peerCidrs: ["10.43.1.0/16"] }, /VPC ranges/],
    // The VPC CNI skips a zero-padded range, so its traffic would still be NATed.
    ["zero-padded peer", { ...MEMBER, peerCidrs: ["010.043.000.000/16"] }, /VPC ranges/],
    // VALID_CONFIG's own VPC is 10.42.0.0/16.
    ["own VPC as a peer", { ...MEMBER, peerCidrs: ["10.42.0.0/16"] }, /own VPC/],
    ["peer inside the own VPC", { ...MEMBER, peerCidrs: ["10.42.128.0/17"] }, /own VPC/],
    [
      "inexact Karmada version",
      { ...HUB, karmadaControlPlane: { ...HUB.karmadaControlPlane!, version: "1.19" } },
      /not supported/,
    ],
    [
      "Karmada version without matching CRDs",
      { ...HUB, karmadaControlPlane: { ...HUB.karmadaControlPlane!, version: "1.18.0" } },
      /not supported/,
    ],
    [
      "bad API source range",
      { ...HUB, karmadaControlPlane: { ...HUB.karmadaControlPlane!, apiSourceCidrs: ["any"] } },
      /API source CIDRs/,
    ],
  ];
  for (const [name, multiCluster, message] of cases) {
    assert.throws(() => synthesize(multiCluster), message, name);
  }
});

test("Ohio is the Karmada hub and West a member, each peering with the other", () => {
  const ohio = US_EAST2_DEPLOYMENT.multiCluster;
  const west = US_WEST2_DEPLOYMENT.multiCluster;
  assert.equal(ohio.memberName, "aws-ohio");
  assert.deepEqual(ohio.peerCidrs, [MOCKER_VPC_US_WEST_2.cidrBlock]);
  assert.deepEqual(ohio.karmadaControlPlane.apiSourceCidrs, [
    MOCKER_VPC_US_EAST_2.cidrBlock,
    MOCKER_VPC_US_WEST_2.cidrBlock,
  ]);
  assert.equal(ohio.karmadaControlPlane.version, KARMADA_VERSION);
  assert.equal(west.memberName, "aws-west");
  assert.deepEqual(west.peerCidrs, [MOCKER_VPC_US_EAST_2.cidrBlock]);
  assert.equal("karmadaControlPlane" in west, false);
});

test("the Karmada control plane is installed only with -c karmada=true", () => {
  const hub: DeploymentConfig = { ...VALID_CONFIG, multiCluster: HUB };
  const withoutFlag = withKarmadaControlPlane(hub, false);
  assert.equal(withoutFlag.multiCluster?.karmadaControlPlane, undefined);
  assert.deepEqual(withoutFlag.multiCluster?.peerCidrs, HUB.peerCidrs);
  assert.equal(withKarmadaControlPlane(hub, true), hub);

  const member: DeploymentConfig = { ...VALID_CONFIG, multiCluster: MEMBER };
  assert.equal(withKarmadaControlPlane(member, false), member);
  assert.throws(() => withKarmadaControlPlane(member, true), /karmada=true/);
  assert.throws(() => withKarmadaControlPlane(VALID_CONFIG, true), /karmada=true/);
});

test("peer ranges overlap only when they share an address", () => {
  assert.equal(cidrsOverlap("10.42.0.0/16", "10.42.128.0/17"), true);
  assert.equal(cidrsOverlap("10.0.0.0/8", "10.42.0.0/16"), true);
  assert.equal(cidrsOverlap("10.42.0.0/16", "10.43.0.0/16"), false);
  assert.throws(() => cidrsOverlap("10.42.0.0", "10.43.0.0/16"), /not an IPv4 CIDR/);
});
