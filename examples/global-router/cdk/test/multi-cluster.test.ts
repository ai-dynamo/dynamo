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

import type { DeploymentConfig, MultiClusterConfig } from "../lib/config";
import {
  MOCKER_VPC_US_EAST_2,
  MOCKER_VPC_US_WEST_2,
  US_EAST2_DEPLOYMENT,
  US_WEST2_DEPLOYMENT,
} from "../lib/const";
import { DynamoMockerCpuStack } from "../lib/dynamo-mocker-cpu-stack";
import { karmadaChartUrl } from "../lib/multi-cluster";
import { VALID_CONFIG } from "./fixtures";

interface Resource {
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

function peerIngress(template: Template): Resource[] {
  return resources(template, "AWS::EC2::SecurityGroupIngress").filter(
    (rule) => String(rule.Properties.Description).startsWith("MCS pod traffic"),
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

  const [patch] = resources(template, "Custom::AWSCDK-EKS-KubernetesPatch");
  assert.equal(patch.Properties.ResourceName, "daemonset/aws-node");
  assert.equal(patch.Properties.ResourceNamespace, "kube-system");
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

  const values = JSON.parse(String(chart.Properties.Values)) as {
    installMode: string;
    certs: { auto: { hosts: string[] } };
    apiServer: { serviceType: string; serviceAnnotations: Record<string, string> };
    scheduler: { image: { tag: string } };
    controllerManager: { image: { tag: string } };
  };
  assert.equal(values.installMode, "host");
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
  assert.equal(values.scheduler.image.tag, "v1.19.0");
  assert.equal(values.controllerManager.image.tag, "v1.19.0");
  for (const tag of imageTags(values)) {
    assert.ok(tag !== "latest", "every image the chart would pull as latest is pinned");
  }

  // Waits for node groups and the load balancer controller.
  assert.ok((chart.DependsOn ?? []).length >= 2);
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
    ["bad peer range", { ...MEMBER, peerCidrs: ["10.43.0.0"] }, /valid IPv4 CIDR/],
    // VALID_CONFIG's own VPC is 10.42.0.0/16.
    ["own VPC as a peer", { ...MEMBER, peerCidrs: ["10.42.0.0/16"] }, /own VPC/],
    [
      "inexact Karmada version",
      { ...HUB, karmadaControlPlane: { ...HUB.karmadaControlPlane!, version: "1.19" } },
      /exact release/,
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
  assert.equal(west.memberName, "aws-west");
  assert.deepEqual(west.peerCidrs, [MOCKER_VPC_US_EAST_2.cidrBlock]);
  assert.equal("karmadaControlPlane" in west, false);
});
