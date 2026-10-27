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

import { readFileSync } from "node:fs";
import {
  App,
  Stack,
  Aspects,
  CfnResource,
  CliCredentialsStackSynthesizer,
  LegacyStackSynthesizer,
  RemovalPolicy,
  aws_s3 as s3,
  aws_iam as iam,
  aws_ec2 as ec2,
  CfnOutput,
  Tags,
} from "aws-cdk-lib";
import * as eks from "aws-cdk-lib/aws-eks-v2";
import { CfnAccessEntry } from "aws-cdk-lib/aws-eks";
import { DynamoMockerCpuStack } from "../lib/dynamo-mocker-cpu-stack";
import { McsFleet, McsPeeringsStack, McsRoutesStack } from "../lib/mcs-fleet";
import { DeploymentConfig } from "../lib/config";

const app = new App();
const configPath = app.node.tryGetContext("fleetConfigPath");
if (typeof configPath !== "string" || !configPath.endsWith(".local.json"))
  throw new Error("Select a private fleetConfigPath ending in .local.json");
const config = JSON.parse(readFileSync(configPath, "utf8")) as {
  prefix: string;
  accountId: string;
  sshSourceCidr: string;
  deployments: Record<string, DeploymentConfig>;
  mcs?: McsFleet;
};
if (
  !/^[a-z](?:[a-z0-9-]{0,14}[a-z0-9])?$/.test(config.prefix) ||
  !/^\d{12}$/.test(config.accountId)
)
  throw new Error(
    "Use a DNS-safe fleet prefix and a twelve-digit AWS account ID",
  );
for (const deployment of Object.values(config.deployments)) {
  if (
    !deployment.clusterConfig.clusterName.startsWith(config.prefix + "-") ||
    !deployment.clusterConfig.bastionConfig.iamRoleName.startsWith(
      config.prefix + "-",
    )
  )
    throw new Error(
      "Every cluster and admin role must belong to the selected fleet prefix",
    );
}
const member = app.node.tryGetContext("member") as string;
const c = config.deployments[member];
if (!c || !c.clusterConfig.clusterName.startsWith(config.prefix + "-"))
  throw new Error("Choose a configured prefixed member");
const mode = app.node.tryGetContext("mode");
const region = c.clusterConfig.region;
const bucket = `${config.prefix}-global-router-assets-${config.accountId}-${region}`;
const base = {
  env: { account: config.accountId, region },
  terminationProtection: true,
};
const credentials = new CliCredentialsStackSynthesizer({
  fileAssetsBucketName: bucket,
});
if (mode === "assets") {
  const stack = new Stack(app, "Assets", {
    ...base,
    stackName: `${config.prefix}-global-router-${member}-assets`,
    synthesizer: new LegacyStackSynthesizer(),
  });
  new s3.Bucket(stack, "Assets", {
    bucketName: bucket,
    blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
    enforceSSL: true,
    encryption: s3.BucketEncryption.S3_MANAGED,
    versioned: true,
    removalPolicy: RemovalPolicy.RETAIN,
  });
} else if (mode === "foundation") {
  const stack = new DynamoMockerCpuStack(app, "CpuFleet", {
    ...base,
    stackName: `${c.clusterConfig.clusterName}-cdk`,
    synthesizer: credentials,
    config: c,
  });
  const cluster = stack.node.findChild("Cluster") as eks.Cluster;
  const bastion = stack.node.findChild("Bastion");
  const role = bastion.node.findChild("Role") as iam.Role;
  role.addManagedPolicy(
    iam.ManagedPolicy.fromAwsManagedPolicyName("AmazonSSMManagedInstanceCore"),
  );
  role.addToPolicy(
    new iam.PolicyStatement({
      actions: ["eks:DescribeCluster"],
      resources: Object.values(config.deployments).map(
        (v) =>
          `arn:aws:eks:${v.clusterConfig.region}:${config.accountId}:cluster/${v.clusterConfig.clusterName}`,
      ),
    }),
  );
  // Narrow only this newly constructed bastion; existing security groups are never imported.
  const securityGroup = (
    bastion.node.findChild("SecurityGroup") as ec2.SecurityGroup
  ).node.defaultChild as ec2.CfnSecurityGroup;
  securityGroup.securityGroupIngress = [];
  const host = (bastion.node.findChild("Instance") as ec2.Instance).node
    .defaultChild as ec2.CfnInstance;
  host.addPropertyDeletionOverride("KeyName");
  host.addPropertyOverride(
    "LaunchTemplate.LaunchTemplateName",
    `${config.prefix}-global-router-${member}-admin`,
  );
  Aspects.of(stack).add({
    visit(child) {
      if (
        child instanceof CfnResource &&
        child.cfnResourceType === "AWS::EC2::LaunchTemplate" &&
        child.node.path.includes("Bastion")
      )
        child.addPropertyOverride(
          "LaunchTemplateName",
          `${config.prefix}-global-router-${member}-admin`,
        );
    },
  });
  // SSM needs no static public address or SSH output.
  bastion.node.tryRemoveChild("ElasticIp");
  for (const id of [
    "BastionPublicIp",
    "BastionKeyPairName",
    "ConnectToBastion",
  ])
    stack.node.tryRemoveChild(id);
  new CfnOutput(stack, "KubectlServiceToken", {
    value: cluster.kubectlProvider!.serviceToken,
  });
  new CfnOutput(stack, "ClusterSecurityGroup", {
    value: cluster.clusterSecurityGroupId,
  });
  new CfnOutput(stack, "VpcId", { value: stack.vpc.vpcId });
  new CfnOutput(stack, "PrivateRouteTables", {
    value: stack.privateRouteTableIds.join(","),
  });
  Tags.of(stack).add("Fleet", config.prefix);
} else if (mode === "admin") {
  const ohio = config.deployments.ohio;
  if (!ohio)
    throw new Error(
      "Temporary fleet administration requires a configured Ohio hub",
    );
  const stack = new Stack(app, "FleetAdmin", {
    ...base,
    stackName: `${config.prefix}-global-router-${member}-admin`,
    synthesizer: credentials,
  });
  new CfnAccessEntry(stack, "FleetAdminAccess", {
    clusterName: c.clusterConfig.clusterName,
    principalArn: `arn:aws:iam::${config.accountId}:role/${ohio.clusterConfig.bastionConfig.iamRoleName}`,
    accessPolicies: [
      {
        policyArn:
          "arn:aws:eks::aws:cluster-access-policy/AmazonEKSClusterAdminPolicy",
        accessScope: { type: "cluster" },
      },
    ],
  });
} else if (mode === "peerings" || mode === "routes") {
  if (!config.mcs)
    throw new Error("Collect new fleet metadata before networking");
  const m = config.mcs.members.find((v) => v.id === member);
  if (!m || !m.clusterName.startsWith(config.prefix + "-"))
    throw new Error("Invalid prefixed MCS member");
  const props = {
    ...base,
    stackName: `${config.prefix}-global-router-${member}-${mode}`,
    synthesizer: credentials,
    fleet: config.mcs,
    member: m,
  };
  if (mode === "peerings") new McsPeeringsStack(app, "Peerings", props);
  else new McsRoutesStack(app, "Routes", props);
} else throw new Error("Choose mode=assets/foundation/admin/peerings/routes");
