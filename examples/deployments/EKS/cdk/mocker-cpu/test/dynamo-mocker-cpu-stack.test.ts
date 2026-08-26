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

import type {
  DeploymentConfig,
  DgdConfig,
  DgdPoolConfig,
} from "../lib/config";
import { DynamoMockerCpuStack } from "../lib/dynamo-mocker-cpu-stack";
import {
  DGD_POOL_LABEL_KEY,
  PURPOSE_LABEL_KEY,
  PURPOSE_LABEL_VALUE,
} from "../lib/helm-values";
import { MULTI_POOL_CONFIG, VALID_CONFIG } from "./fixtures";

interface SynthesizedResource {
  readonly CreationPolicy?: {
    readonly ResourceSignal?: {
      readonly Count?: number;
      readonly Timeout?: string;
    };
  };
  readonly DependsOn?: readonly string[];
  readonly Properties: Record<string, unknown>;
}

function synthesizeTemplate(config: DeploymentConfig = VALID_CONFIG): Template {
  const app = new App();
  const stack = new DynamoMockerCpuStack(app, "TestStack", {
    config,
    env: {
      account: "111111111111",
      region: config.clusterConfig.region,
    },
  });
  return Template.fromStack(stack);
}

function findNodeGroupForPool(
  template: Template,
  poolName: string,
): [string, SynthesizedResource] {
  const nodeGroups = template.findResources(
    "AWS::EKS::Nodegroup",
  ) as Record<string, SynthesizedResource>;
  const entry = Object.entries(nodeGroups).find(
    ([, nodeGroup]) =>
      (nodeGroup.Properties.Labels as Record<string, string>)[
        DGD_POOL_LABEL_KEY
      ] === poolName,
  );
  assert.ok(entry);
  return entry;
}

function selectedPoolSubnet(
  template: Template,
  poolName: string,
): [string, SynthesizedResource] {
  const [, nodeGroup] = findNodeGroupForPool(template, poolName);
  const subnetIds = nodeGroup.Properties.Subnets as Array<{ Ref: string }>;
  assert.equal(subnetIds.length, 1);
  const subnetLogicalId = subnetIds[0].Ref;
  const subnets = template.findResources("AWS::EC2::Subnet") as Record<
    string,
    SynthesizedResource
  >;
  return [subnetLogicalId, subnets[subnetLogicalId]];
}

function assertPrivateNatRoutedSubnet(
  template: Template,
  subnetLogicalId: string,
): void {
  const subnets = template.findResources("AWS::EC2::Subnet") as Record<
    string,
    SynthesizedResource
  >;
  assert.equal(subnets[subnetLogicalId].Properties.MapPublicIpOnLaunch, false);

  const associations = template.findResources(
    "AWS::EC2::SubnetRouteTableAssociation",
  ) as Record<string, SynthesizedResource>;
  const association = Object.values(associations).find(
    (candidate) =>
      JSON.stringify(candidate.Properties.SubnetId) ===
      JSON.stringify({ Ref: subnetLogicalId }),
  );
  assert.ok(association);

  const routes = template.findResources("AWS::EC2::Route") as Record<
    string,
    SynthesizedResource
  >;
  const natGatewayLogicalIds = new Set(
    Object.keys(template.findResources("AWS::EC2::NatGateway")),
  );
  const natRoute = Object.values(routes).find(
    (candidate) => {
      const natGatewayId = candidate.Properties.NatGatewayId as
        | { Ref: string }
        | undefined;
      return (
        JSON.stringify(candidate.Properties.RouteTableId) ===
        JSON.stringify(association.Properties.RouteTableId) &&
        candidate.Properties.DestinationCidrBlock === "0.0.0.0/0" &&
        natGatewayId !== undefined &&
        natGatewayLogicalIds.has(natGatewayId.Ref)
      );
    },
  );
  assert.ok(natRoute);
}

function findChartForRelease(
  charts: Record<string, SynthesizedResource>,
  release: string,
): [string, SynthesizedResource] {
  const entry = Object.entries(charts).find(
    ([, chart]) => chart.Properties.Release === release,
  );
  assert.ok(entry);
  return entry;
}

function expectedPoolAffinity(poolNames: readonly string[]): unknown {
  return {
    requiredDuringSchedulingIgnoredDuringExecution: {
      nodeSelectorTerms: [
        {
          matchExpressions: [
            {
              key: DGD_POOL_LABEL_KEY,
              operator: "In",
              values: [...poolNames].sort((left, right) =>
                left < right ? -1 : left > right ? 1 : 0,
              ),
            },
          ],
        },
      ],
    },
  };
}

test("uses two AZs for EKS and one subnet for each managed node group", () => {
  const template = synthesizeTemplate();
  template.resourceCountIs("AWS::EC2::NatGateway", 1);
  template.resourceCountIs("AWS::EC2::Subnet", 6);
  template.hasResourceProperties("AWS::EKS::Cluster", {
    AccessConfig: {
      AuthenticationMode: "API",
      BootstrapClusterCreatorAdminPermissions: false,
    },
    ResourcesVpcConfig: {
      EndpointPrivateAccess: true,
      EndpointPublicAccess: false,
    },
    Version: "1.35",
  });

  const clusterResources = template.findResources("AWS::EKS::Cluster");
  const cluster = Object.values(clusterResources)[0] as SynthesizedResource;
  const vpcConfig = cluster.Properties.ResourcesVpcConfig as {
    SubnetIds: unknown[];
    PublicAccessCidrs?: unknown[];
  };
  assert.equal(vpcConfig.SubnetIds.length, 2);
  assert.equal(vpcConfig.PublicAccessCidrs, undefined);

  const nodeGroupResources = template.findResources("AWS::EKS::Nodegroup");
  const nodeGroup = Object.values(nodeGroupResources)[0] as SynthesizedResource;
  const nodeGroupSubnets = nodeGroup.Properties.Subnets as unknown[];
  assert.equal(nodeGroupSubnets.length, 1);
  const poolSubnetId = nodeGroupSubnets[0] as { Ref: string };
  const subnets = template.findResources("AWS::EC2::Subnet") as Record<
    string,
    SynthesizedResource
  >;
  assert.equal(subnets[poolSubnetId.Ref].Properties.AvailabilityZone, "us-west-2a");
  assertPrivateNatRoutedSubnet(template, poolSubnetId.Ref);
  assert.ok(
    (
      subnets[poolSubnetId.Ref].Properties.Tags as Array<{
        Key: string;
        Value: string;
      }>
    ).some(
      (tag) =>
        tag.Key === DGD_POOL_LABEL_KEY && tag.Value === "cpu-a",
    ),
  );
  assert.deepEqual(nodeGroup.Properties.Labels, {
    [DGD_POOL_LABEL_KEY]: "cpu-a",
    [PURPOSE_LABEL_KEY]: PURPOSE_LABEL_VALUE,
  });
  assert.deepEqual(nodeGroup.Properties.ScalingConfig, {
    DesiredSize: 2,
    MaxSize: 4,
    MinSize: 1,
  });
  assert.equal(nodeGroup.Properties.AmiType, "AL2023_x86_64_STANDARD");
  assert.deepEqual(nodeGroup.Properties.NodeRepairConfig, { Enabled: true });
});

test("uses a public SSH bastion with the configured EC2 key pair", () => {
  const template = synthesizeTemplate();
  const instances = template.findResources("AWS::EC2::Instance") as Record<
    string,
    SynthesizedResource
  >;
  assert.equal(Object.keys(instances).length, 1);
  const [instanceLogicalId, instance] = Object.entries(instances)[0];

  assert.equal(instance.Properties.AvailabilityZone, "us-west-2a");
  assert.equal(instance.Properties.InstanceType, "t3.small");
  assert.equal(instance.Properties.KeyName, "test-admin-key");
  const imageParameter = (instance.Properties.ImageId as { Ref: string }).Ref;
  const parameters = template.toJSON().Parameters as Record<
    string,
    { Default?: string }
  >;
  assert.match(parameters[imageParameter].Default ?? "", /x86_64$/);
  const networkInterfaces = instance.Properties.NetworkInterfaces as Array<{
    AssociatePublicIpAddress: boolean;
    SubnetId: { Ref: string };
  }>;
  assert.equal(networkInterfaces.length, 1);
  assert.equal(networkInterfaces[0].AssociatePublicIpAddress, true);
  assert.match(JSON.stringify(networkInterfaces[0].SubnetId), /publicSubnet1/);

  assert.deepEqual(instance.Properties.UserData, {
    "Fn::Base64": "#!/bin/bash",
  });
  assert.equal(instance.CreationPolicy, undefined);

  const launchTemplateReference = instance.Properties.LaunchTemplate as {
    Version: { "Fn::GetAtt": [string, string] };
  };
  const launchTemplates = template.findResources(
    "AWS::EC2::LaunchTemplate",
  ) as Record<string, SynthesizedResource>;
  const bastionLaunchTemplate =
    launchTemplates[launchTemplateReference.Version["Fn::GetAtt"][0]];
  assert.deepEqual(
    (bastionLaunchTemplate.Properties.LaunchTemplateData as {
      MetadataOptions: Record<string, string>;
    }).MetadataOptions,
    { HttpTokens: "required" },
  );

  const subnets = template.findResources("AWS::EC2::Subnet") as Record<
    string,
    SynthesizedResource
  >;
  assert.equal(
    subnets[networkInterfaces[0].SubnetId.Ref].Properties.MapPublicIpOnLaunch,
    true,
  );

  const roles = template.findResources("AWS::IAM::Role") as Record<
    string,
    SynthesizedResource
  >;
  const bastionRoleEntry = Object.entries(roles).find(
    ([, role]) =>
      role.Properties.RoleName === "dynamo-mocker-bastion-test",
  );
  assert.ok(bastionRoleEntry);
  assert.equal(bastionRoleEntry[1].Properties.ManagedPolicyArns, undefined);

  const policies = template.findResources("AWS::IAM::Policy") as Record<
    string,
    SynthesizedResource
  >;
  const bastionPolicy = Object.values(policies).find((policy) => {
    const properties = JSON.stringify(policy.Properties);
    return (
      properties.includes("eks:DescribeCluster") &&
      properties.includes(bastionRoleEntry[0])
    );
  });
  assert.ok(bastionPolicy);
  const clusterResources = template.findResources(
    "AWS::EKS::Cluster",
  ) as Record<string, SynthesizedResource>;
  const clusterLogicalId = Object.keys(clusterResources)[0];
  assert.deepEqual(bastionPolicy.Properties, {
    PolicyDocument: {
      Statement: [
        {
          Action: "eks:DescribeCluster",
          Effect: "Allow",
          Resource: { "Fn::GetAtt": [clusterLogicalId, "Arn"] },
        },
      ],
      Version: "2012-10-17",
    },
    PolicyName: bastionPolicy.Properties.PolicyName,
    Roles: [{ Ref: bastionRoleEntry[0] }],
  });

  const allIamActions = Object.values(policies).flatMap((policy) => {
    const document = policy.Properties.PolicyDocument as {
      Statement: Array<{ Action: string | string[] }>;
    };
    return document.Statement.flatMap((statement) => statement.Action);
  });
  assert.equal(
    allIamActions.some((action) =>
      /^(?:ssm|ssmmessages|ec2messages):|^ec2:DescribeKeyPairs$/i.test(action),
    ),
    false,
  );

  const accessEntries = template.findResources(
    "AWS::EKS::AccessEntry",
  ) as Record<string, SynthesizedResource>;
  const bastionAccessEntry = Object.entries(accessEntries).find(
    ([, entry]) => {
      const properties = JSON.stringify(entry.Properties);
      return (
        properties.includes("AmazonEKSClusterAdminPolicy") &&
        properties.includes(bastionRoleEntry[0])
      );
    },
  );
  assert.ok(bastionAccessEntry);
  assert.match(
    JSON.stringify(bastionAccessEntry[1].Properties.PrincipalArn),
    new RegExp(bastionRoleEntry[0]),
  );
  assert.ok(instance.DependsOn?.includes(bastionAccessEntry[0]));

  const securityGroups = template.findResources(
    "AWS::EC2::SecurityGroup",
  ) as Record<string, SynthesizedResource>;
  const bastionSecurityGroup = Object.entries(securityGroups).find(
    ([, group]) =>
      group.Properties.GroupDescription ===
      "Public SSH access for the Dynamo bastion",
  );
  assert.ok(bastionSecurityGroup);
  assert.deepEqual(bastionSecurityGroup[1].Properties.SecurityGroupIngress, [
    {
      CidrIp: "0.0.0.0/0",
      Description: "SSH with the EC2 launch key pair",
      FromPort: 22,
      IpProtocol: "tcp",
      ToPort: 22,
    },
  ]);

  const publicIngress: Array<{
    readonly groupId: unknown;
    readonly rule: Record<string, unknown>;
  }> = [];
  for (const [groupLogicalId, group] of Object.entries(securityGroups)) {
    const rules = (group.Properties.SecurityGroupIngress ?? []) as Array<
      Record<string, unknown>
    >;
    for (const rule of rules) {
      if (rule.CidrIp === "0.0.0.0/0" || rule.CidrIpv6 === "::/0") {
        publicIngress.push({ groupId: groupLogicalId, rule });
      }
    }
  }

  const ingressRules = template.findResources(
    "AWS::EC2::SecurityGroupIngress",
  ) as Record<string, SynthesizedResource>;
  for (const rule of Object.values(ingressRules)) {
    if (
      rule.Properties.CidrIp === "0.0.0.0/0" ||
      rule.Properties.CidrIpv6 === "::/0"
    ) {
      publicIngress.push({
        groupId: rule.Properties.GroupId,
        rule: rule.Properties,
      });
    }
  }
  assert.deepEqual(publicIngress, [
    {
      groupId: bastionSecurityGroup[0],
      rule: {
        CidrIp: "0.0.0.0/0",
        Description: "SSH with the EC2 launch key pair",
        FromPort: 22,
        IpProtocol: "tcp",
        ToPort: 22,
      },
    },
  ]);

  const bastionSource = {
    "Fn::GetAtt": [bastionSecurityGroup[0], "GroupId"],
  };
  const eksApiRules = Object.values(ingressRules).filter(
    (rule) =>
      JSON.stringify(rule.Properties.SourceSecurityGroupId) ===
      JSON.stringify(bastionSource),
  );
  assert.equal(eksApiRules.length, 2);
  const expectedEksDestinations = [
    (
      clusterResources[clusterLogicalId].Properties.ResourcesVpcConfig as {
        SecurityGroupIds: unknown[];
      }
    ).SecurityGroupIds[0],
    { "Fn::GetAtt": [clusterLogicalId, "ClusterSecurityGroupId"] },
  ];
  assert.deepEqual(
    eksApiRules
      .map((rule) => {
        assert.equal(
          rule.Properties.Description,
          "Bastion access to the private EKS API",
        );
        assert.equal(rule.Properties.FromPort, 443);
        assert.equal(rule.Properties.ToPort, 443);
        assert.equal(rule.Properties.IpProtocol, "tcp");
        assert.deepEqual(rule.Properties.SourceSecurityGroupId, bastionSource);
        return rule.Properties.GroupId;
      })
      .map((groupId) => JSON.stringify(groupId))
      .sort(),
    expectedEksDestinations
      .map((groupId) => JSON.stringify(groupId))
      .sort(),
  );

  const elasticIps = template.findResources("AWS::EC2::EIP") as Record<
    string,
    SynthesizedResource
  >;
  const bastionElasticIp = Object.entries(elasticIps).find(
    ([, elasticIp]) =>
      String(JSON.stringify(elasticIp.Properties.InstanceId)).includes(
        instanceLogicalId,
      ),
  );
  assert.ok(bastionElasticIp);
  assert.equal(bastionElasticIp[1].Properties.Domain, "vpc");
  assert.deepEqual(bastionElasticIp[1].Properties.InstanceId, {
    Ref: instanceLogicalId,
  });
  const internetGateways = template.findResources(
    "AWS::EC2::InternetGateway",
  );
  const internetGatewayLogicalId = Object.keys(internetGateways)[0];
  assert.ok(
    bastionElasticIp[1].DependsOn?.includes(internetGatewayLogicalId),
  );

  template.hasOutput("BastionInstanceId", {
    Value: { Ref: instanceLogicalId },
  });
  template.hasOutput("BastionPublicIp", {
    Value: { "Fn::GetAtt": [bastionElasticIp[0], "PublicIp"] },
  });
  template.hasOutput("BastionKeyPairName", {
    Value: "test-admin-key",
  });
});

test("places the CDK kubectl provider in egress-capable cluster subnets", () => {
  const template = synthesizeTemplate();
  const functions = template.findResources("AWS::Lambda::Function");
  const provider = Object.values(functions).find((resource) => {
    const candidate = resource as SynthesizedResource;
    return candidate.Properties.Description ===
      "onEvent handler for EKS kubectl resource provider";
  }) as SynthesizedResource | undefined;

  assert.ok(provider);
  const vpcConfig = provider.Properties.VpcConfig as { SubnetIds: unknown[] };
  assert.equal(vpcConfig.SubnetIds.length, 2);
  for (const subnetId of vpcConfig.SubnetIds) {
    assert.match(JSON.stringify(subnetId), /clusterSubnet/);
  }
});

test("installs the pool, platform, then DGD in dependency order", () => {
  const template = synthesizeTemplate();
  const charts = template.findResources(
    "Custom::AWSCDK-EKS-HelmChart",
  ) as Record<string, SynthesizedResource>;
  assert.equal(Object.keys(charts).length, 2);

  const platformEntry = findChartForRelease(charts, "dynamo-platform");
  const mockerEntry = findChartForRelease(charts, "mocker-a");

  assert.match(
    String(platformEntry[1].Properties.Chart),
    /dynamo-platform-1\.4\.0\.tgz$/,
  );
  const poolNodeGroup = findNodeGroupForPool(template, "cpu-a");
  assert.ok(platformEntry[1].DependsOn?.includes(poolNodeGroup[0]));
  assert.ok(mockerEntry[1].DependsOn?.includes(platformEntry[0]));
  assert.ok(mockerEntry[1].DependsOn?.includes(poolNodeGroup[0]));

  const platformValues = JSON.parse(
    String(platformEntry[1].Properties.Values),
  ) as {
    global: {
      etcd: { install: boolean };
      nats: { install: boolean };
    };
  };
  assert.equal(platformValues.global.etcd.install, false);
  assert.equal(platformValues.global.nats.install, false);

  const mockerValues = JSON.parse(String(mockerEntry[1].Properties.Values)) as {
    placement: {
      nodeAffinity: unknown;
      nodeSelector: Record<string, string>;
    };
  };
  assert.deepEqual(mockerValues.placement.nodeSelector, {
    [PURPOSE_LABEL_KEY]: PURPOSE_LABEL_VALUE,
  });
  assert.deepEqual(
    mockerValues.placement.nodeAffinity,
    expectedPoolAffinity(["cpu-a"]),
  );
});

test("creates independent same-AZ pools and maps DGDs through affinity", () => {
  const template = synthesizeTemplate(MULTI_POOL_CONFIG);
  template.resourceCountIs("AWS::EC2::NatGateway", 1);
  template.resourceCountIs("AWS::EC2::Subnet", 8);
  template.resourceCountIs("AWS::EKS::Nodegroup", 2);

  const subnets = template.findResources("AWS::EC2::Subnet") as Record<
    string,
    SynthesizedResource
  >;
  const nodeGroups = template.findResources(
    "AWS::EKS::Nodegroup",
  ) as Record<string, SynthesizedResource>;
  const charts = template.findResources(
    "Custom::AWSCDK-EKS-HelmChart",
  ) as Record<string, SynthesizedResource>;
  assert.equal(Object.keys(charts).length, 4);
  const platformEntry = findChartForRelease(charts, "dynamo-platform");

  const selectedSubnetRefs = new Map<string, string>();
  const nodeGroupLogicalIds = new Map<string, string>();
  for (const pool of MULTI_POOL_CONFIG.dgdPools) {
    const nodeGroupEntry = findNodeGroupForPool(template, pool.name);
    nodeGroupLogicalIds.set(pool.name, nodeGroupEntry[0]);
    assert.deepEqual(nodeGroupEntry[1].Properties.Labels, {
      [DGD_POOL_LABEL_KEY]: pool.name,
      [PURPOSE_LABEL_KEY]: PURPOSE_LABEL_VALUE,
    });
    assert.deepEqual(nodeGroupEntry[1].Properties.ScalingConfig, {
      DesiredSize: pool.nodeGroupConfig.scaling.desiredSize,
      MaxSize: pool.nodeGroupConfig.scaling.maxSize,
      MinSize: pool.nodeGroupConfig.scaling.minSize,
    });
    assert.deepEqual(
      nodeGroupEntry[1].Properties.InstanceTypes,
      pool.nodeGroupConfig.instanceTypes,
    );

    const nodeGroupSubnets = nodeGroupEntry[1].Properties.Subnets as Array<{
      Ref: string;
    }>;
    assert.equal(nodeGroupSubnets.length, 1);
    const subnetLogicalId = nodeGroupSubnets[0].Ref;
    selectedSubnetRefs.set(pool.name, subnetLogicalId);
    assertPrivateNatRoutedSubnet(template, subnetLogicalId);
    assert.equal(
      subnets[subnetLogicalId].Properties.AvailabilityZone,
      pool.availabilityZone,
    );
    assert.ok(
      (
        subnets[subnetLogicalId].Properties.Tags as Array<{
          Key: string;
          Value: string;
        }>
      ).some(
        (tag) => tag.Key === DGD_POOL_LABEL_KEY && tag.Value === pool.name,
      ),
    );
    assert.ok(platformEntry[1].DependsOn?.includes(nodeGroupEntry[0]));
  }

  for (const dgd of MULTI_POOL_CONFIG.dgds) {
    const chartEntry = findChartForRelease(charts, dgd.dgdName);
    assert.ok(chartEntry[1].DependsOn?.includes(platformEntry[0]));
    const expectedNodeGroupDependencies = dgd.poolNames
      .map((poolName) => nodeGroupLogicalIds.get(poolName))
      .filter((logicalId): logicalId is string => logicalId !== undefined)
      .sort();
    const actualNodeGroupDependencies = (chartEntry[1].DependsOn ?? [])
      .filter((logicalId) => nodeGroups[logicalId] !== undefined)
      .sort();
    assert.deepEqual(
      actualNodeGroupDependencies,
      expectedNodeGroupDependencies,
    );

    const values = JSON.parse(String(chartEntry[1].Properties.Values)) as {
      deployment: { name: string };
      image: string;
      model: { name: string };
      mocker: { replicas: number; speedup_ratio: number };
      placement: {
        nodeAffinity: unknown;
        nodeSelector: Record<string, string>;
      };
    };
    assert.equal(values.deployment.name, dgd.dgdName);
    assert.equal(
      values.image,
      "nvcr.io/nvidia/ai-dynamo/dynamo-planner:1.4.0",
    );
    assert.equal(values.model.name, dgd.modelName);
    assert.deepEqual(values.mocker, {
      replicas: dgd.mockerReplicas,
      speedup_ratio: dgd.mockerSpeedupRatio,
    });
    assert.deepEqual(values.placement.nodeSelector, {
      [PURPOSE_LABEL_KEY]: PURPOSE_LABEL_VALUE,
    });
    assert.deepEqual(
      values.placement.nodeAffinity,
      expectedPoolAffinity(dgd.poolNames),
    );
  }

  assert.equal(new Set(selectedSubnetRefs.values()).size, 2);
  template.hasOutput("DgdNames", {
    Value: "mocker-a,mocker-b,mocker-flex",
  });
  template.hasOutput("DgdPoolNames", {
    Value: "cpu-a,cpu-b",
  });
  template.hasOutput("DgdPoolPlacements", {
    Value: "cpu-a=us-west-2a,cpu-b=us-west-2a",
  });
  template.hasOutput("DgdPoolMappings", {
    Value: "mocker-a=cpu-a;mocker-b=cpu-b;mocker-flex=cpu-a+cpu-b",
  });
  const outputs = template.toJSON().Outputs as Record<
    string,
    { Description: string; Value: unknown }
  >;
  const subnetMappings = outputs.DgdPoolSubnetIds;
  assert.deepEqual(subnetMappings.Value, {
    "Fn::Join": [
      "",
      [
        "cpu-a=",
        { Ref: selectedSubnetRefs.get("cpu-a") },
        ",cpu-b=",
        { Ref: selectedSubnetRefs.get("cpu-b") },
      ],
    ],
  });
  template.hasOutput("ModelNames", {
    Value: "Qwen/Qwen3-0.6B,Qwen/Qwen3-1.7B,Qwen/Qwen3-4B",
  });
  template.hasOutput("PortForwardFrontends", {
    Value:
      "mocker-a=kubectl port-forward -n dynamo-system " +
      "svc/mocker-a-frontend 8000:8000; " +
      "mocker-b=kubectl port-forward -n dynamo-system " +
      "svc/mocker-b-frontend 8000:8000; " +
      "mocker-flex=kubectl port-forward -n dynamo-system " +
      "svc/mocker-flex-frontend 8000:8000",
  });
});

test("honors a pool that selects the second VPC Availability Zone", () => {
  const secondAzPool: DgdPoolConfig = {
    ...MULTI_POOL_CONFIG.dgdPools[1],
    availabilityZone: "us-west-2b",
  };
  const template = synthesizeTemplate({
    ...MULTI_POOL_CONFIG,
    dgdPools: [MULTI_POOL_CONFIG.dgdPools[0], secondAzPool],
  });
  const [subnetLogicalId, subnet] = selectedPoolSubnet(
    template,
    secondAzPool.name,
  );
  assert.equal(subnet.Properties.AvailabilityZone, "us-west-2b");
  assertPrivateNatRoutedSubnet(template, subnetLogicalId);

  const charts = template.findResources(
    "Custom::AWSCDK-EKS-HelmChart",
  ) as Record<string, SynthesizedResource>;
  const flexChart = findChartForRelease(charts, "mocker-flex")[1];
  const values = JSON.parse(String(flexChart.Properties.Values)) as {
    placement: {
      nodeAffinity: unknown;
      nodeSelector: Record<string, string>;
    };
  };
  assert.deepEqual(
    values.placement.nodeAffinity,
    expectedPoolAffinity(["cpu-a", "cpu-b"]),
  );
  assert.deepEqual(values.placement.nodeSelector, {
    [PURPOSE_LABEL_KEY]: PURPOSE_LABEL_VALUE,
  });
  template.hasOutput("DgdPoolPlacements", {
    Value: "cpu-a=us-west-2a,cpu-b=us-west-2b",
  });
});

test("keeps existing pool CIDRs stable when an earlier-named pool is added", () => {
  const existingPool: DgdPoolConfig = {
    ...VALID_CONFIG.dgdPools[0],
    subnetSlot: 2,
  };
  const baseTemplate = synthesizeTemplate({
    ...VALID_CONFIG,
    dgdPools: [existingPool],
  });
  baseTemplate.resourceCountIs("AWS::EC2::Subnet", 8);
  const earlierNamedPool: DgdPoolConfig = {
    ...MULTI_POOL_CONFIG.dgdPools[1],
    name: "aaa-pool",
    subnetSlot: 1,
  };
  const expandedTemplate = synthesizeTemplate({
    ...VALID_CONFIG,
    dgdPools: [existingPool, earlierNamedPool],
  });
  const [baseLogicalId, baseSubnet] = selectedPoolSubnet(
    baseTemplate,
    "cpu-a",
  );
  const [expandedLogicalId, expandedSubnet] = selectedPoolSubnet(
    expandedTemplate,
    "cpu-a",
  );

  assert.equal(expandedLogicalId, baseLogicalId);
  assert.equal(
    expandedSubnet.Properties.CidrBlock,
    baseSubnet.Properties.CidrBlock,
  );
});

test("adding a DGD workload leaves pool infrastructure unchanged", () => {
  const baseConfig: DeploymentConfig = {
    ...MULTI_POOL_CONFIG,
    dgds: [MULTI_POOL_CONFIG.dgds[0], MULTI_POOL_CONFIG.dgds[1]],
  };
  const baseTemplate = synthesizeTemplate(baseConfig);
  const expandedTemplate = synthesizeTemplate(MULTI_POOL_CONFIG);

  baseTemplate.resourceCountIs("Custom::AWSCDK-EKS-HelmChart", 3);
  expandedTemplate.resourceCountIs("Custom::AWSCDK-EKS-HelmChart", 4);
  for (const resourceType of [
    "AWS::EC2::Subnet",
    "AWS::EKS::Nodegroup",
    "AWS::EC2::LaunchTemplate",
  ]) {
    assert.deepEqual(
      expandedTemplate.findResources(resourceType),
      baseTemplate.findResources(resourceType),
    );
  }
});

test("keeps bastion instance configuration independent of pools", () => {
  const singlePoolInstances = synthesizeTemplate().findResources(
    "AWS::EC2::Instance",
  ) as Record<string, SynthesizedResource>;
  const multiPoolInstances = synthesizeTemplate(
    MULTI_POOL_CONFIG,
  ).findResources("AWS::EC2::Instance") as Record<
    string,
    SynthesizedResource
  >;
  const singlePoolInstance = Object.values(singlePoolInstances)[0];
  const multiPoolInstance = Object.values(multiPoolInstances)[0];

  assert.deepEqual(multiPoolInstance.Properties, singlePoolInstance.Properties);
});

test("keeps synthesized resources stable when pools and DGDs are reordered", () => {
  const reversedPools = [...MULTI_POOL_CONFIG.dgdPools].reverse() as [
    DgdPoolConfig,
    ...DgdPoolConfig[],
  ];
  const reversedDgds = [...MULTI_POOL_CONFIG.dgds]
    .reverse()
    .map((dgd) =>
      dgd.dgdName === "mocker-flex"
        ? {
            ...dgd,
            poolNames: [...dgd.poolNames].reverse() as [string, ...string[]],
          }
        : dgd,
    ) as [DgdConfig, ...DgdConfig[]];
  const reversedConfig: DeploymentConfig = {
    ...MULTI_POOL_CONFIG,
    dgdPools: reversedPools,
    dgds: reversedDgds,
  };

  assert.deepEqual(
    synthesizeTemplate(reversedConfig).toJSON(),
    synthesizeTemplate(MULTI_POOL_CONFIG).toJSON(),
  );
});

test("uses an encrypted gp3 root volume and IMDSv2", () => {
  const template = synthesizeTemplate();
  template.hasResourceProperties("AWS::EC2::LaunchTemplate", {
    LaunchTemplateData: {
      BlockDeviceMappings: [
        {
          DeviceName: "/dev/xvda",
          Ebs: {
            DeleteOnTermination: true,
            Encrypted: true,
            VolumeSize: 50,
            VolumeType: "gp3",
          },
        },
      ],
      MetadataOptions: {
        HttpTokens: "required",
      },
    },
  });
});
