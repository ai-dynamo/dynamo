// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import { KubectlV35Layer } from "@aws-cdk/lambda-layer-kubectl-v35";
import { CfnOutput, RemovalPolicy, Stack, StackProps, aws_ec2 as ec2 } from "aws-cdk-lib";
import * as eks from "aws-cdk-lib/aws-eks-v2";
import { Construct } from "constructs";
import { dnsName, positive } from "../config";
import { isValidVpcCidr } from "./cidr";

export interface FoundationConfig {
  readonly name: string;
  readonly region: string;
  readonly vpcCidr: string;
  readonly availabilityZones: readonly [string, string];
  readonly systemInstanceType: string;
  readonly systemNodeCount: number;
}

/** New infrastructure only. Existing clusters are consumed through ClusterTarget. */
export class RegionalFoundationStack extends Stack {
  constructor(scope: Construct, id: string, props: StackProps & { config: FoundationConfig }) {
    super(scope, id, { ...props, terminationProtection: true });
    const c = props.config;
    dnsName(c.name, "foundation name"); positive(c.systemNodeCount, "systemNodeCount");
    if (this.region !== c.region || !isValidVpcCidr(c.vpcCidr) || c.availabilityZones.length !== 2 ||
        new Set(c.availabilityZones).size !== 2 || c.availabilityZones.some(az => !az.startsWith(c.region))) {
      throw new Error("Foundation requires a regional VPC and two distinct regional AZs");
    }
    const vpc = new ec2.Vpc(this, "Vpc", {
      vpcName: c.name, ipAddresses: ec2.IpAddresses.cidr(c.vpcCidr),
      availabilityZones: [...c.availabilityZones], natGateways: 1,
      subnetConfiguration: [
        { name: "public", subnetType: ec2.SubnetType.PUBLIC, cidrMask: 24 },
        { name: "private", subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS, cidrMask: 20 },
      ],
    });
    const cluster = new eks.Cluster(this, "Cluster", {
      clusterName: c.name, version: eks.KubernetesVersion.V1_35, vpc,
      vpcSubnets: [{ subnets: vpc.privateSubnets }], defaultCapacity: 0,
      defaultCapacityType: eks.DefaultCapacityType.NODEGROUP,
      bootstrapClusterCreatorAdminPermissions: false, endpointAccess: eks.EndpointAccess.PRIVATE,
      albController: { version: eks.AlbControllerVersion.V2_14_1 },
      kubectlProviderOptions: { kubectlLayer: new KubectlV35Layer(this, "KubectlLayer"), privateSubnets: vpc.privateSubnets },
    });
    const system = cluster.addNodegroupCapacity("SystemNodes", {
      instanceTypes: [new ec2.InstanceType(c.systemInstanceType)],
      amiType: eks.NodegroupAmiType.AL2023_X86_64_STANDARD,
      minSize: c.systemNodeCount, desiredSize: c.systemNodeCount, maxSize: c.systemNodeCount,
      labels: { "dynamo.nvidia.com/compute-group": "system" },
      subnets: { subnets: vpc.privateSubnets },
    });
    cluster.applyRemovalPolicy(RemovalPolicy.RETAIN);
    system.applyRemovalPolicy(RemovalPolicy.RETAIN);
    const values = {
      ClusterName: cluster.clusterName, Region: this.region, VpcId: vpc.vpcId,
      ClusterSecurityGroupId: cluster.clusterSecurityGroupId,
      KubectlServiceToken: cluster.kubectlProvider!.serviceToken,
      OidcProviderArn: cluster.openIdConnectProvider.openIdConnectProviderArn,
      PrivateSubnetIds: vpc.privateSubnets.map(s => s.subnetId).join(","),
      PrivateRouteTableIds: vpc.privateSubnets.map(s => s.routeTable.routeTableId).join(","),
    };
    for (const [key, value] of Object.entries(values)) new CfnOutput(this, key, { value });
  }
}
