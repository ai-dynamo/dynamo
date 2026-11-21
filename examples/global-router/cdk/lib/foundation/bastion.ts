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

import {
  Aspects,
  CfnResource,
  Tags,
  Validations,
  aws_ec2 as ec2,
  aws_iam as iam,
} from "aws-cdk-lib";
import * as eks from "aws-cdk-lib/aws-eks-v2";
import { Construct } from "constructs";

import type { ClusterConfig } from "./config";

export interface AdminAccess {
  readonly mode: "ssh" | "ssm";
  readonly launchTemplateName?: string;
  readonly describeClusterArns?: readonly string[];
}

export interface EksAdminBastionProps {
  readonly adminAccess?: AdminAccess;
  readonly cluster: eks.Cluster;
  readonly clusterConfig: ClusterConfig;
  readonly vpc: ec2.Vpc;
}

export class EksAdminBastion extends Construct {
  public readonly accessEntry: eks.AccessEntry;
  public readonly instance: ec2.Instance;
  public readonly publicIp: string;
  public readonly role: iam.Role;
  public readonly securityGroup: ec2.SecurityGroup;

  constructor(scope: Construct, id: string, props: EksAdminBastionProps) {
    super(scope, id);

    const { cluster, clusterConfig, vpc } = props;
    const { bastionConfig } = clusterConfig;
    const subnets = vpc.selectSubnets({
      availabilityZones: [bastionConfig.availabilityZone],
      subnetGroupName: "public",
    });
    if (subnets.subnets.length !== 1) {
      throw new Error(
        `Expected one public bastion subnet in ${bastionConfig.availabilityZone}.`,
      );
    }

    this.role = new iam.Role(this, "Role", {
      assumedBy: new iam.ServicePrincipal("ec2.amazonaws.com"),
      description: "Private EKS access for the Dynamo bastion",
      roleName: bastionConfig.iamRoleName,
    });
    this.role.addToPolicy(
      new iam.PolicyStatement({
        actions: ["eks:DescribeCluster"],
        resources: [cluster.clusterArn],
      }),
    );
    this.securityGroup = new ec2.SecurityGroup(this, "SecurityGroup", {
      allowAllOutbound: true,
      description: "Public SSH access for the Dynamo bastion",
      vpc,
    });
    if (props.adminAccess?.mode !== "ssm") this.securityGroup.addIngressRule(
      ec2.Peer.anyIpv4(),
      ec2.Port.tcp(22),
      "SSH with the EC2 launch key pair",
    );
    Validations.of(this.securityGroup).acknowledge({
      id: "CloudFormation-Validate::W2508",
      reason:
        "The user explicitly accepts public TCP 22; Amazon EC2 installs the configured launch key for ec2-user.",
    });

    const launchKeyPair = props.adminAccess?.mode === "ssm" ? undefined : ec2.KeyPair.fromKeyPairName(
      this,
      "LaunchKeyPair",
      bastionConfig.keyPairName,
    );

    this.instance = new ec2.Instance(this, "Instance", {
      associatePublicIpAddress: true,
      availabilityZone: bastionConfig.availabilityZone,
      blockDevices: [
        {
          deviceName: "/dev/xvda",
          volume: ec2.BlockDeviceVolume.ebs(20, {
            deleteOnTermination: true,
            encrypted: true,
            volumeType: ec2.EbsDeviceVolumeType.GP3,
          }),
        },
      ],
      instanceName: bastionConfig.name,
      instanceType: new ec2.InstanceType(bastionConfig.instanceType),
      keyPair: launchKeyPair,
      machineImage: bastionConfig.amiId
        ? ec2.MachineImage.genericLinux({
            [clusterConfig.region]: bastionConfig.amiId,
          })
        : ec2.MachineImage.latestAmazonLinux2023({
            cpuType: ec2.AmazonLinuxCpuType.X86_64,
          }),
      requireImdsv2: true,
      role: this.role,
      securityGroup: this.securityGroup,
      vpc,
      vpcSubnets: { subnets: subnets.subnets },
    });

    if (props.adminAccess?.mode !== "ssm") {
    const elasticIp = new ec2.CfnEIP(this, "ElasticIp", {
      domain: "vpc",
      instanceId: this.instance.instanceId,
      tags: [
        { key: "Name", value: `${bastionConfig.name}-public-ip` },
        { key: "Project", value: "dynamo-mocker-cpu" },
      ],
    });
    elasticIp.node.addDependency(vpc.internetConnectivityEstablished);
    this.publicIp = elasticIp.attrPublicIp;
    } else {
      this.publicIp = "";
      this.role.addManagedPolicy(iam.ManagedPolicy.fromAwsManagedPolicyName("AmazonSSMManagedInstanceCore"));
      if (props.adminAccess.describeClusterArns?.length) this.role.addToPolicy(new iam.PolicyStatement({
        actions: ["eks:DescribeCluster"], resources: [...props.adminAccess.describeClusterArns],
      }));
      // CDK creates the IMDS launch template during synthesis. Give it and the instance the same name.
      const name = props.adminAccess.launchTemplateName;
      if (name) {
        const host = this.instance.node.defaultChild as ec2.CfnInstance;
        host.addPropertyOverride("LaunchTemplate.LaunchTemplateName", name);
        Aspects.of(this).add({ visit(child) {
          if (child instanceof CfnResource && child.cfnResourceType === "AWS::EC2::LaunchTemplate") {
            child.addPropertyOverride("LaunchTemplateName", name);
          }
        } });
      }
    }

    this.accessEntry = cluster.grantClusterAdmin(
      "BastionClusterAdministrator",
      this.role.roleArn,
    );
    this.instance.node.addDependency(this.accessEntry);
    cluster.connections.allowFrom(
      this.instance,
      ec2.Port.tcp(443),
      "Bastion access to the private EKS API",
    );

    Tags.of(this.instance).add("Project", "dynamo-mocker-cpu");
  }
}
