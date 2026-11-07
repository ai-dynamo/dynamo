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
  CfnOutput,
  CfnResource,
  RemovalPolicy,
  Stack,
  StackProps,
  Tags,
  aws_ec2 as ec2,
  aws_iam as iam,
  aws_s3 as s3,
} from "aws-cdk-lib";
import * as eks from "aws-cdk-lib/aws-eks-v2";
import { CfnNodegroup } from "aws-cdk-lib/aws-eks";
import { KubectlV35Layer } from "@aws-cdk/lambda-layer-kubectl-v35";
import { Construct } from "constructs";

export interface GpuRegion {
  readonly region: string;
  readonly availabilityZone: string;
  readonly reservationId: string;
  readonly instanceType: string;
  readonly count: number;
  readonly hourlyRate: number;
  readonly releaseVersion: string;
  readonly clusterName: string;
  readonly subnetId?: string;
  readonly foundation?: {
    readonly cidr: string;
    readonly availabilityZones: string[];
  };
}

export function validateGpuFleet(regions: GpuRegion[]): void {
  if (!Array.isArray(regions) || !regions.length || regions.length > 4)
    throw new Error("Specify one to four GPU regions");
  const seen = new Set<string>();
  let total = 0;
  for (const r of regions) {
    if (!r || seen.has(r.region) || !/^[a-z]{2}-[a-z]+-\d$/.test(r.region))
      throw new Error("Regions must be valid and unique");
    seen.add(r.region);
    if (
      !/^cr-[a-f0-9]{17}$/.test(r.reservationId) ||
      !r.availabilityZone.startsWith(r.region) ||
      !/^[a-z]{2}-[a-z]+-\d[a-z]$/.test(r.availabilityZone)
    )
      throw new Error("Specify reservation ID and matching regional AZ");
    if (!/^(g5|g6|g6e)\.(2|4)xlarge$/.test(r.instanceType))
      throw new Error("Unsupported POC GPU instance type");
    if (!Number.isInteger(r.count) || r.count < 2 || r.count > 4)
      throw new Error("GPU count must be two to four per region");
    if (!Number.isFinite(r.hourlyRate) || r.hourlyRate <= 0 || r.hourlyRate > 7)
      throw new Error("GPU hourly rate must be positive and at most $7");
    if (
      !/^1\.35\.\d+-\d{8}$/.test(r.releaseVersion) ||
      !/^[A-Za-z0-9][A-Za-z0-9_-]{0,99}$/.test(r.clusterName)
    )
      throw new Error("Specify EKS 1.35 NVIDIA release and cluster name");
    if (r.subnetId && !/^subnet-[a-f0-9]{17}$/.test(r.subnetId))
      throw new Error("Invalid GPU subnet ID");
    if (!r.subnetId && !r.foundation)
      throw new Error("Specify an existing subnet or a new foundation");
    if (
      r.foundation &&
      (r.foundation.availabilityZones.length !== 2 ||
        new Set(r.foundation.availabilityZones).size !== 2 ||
        !r.foundation.availabilityZones.includes(r.availabilityZone) ||
        r.foundation.availabilityZones.some(
          (az) => !new RegExp(`^${r.region}[a-z]$`).test(az),
        ))
    )
      throw new Error(
        "Foundation requires two distinct regional AZs including reserved AZ",
      );
    total += r.count;
  }
  if (total > 16) throw new Error("GPU fleet must not exceed 16 nodes");
}

/** Asset storage only; deployment uses the current operator session without deployment roles. */
export class GpuAssetsStack extends Stack {
  constructor(scope: Construct, id: string, props: StackProps) {
    super(scope, id, { ...props, terminationProtection: true });
    new s3.Bucket(this, "Assets", {
      bucketName: `global-router-cdk-assets-${this.account}-${this.region}`,
      blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
      enforceSSL: true,
      encryption: s3.BucketEncryption.S3_MANAGED,
      versioned: true,
      removalPolicy: RemovalPolicy.RETAIN,
    });
  }
}

/** Stable cluster/network lifecycle. Never contains application Helm releases. */
export class GpuFoundationStack extends Stack {
  constructor(
    scope: Construct,
    id: string,
    props: StackProps & { config: GpuRegion },
  ) {
    super(scope, id, { ...props, terminationProtection: true });
    const c = props.config;
    if (!c.foundation) throw new Error("Foundation settings required");
    const vpc = new ec2.Vpc(this, "Vpc", {
      ipAddresses: ec2.IpAddresses.cidr(c.foundation.cidr),
      availabilityZones: c.foundation.availabilityZones,
      natGateways: 1,
      subnetConfiguration: [
        { name: "public", subnetType: ec2.SubnetType.PUBLIC, cidrMask: 24 },
        {
          name: "private",
          subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS,
          cidrMask: 20,
        },
      ],
    });
    const cluster = new eks.Cluster(this, "Cluster", {
      clusterName: c.clusterName,
      version: eks.KubernetesVersion.V1_35,
      bootstrapClusterCreatorAdminPermissions: false,
      defaultCapacity: 0,
      defaultCapacityType: eks.DefaultCapacityType.NODEGROUP,
      endpointAccess: eks.EndpointAccess.PRIVATE,
      vpc,
      vpcSubnets: [{ subnets: vpc.privateSubnets }],
      kubectlProviderOptions: {
        kubectlLayer: new KubectlV35Layer(this, "KubectlLayer"),
        privateSubnets: vpc.privateSubnets,
      },
    });
    cluster.addHelmChart("NvidiaDevicePlugin", {
      chart: "nvidia-device-plugin",
      repository: "https://nvidia.github.io/k8s-device-plugin",
      version: "0.17.1",
      release: "nvidia-device-plugin",
      namespace: "kube-system",
      values: {
        affinity: null,
        nodeSelector: { "dynamo.nvidia.com/purpose": "gpu" },
      },
      wait: false,
    });
    const subnet = vpc.privateSubnets.find(
      (s) => s.availabilityZone === c.availabilityZone,
    );
    if (!subnet) throw new Error("No private subnet in reservation AZ");
    new CfnOutput(this, "GpuSubnetId", { value: subnet.subnetId });
    new CfnOutput(this, "ClusterName", { value: cluster.clusterName });
    new CfnOutput(this, "VpcId", { value: vpc.vpcId });
    // Keep cluster dependencies together on removal; a failed first creation may roll back.
    for (const child of this.node.findAll()) {
      if (child instanceof CfnResource)
        child.applyRemovalPolicy(RemovalPolicy.RETAIN_ON_UPDATE_OR_DELETE);
    }
    Tags.of(this).add("Project", "global-router-gpu");
  }
}

/** Independently deployed GPU nodes targeting externally owned reservations. */
export class ReservedGpuComputeStack extends Stack {
  constructor(
    scope: Construct,
    id: string,
    props: StackProps & { config: GpuRegion },
  ) {
    super(scope, id, { ...props, terminationProtection: true });
    const c = props.config;
    if (!c.subnetId)
      throw new Error(
        "Compute requires a preflight-verified private subnet in reservation AZ",
      );
    const role = new iam.Role(this, "NodeRole", {
      assumedBy: new iam.ServicePrincipal("ec2.amazonaws.com"),
      managedPolicies: [
        "AmazonEKSWorkerNodePolicy",
        "AmazonEC2ContainerRegistryPullOnly",
        "AmazonEKS_CNI_Policy",
        "AmazonSSMManagedInstanceCore",
      ].map((name) => iam.ManagedPolicy.fromAwsManagedPolicyName(name)),
    });
    role.applyRemovalPolicy(RemovalPolicy.RETAIN);
    const template = new ec2.CfnLaunchTemplate(this, "GpuLaunchTemplate", {
      launchTemplateData: {
        instanceType: c.instanceType,
        capacityReservationSpecification: {
          capacityReservationTarget: { capacityReservationId: c.reservationId },
        },
        metadataOptions: { httpTokens: "required", httpPutResponseHopLimit: 1 },
        blockDeviceMappings: [
          {
            deviceName: "/dev/xvda",
            ebs: {
              encrypted: true,
              volumeSize: 200,
              volumeType: "gp3",
              deleteOnTermination: true,
            },
          },
        ],
        tagSpecifications: [
          {
            resourceType: "instance",
            tags: [
              { key: "Project", value: "global-router-gpu" },
              { key: "Name", value: `${c.clusterName}-reserved-gpu` },
            ],
          },
        ],
      },
    });
    template.applyRemovalPolicy(RemovalPolicy.RETAIN);
    const nodes = new CfnNodegroup(this, "GpuNodes", {
      clusterName: c.clusterName,
      nodegroupName: "global-router-reserved-gpu",
      nodeRole: role.roleArn,
      subnets: [c.subnetId],
      capacityType: "ON_DEMAND",
      amiType: "AL2023_x86_64_NVIDIA",
      version: "1.35",
      releaseVersion: c.releaseVersion,
      launchTemplate: {
        id: template.ref,
        version: template.attrLatestVersionNumber,
      },
      scalingConfig: {
        minSize: c.count,
        desiredSize: c.count,
        maxSize: c.count,
      },
      updateConfig: { maxUnavailable: 1, updateStrategy: "MINIMAL" },
      labels: { "dynamo.nvidia.com/purpose": "gpu" },
    });
    nodes.applyRemovalPolicy(RemovalPolicy.RETAIN);
    new CfnOutput(this, "NodegroupName", { value: nodes.ref });
    new CfnOutput(this, "NodeRoleArn", { value: role.roleArn });
    Tags.of(this).add("Project", "global-router-gpu");
  }
}
