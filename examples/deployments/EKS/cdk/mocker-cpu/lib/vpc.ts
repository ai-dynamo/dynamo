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

import { Tags, Validations, aws_ec2 as ec2 } from "aws-cdk-lib";
import { Construct } from "constructs";

import type { DgdPoolConfig, VpcConfig } from "./config";

export interface MockerVpcProps {
  readonly config: VpcConfig;
  readonly dgdPools: readonly DgdPoolConfig[];
}

export const MAX_DGD_POOL_SUBNET_SLOTS = 6;

export class MockerVpc extends Construct {
  public readonly clusterSubnets: ec2.SelectedSubnets;
  public readonly poolSubnets: ReadonlyMap<string, ec2.SelectedSubnets>;
  public readonly vpc: ec2.Vpc;

  constructor(scope: Construct, id: string, props: MockerVpcProps) {
    super(scope, id);

    const { config, dgdPools } = props;
    const vpcCidrMask = Number(config.cidrBlock.split("/")[1]);
    const subnetCidrMask = vpcCidrMask + 4;
    const highestSubnetSlot = Math.max(
      ...dgdPools.map((pool) => pool.subnetSlot),
    );
    const requiredSubnetCount =
      config.availabilityZones.length * (2 + highestSubnetSlot);
    const availableSubnetCount = 2 ** (subnetCidrMask - vpcCidrMask);
    if (requiredSubnetCount > availableSubnetCount) {
      throw new Error(
        `VpcConfig.cidrBlock supports at most ${
          availableSubnetCount / config.availabilityZones.length - 2
        } DGD pools with the current subnet layout.`,
      );
    }
    const subnetSlots = Array.from(
      { length: highestSubnetSlot },
      (_, index) => index + 1,
    );

    this.vpc = new ec2.Vpc(this, "Resource", {
      availabilityZones: [...config.availabilityZones],
      ipAddresses: ec2.IpAddresses.cidr(config.cidrBlock),
      natGateways: 1,
      natGatewaySubnets: {
        availabilityZones: [config.availabilityZones[0]],
        subnetGroupName: "public",
      },
      subnetConfiguration: [
        {
          cidrMask: subnetCidrMask,
          name: "public",
          subnetType: ec2.SubnetType.PUBLIC,
        },
        {
          cidrMask: subnetCidrMask,
          name: "cluster",
          subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS,
        },
        ...subnetSlots.map((subnetSlot) => ({
          cidrMask: subnetCidrMask,
          name: dgdPoolSubnetGroupName(subnetSlot),
          subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS,
        })),
      ],
      vpcName: config.name,
    });
    Validations.of(this.vpc).acknowledge({
      id: "CloudFormation-Validate::W3010",
      reason:
        "The typed pool config selects explicit AZs for single-subnet pool node groups.",
    });

    this.clusterSubnets = this.vpc.selectSubnets({
      subnetGroupName: "cluster",
    });

    if (this.clusterSubnets.subnets.length !== 2) {
      throw new Error("Expected two EKS cluster subnets in distinct zones.");
    }

    const poolSubnets = new Map<string, ec2.SelectedSubnets>();
    for (const pool of dgdPools) {
      const selectedSubnets = this.vpc.selectSubnets({
        availabilityZones: [pool.availabilityZone],
        subnetGroupName: dgdPoolSubnetGroupName(pool.subnetSlot),
      });
      if (selectedSubnets.subnets.length !== 1) {
        throw new Error(
          `Expected one subnet for pool ${pool.name} in ${pool.availabilityZone}.`,
        );
      }
      for (const subnet of selectedSubnets.subnets) {
        Tags.of(subnet).add("dynamo.nvidia.com/dgd-pool", pool.name);
      }
      poolSubnets.set(pool.name, selectedSubnets);
    }
    this.poolSubnets = poolSubnets;
  }

  public subnetsForPool(poolName: string): ec2.SelectedSubnets {
    const subnets = this.poolSubnets.get(poolName);
    if (subnets === undefined) {
      throw new Error(`No subnet exists for pool ${poolName}.`);
    }
    return subnets;
  }
}

export function dgdPoolSubnetGroupName(subnetSlot: number): string {
  return `dgd-pool-slot-${subnetSlot}`;
}
