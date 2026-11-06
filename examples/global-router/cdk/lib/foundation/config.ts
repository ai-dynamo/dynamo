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

export interface NodeGroupScaling {
  readonly minSize: number;
  readonly desiredSize: number;
  readonly maxSize: number;
}

export interface NodeGroupConfig {
  readonly instanceTypes: readonly [string, ...string[]];
  readonly scaling: NodeGroupScaling;
}

export interface DynamoPlatformConfig {
  readonly dynamoVersion: string;
  readonly dynamoNamespace: string;
}

export type ComputeSubnetSlot = 1 | 2 | 3 | 4 | 5 | 6;

export interface ComputeGroupConfig {
  readonly name: string;
  readonly availabilityZone: string;
  readonly subnetSlot: ComputeSubnetSlot;
  readonly nodeGroupConfig: NodeGroupConfig;
}

export interface VpcConfig {
  readonly region: string;
  readonly availabilityZones: readonly [string, string];
  readonly name: string;
  readonly cidrBlock: string;
  /** Optional CPU lab egress without consuming Elastic IP quota. */
  readonly natInstanceType?: string;
  readonly natInstanceAmiId?: string;
}

export interface BastionConfig {
  readonly name: string;
  readonly instanceType: string;
  /** Pin deployed AMIs so unrelated stack updates cannot replace the bastion. */
  readonly amiId?: string;
  readonly iamRoleName: string;
  readonly keyPairName: string;
  readonly region: string;
  readonly availabilityZone: string;
  readonly vpcConfig: VpcConfig;
}

export interface ClusterConfig {
  readonly region: string;
  readonly clusterName: string;
  readonly vpcConfig: VpcConfig;
  readonly bastionConfig: BastionConfig;
}

export interface PeeringRegionConfig {
  readonly region: string;
  readonly cidrBlock: string;
  readonly vpcId: string;
  readonly privateRouteTableIds: readonly [string, ...string[]];
  /**
   * Public subnet route tables. Routing them over the peering lets each
   * bastion reach the other VPC, for example to join Karmada.
   */
  readonly publicRouteTableIds?: readonly string[];
}

export interface KarmadaControlPlaneConfig {
  /**
   * Karmada release, which pins the chart and its images. It must be
   * `KARMADA_VERSION`, the release that `crds/` comes from.
   */
  readonly version: string;
  /** VPC ranges allowed to reach the internal Karmada API load balancer. */
  readonly apiSourceCidrs: readonly [string, ...string[]];
}

export interface MultiClusterConfig {
  /** Karmada member name. The Global Router uses it as the pool site ID. */
  readonly memberName: string;
  /**
   * VPC ranges of the other member clusters. Their pods may reach pods in
   * this cluster, and traffic from pods here to them keeps its pod address.
   */
  readonly peerCidrs: readonly [string, ...string[]];
  /** Set in the hub cluster only. */
  readonly karmadaControlPlane?: KarmadaControlPlaneConfig;
}
