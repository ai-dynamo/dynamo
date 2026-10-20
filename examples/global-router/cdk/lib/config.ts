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

export interface DgdConfig {
  readonly dgdName: string;
  /** Effective worker namespace from DGD status; changes on worker rollout. */
  readonly workerRuntimeNamespace?: string;
  /** Local worker selection policy; omitted keeps the Frontend image default. */
  readonly frontendRouterMode?: "round-robin" | "kv";
  readonly poolNames: readonly [string, ...string[]];
  readonly mockerReplicas: number;
  readonly modelName: string;
  readonly mockerSpeedupRatio: number;
}

export type DgdPoolSubnetSlot = 1 | 2 | 3 | 4 | 5 | 6;

export interface DgdPoolConfig {
  readonly name: string;
  readonly availabilityZone: string;
  readonly subnetSlot: DgdPoolSubnetSlot;
  readonly nodeGroupConfig: NodeGroupConfig;
}

export interface VpcConfig {
  readonly region: string;
  readonly availabilityZones: readonly [string, string];
  readonly name: string;
  readonly cidrBlock: string;
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

/** IDs from the already-deployed regional stacks used by peering-only mode. */
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

/** Karmada control plane, installed in the hub cluster only. */
export interface KarmadaControlPlaneConfig {
  /** Karmada release, for example `1.19.0`. Pins the chart and its images. */
  readonly version: string;
  /** VPC ranges allowed to reach the internal Karmada API load balancer. */
  readonly apiSourceCidrs: readonly [string, ...string[]];
}

/** Karmada and SIG Multicluster (MCS) support for one cluster. */
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

export interface DeploymentConfig {
  readonly clusterConfig: ClusterConfig;
  readonly mockerImage?: string;
  /** Dynamo runtime version for a Mocker image without a semantic-version tag. */
  readonly mockerRuntimeVersionOverride?: string;
  /** Region-local POC Relay image; when set, deploy one Relay per DGD. */
  readonly kvDcRelayImage?: string;
  readonly privateFrontendSourceCidrs: readonly [string, ...string[]];
  /** CIDRs of regional Router VPCs allowed to query private Relay endpoints. */
  readonly privateRelaySourceCidrs: readonly [string, ...string[]];
  readonly dynamoPlatformConfig: DynamoPlatformConfig;
  readonly dgdPools: readonly [DgdPoolConfig, ...DgdPoolConfig[]];
  readonly dgds: readonly [DgdConfig, ...DgdConfig[]];
  /** Karmada and MCS support; omit for a single-cluster deployment. */
  readonly multiCluster?: MultiClusterConfig;
}
