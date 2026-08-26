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

export interface DeploymentConfig {
  readonly clusterConfig: ClusterConfig;
  readonly dynamoPlatformConfig: DynamoPlatformConfig;
  readonly dgdPools: readonly [DgdPoolConfig, ...DgdPoolConfig[]];
  readonly dgds: readonly [DgdConfig, ...DgdConfig[]];
}
