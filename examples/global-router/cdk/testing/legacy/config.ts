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

import type { ClusterConfig, DynamoPlatformConfig, ComputeGroupConfig, PeerNetworkingConfig } from "../../lib/foundation/config";
export * from "../../lib/foundation/config";
export type { ComputeGroupConfig as DgdPoolConfig, ComputeSubnetSlot as DgdPoolSubnetSlot } from "../../lib/foundation/config";

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
  readonly dgdPools: readonly [ComputeGroupConfig, ...ComputeGroupConfig[]];
  readonly dgds: readonly [DgdConfig, ...DgdConfig[]];
  /** Peer pod ingress and source-address preservation for private routing. */
  readonly peerNetworking?: PeerNetworkingConfig;
}
