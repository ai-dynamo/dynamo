// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/** References to infrastructure owned outside an application stack. */
export interface ClusterTarget {
  readonly name: string;
  readonly region: string;
  readonly securityGroupId: string;
  readonly kubectlServiceToken: string;
  readonly oidcProviderArn: string;
}

export interface GlobalStoresConfig {
  /** Exactly one stack, in this region, owns both tables and all their replicas. */
  readonly homeRegion: string;
  readonly replicaRegions: readonly string[];
  readonly affinityTableName: string;
  readonly discoveryTableName: string;
}

export interface GlobalLayerConfig {
  readonly name: string;
  readonly namespace: string;
  readonly cluster: ClusterTarget;
  readonly frontendImage: string;
  readonly discoveryImage: string;
  readonly frontendReplicas: number;
  readonly discoveryReplicas: number;
  readonly stores: GlobalStoresConfig;
  readonly affinityScope: string;
  /** Total DynamoDB lookup/write budget, including cold AWS credential retrieval. */
  readonly affinityTimeoutMs?: number;
  /** Global discovery is exclusively the DynamoDB-backed registration directory. */
  readonly discoveryMode: "registrations";
  /** Primary pod address family; must match the target cluster. Defaults to IPv4. */
  readonly podAddressFamily?: "ipv4" | "ipv6";
  /** Static membership is rejected; router replicas register their own pod endpoints. */
  readonly routers?: never;
  readonly ingress?: {
    readonly sourceCidrs: readonly string[];
    readonly certificateArn?: string;
  };
}

export interface PoolConfig {
  readonly name: string;
  /** Must already exist; namespace ownership is independent of individual pools. */
  readonly namespace: string;
  readonly cluster: ClusterTarget;
  readonly clusterId: string;
  readonly discoveryUrl: string;
  /** A complete DGD spec, supplied by a serving recipe or a test workload. */
  readonly dgdSpec: Record<string, unknown>;
  readonly relayImage: string;
  readonly relayReplicas: number;
  readonly frontendServiceName: string;
  /** Required in registrations mode; each URL must be reachable from every router. */
  readonly frontendUrls?: readonly string[];
  /** Alternatively, provision a pool-owned internal NLB and register its resolved hostname. */
  readonly frontendEndpoint?: { readonly sourceCidrs: readonly string[] };
  readonly discoveryMode: GlobalLayerConfig["discoveryMode"];
}

export function dnsName(value: string, field: string): void {
  if (typeof value !== "string" || value.length > 63 || !/^[a-z0-9](?:[-a-z0-9]*[a-z0-9])?$/.test(value)) {
    throw new Error(`${field} must be a Kubernetes DNS label`);
  }
}

export function pinnedImage(value: string): void {
  if (typeof value !== "string" || /\s/.test(value) || value.endsWith(":latest") ||
      !(/@sha256:[a-fA-F0-9]{64}$/.test(value) || /(?:^|\/)[^/:]+:[A-Za-z0-9_][A-Za-z0-9_.-]*$/.test(value))) {
    throw new Error("Images must use an explicit version tag or SHA256 digest, not latest");
  }
}

export function positive(value: number, field: string): void {
  if (!Number.isSafeInteger(value) || value < 1) throw new Error(`${field} must be a positive integer`);
}

export function httpUrl(value: string, field: string): void {
  const url = new URL(value);
  if (!["http:", "https:"].includes(url.protocol) || !url.hostname || url.username || url.password) {
    throw new Error(`${field} must be an HTTP(S) URL without credentials`);
  }
}

export function validateStores(config: GlobalStoresConfig): void {
  const regions = [config.homeRegion, ...config.replicaRegions];
  if (regions.some(r => !/^[a-z]{2}(?:-[a-z]+)+-\d+$/.test(r)) || new Set(regions).size !== regions.length) {
    throw new Error("Store regions must be valid and unique, excluding homeRegion from replicaRegions");
  }
  for (const name of [config.affinityTableName, config.discoveryTableName]) {
    if (!/^[A-Za-z0-9_.-]{3,255}$/.test(name)) throw new Error("Invalid DynamoDB table name");
  }
  if (config.affinityTableName === config.discoveryTableName) throw new Error("Affinity and discovery require separate tables");
}
