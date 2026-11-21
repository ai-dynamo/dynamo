// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import { ClusterTarget, GlobalStoresConfig } from "../config";
export interface ObservabilityConfig {
  readonly cluster: ClusterTarget;
  readonly fleet: string;
  readonly globalNamespace: string;
  readonly poolNamespace: string;
  readonly routerName: string;
  readonly stores: GlobalStoresConfig;
  readonly collectorImage: string;
  readonly addonVersion: string;
  readonly dashboardRegions: readonly { region: string; clusterName: string; pool: string }[];
  readonly monitoredServices: readonly { namespace: string; name: string; minimumPods: number }[];
  readonly globalDashboard?: boolean;
}
