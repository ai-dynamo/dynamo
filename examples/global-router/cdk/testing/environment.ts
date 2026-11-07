// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import { App, Stack } from "aws-cdk-lib";
import { GlobalLayerConfig, GlobalStoresConfig, PoolConfig, dnsName } from "../lib/config";
import { GlobalStoresStack } from "../lib/global/stores";
import { GlobalLayerStack } from "../lib/global/stack";
import { PoolStack } from "../lib/pools/stack";
import { MockerWorkload, mockerDgdSpec } from "./workloads/mocker";

export interface TestEnvironment {
  readonly account: string;
  readonly name: string;
  readonly stores: GlobalStoresConfig;
  readonly regions: readonly {
    readonly id: string;
    readonly global: Omit<GlobalLayerConfig, "stores" | "affinityScope">;
    readonly pools: readonly (Omit<PoolConfig, "cluster" | "dgdSpec"> & { workload: MockerWorkload })[];
  }[];
}

/** Select one lifecycle at a time; changing a workload never synthesizes its foundation. */
export function testEnvironmentStack(app: App, config: TestEnvironment, layer: string, member?: string, poolName?: string): Stack {
  dnsName(config.name, "test environment name");
  if (!/^\d{12}$/.test(config.account)) throw new Error("Test environment requires an AWS account");
  if (config.name.length > 30) throw new Error("Test environment name must be at most 30 characters");
  const ids = config.regions.map(r => r.id);
  if (!ids.length || new Set(ids).size !== ids.length) throw new Error("Test members must have unique IDs");
  ids.forEach(id => dnsName(id, "test member"));
  if (layer === "stores") return new GlobalStoresStack(app, "Stores", {
    config: config.stores, stackName: `${config.name}-stores`, env: { account: config.account, region: config.stores.homeRegion },
  });
  const selected = config.regions.find(r => r.id === member);
  if (!selected) throw new Error("Select a configured -c member");
  const env = { account: config.account, region: selected.global.cluster.region };
  if (layer === "global") return new GlobalLayerStack(app, "Global", {
    config: { ...selected.global, stores: config.stores, affinityScope: config.name },
    stackName: `${config.name}-${member}-global`, env,
  });
  if (layer !== "pool") throw new Error("Test layer must be stores/global/pool");
  const names = selected.pools.map(p => p.name);
  if (new Set(names).size !== names.length) throw new Error("Test pool names must be unique per member");
  const pool = selected.pools.find(p => p.name === poolName);
  if (!pool) throw new Error("Select a configured -c pool");
  const { workload, ...settings } = pool;
  return new PoolStack(app, "Pool", {
    config: { ...settings, cluster: selected.global.cluster, dgdSpec: mockerDgdSpec(workload) },
    stackName: `${config.name}-${member}-${pool.name}`, env,
  });
}
