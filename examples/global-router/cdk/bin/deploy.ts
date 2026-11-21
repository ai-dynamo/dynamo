// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import { readFileSync } from "node:fs";
import { App, CliCredentialsStackSynthesizer } from "aws-cdk-lib";
import { RegionalFoundationStack } from "../lib/foundation/stack";
import { ClusterServicesStack } from "../lib/foundation/services";
import { GlobalStoresStack } from "../lib/global/stores";
import { GlobalLayerStack } from "../lib/global/stack";
import { PoolStack } from "../lib/pools/stack";
import { ObservabilityStack } from "../lib/observability/stack";

const app = new App();
const file: unknown = app.node.tryGetContext("configPath");
const layer: unknown = app.node.tryGetContext("layer");
if (typeof file !== "string" || !file) throw new Error("Select -c configPath=<layer-config.json>");
const input = JSON.parse(readFileSync(file, "utf8"));
if (!/^\d{12}$/.test(input.account) || !/^[A-Za-z][A-Za-z0-9-]{0,127}$/.test(input.stackName) || !input.config) {
  throw new Error("Layer config needs account, stackName and config");
}
const config = input.config;
const region = layer === "stores" ? config.homeRegion : layer === "foundation" ? config.region : config.cluster?.region;
if (typeof region !== "string") throw new Error("Selected layer requires a region");
if (input.fileAssetsBucketName !== undefined &&
    (typeof input.fileAssetsBucketName !== "string" || !/^[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]$/.test(input.fileAssetsBucketName))) {
  throw new Error("fileAssetsBucketName must name the existing regional CDK asset bucket");
}
const props = { config, stackName: input.stackName, env: { account: input.account, region },
  ...(input.fileAssetsBucketName ? { synthesizer: new CliCredentialsStackSynthesizer({ fileAssetsBucketName: input.fileAssetsBucketName }) } : {}),
};
switch (layer) {
  case "foundation": new RegionalFoundationStack(app, "Foundation", props); break;
  case "services": new ClusterServicesStack(app, "Services", props); break;
  case "stores": new GlobalStoresStack(app, "Stores", props); break;
  case "global": new GlobalLayerStack(app, "Global", props); break;
  case "observability": new ObservabilityStack(app, "Observability", props); break;
  case "pool": new PoolStack(app, "Pool", props); break;
  default: throw new Error("Choose layer=foundation/services/stores/global/pool/observability");
}
