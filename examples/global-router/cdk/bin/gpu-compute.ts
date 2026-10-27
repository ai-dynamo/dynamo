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

import { readFileSync } from "node:fs";
import {
  App,
  LegacyStackSynthesizer,
  CliCredentialsStackSynthesizer,
} from "aws-cdk-lib";
import {
  GpuAssetsStack,
  GpuFoundationStack,
  GpuRegion,
  ReservedGpuComputeStack,
  validateGpuFleet,
} from "../lib/gpu-compute";

const app = new App();
const file = app.node.tryGetContext("gpuConfigPath");
if (typeof file !== "string" || !file.endsWith(".local.json"))
  throw new Error(
    "Select an ignored operator file with -c gpuConfigPath=config/gpu-capacity.local.json",
  );
const regions: GpuRegion[] = JSON.parse(readFileSync(file, "utf8")).regions;
validateGpuFleet(regions);
const selected = regions.find(
  (r) => r.region === app.node.tryGetContext("gpuRegion"),
);
if (!selected) throw new Error("Choose a configured gpuRegion");
const mode = app.node.tryGetContext("gpuMode");
const props = {
  config: selected,
  env: { account: process.env.CDK_DEFAULT_ACCOUNT, region: selected.region },
  stackName: `${selected.clusterName}-gpu-${mode}`,
  synthesizer:
    mode === "assets"
      ? new LegacyStackSynthesizer()
      : new CliCredentialsStackSynthesizer({
          fileAssetsBucketName: `global-router-cdk-assets-${process.env.CDK_DEFAULT_ACCOUNT}-${selected.region}`,
        }),
};
if (mode === "assets") new GpuAssetsStack(app, "GpuAssets", props);
else if (mode === "foundation")
  new GpuFoundationStack(app, "GpuFoundation", props);
else if (mode === "compute")
  new ReservedGpuComputeStack(app, "GpuCompute", props);
else
  throw new Error(
    "Choose gpuMode=assets, gpuMode=foundation or gpuMode=compute",
  );
