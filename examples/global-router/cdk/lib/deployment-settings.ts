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

import * as fs from "node:fs";
import * as path from "node:path";

import type { DeploymentConfig, PeeringRegionConfig } from "./config";
import { DEPLOYMENTS, POC_PEERING_OHIO, POC_PEERING_WEST } from "./const";

export interface DeploymentSettings {
  readonly deployments: Record<string, DeploymentConfig>;
  readonly peering: {
    readonly ohio: PeeringRegionConfig;
    readonly west: PeeringRegionConfig;
  };
}

/** Load a complete operator configuration only when explicitly selected. */
export function loadDeploymentSettings(
  configPathValue: unknown,
  cwd = process.cwd(),
): DeploymentSettings {
  if (configPathValue === undefined) {
    return {
      deployments: DEPLOYMENTS,
      peering: { ohio: POC_PEERING_OHIO, west: POC_PEERING_WEST },
    };
  }
  if (typeof configPathValue !== "string" || configPathValue.trim() === "") {
    throw new Error("-c deploymentConfigPath must name a JSON settings file.");
  }
  let settings: unknown;
  try {
    settings = JSON.parse(fs.readFileSync(path.resolve(cwd, configPathValue), "utf8"));
  } catch (error) {
    throw new Error("Could not read deployment settings JSON.", { cause: error });
  }
  if (!isObject(settings) || !isObject(settings.deployments)
      || Object.keys(settings.deployments).length === 0
      || !Object.values(settings.deployments).every(isObject)
      || !isObject(settings.peering)
      || !isObject(settings.peering.ohio) || !isObject(settings.peering.west)) {
    throw new Error("Deployment settings must contain deployments and peering.ohio/peering.west objects.");
  }
  // The stack validates deployment fields; peering fields are checked here
  // because those stacks do not pass through DynamoMockerCpuStack.
  for (const region of [settings.peering.ohio, settings.peering.west]) {
    if (typeof region.region !== "string" || typeof region.cidrBlock !== "string"
        || typeof region.vpcId !== "string" || !/^vpc-[0-9a-f]{8,17}$/.test(region.vpcId)
        || !Array.isArray(region.privateRouteTableIds) || region.privateRouteTableIds.length === 0
        || !region.privateRouteTableIds.every((id: unknown) =>
          typeof id === "string" && /^rtb-[0-9a-f]{8,17}$/.test(id))) {
      throw new Error("Each peering region must specify region, cidrBlock, vpcId and privateRouteTableIds.");
    }
  }
  return settings as unknown as DeploymentSettings;
}

function isObject(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}
