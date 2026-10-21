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

import { App } from "aws-cdk-lib";

import { POC_PEERING_OHIO, POC_PEERING_WEST, selectDeployment } from "../lib/const";
import type { DeploymentConfig } from "../lib/config";
import { loadGlobalRouterDeployment } from "../lib/global-router-deployment";
import { DynamoMockerCpuStack } from "../lib/dynamo-mocker-cpu-stack";
import { addInterRegionPeering } from "../lib/inter-region-peering";
import { withKarmadaControlPlane } from "../lib/multi-cluster";

const app = new App();
const deploymentName = app.node.tryGetContext("deployment") ?? "ohio";
const globalRouter = loadGlobalRouterDeployment(
  app.node.tryGetContext("routerImage"),
  app.node.tryGetContext("routerConfigPath"),
);
const installKarmada = contextFlag("karmada");
if (deploymentName === "both") {
  if (globalRouter !== undefined) {
    throw new Error("Deploy the Global Router with -c deployment=ohio after peering.");
  }
  if (installKarmada) {
    throw new Error("Install Karmada with -c deployment=ohio.");
  }
  // Existing regional stacks are deployed separately. This mode changes only
  // the connection and routes between their VPCs.
  addInterRegionPeering(
    app,
    POC_PEERING_OHIO,
    POC_PEERING_WEST,
    process.env.CDK_DEFAULT_ACCOUNT,
  );
} else {
  if (globalRouter !== undefined && deploymentName !== "ohio") {
    throw new Error("The Global Router POC workload is only supported in Ohio.");
  }
  createStack(
    "DynamoMockerCpu",
    withKarmadaControlPlane(selectDeployment(deploymentName), installKarmada),
    globalRouter,
  );
}

/** `-c <name>=true` or `false`; unset means false. */
function contextFlag(name: string): boolean {
  const value: unknown = app.node.tryGetContext(name);
  if (value === undefined || value === false || value === "false") {
    return false;
  }
  if (value === true || value === "true") {
    return true;
  }
  throw new Error(`-c ${name} must be true or false.`);
}

function createStack(
  id: string,
  config: DeploymentConfig,
  routerDeployment?: ReturnType<typeof loadGlobalRouterDeployment>,
): DynamoMockerCpuStack {
  return new DynamoMockerCpuStack(app, id, {
    config,
    globalRouter: routerDeployment,
    description:
      "Single-subnet NVIDIA Dynamo CPU Mocker pools on Amazon EKS",
    env: {
      account: process.env.CDK_DEFAULT_ACCOUNT,
      region: config.clusterConfig.region,
    },
    stackName: `${config.clusterConfig.clusterName}-cdk`,
  });
}
