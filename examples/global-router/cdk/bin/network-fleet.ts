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
import { App, CliCredentialsStackSynthesizer } from "aws-cdk-lib";
import {
  FleetConfig,
  FleetMemberStack,
  FleetPeeringsStack,
  FleetRoutesStack,
  validateFleet,
} from "../lib/foundation/fleet-network";

const app = new App();
const file: unknown = app.node.tryGetContext("networkConfigPath");
if (typeof file !== "string" || !file.endsWith(".local.json"))
  throw new Error("Select ignored networkConfigPath=config/network-fleet.local.json");
const fleet: FleetConfig = JSON.parse(readFileSync(file, "utf8"));
validateFleet(fleet);
const member = fleet.members.find(
  (m) => m.id === app.node.tryGetContext("networkMember"),
);
if (!member) throw new Error("Choose a configured networkMember");
const mode = app.node.tryGetContext("networkMode");
const props = {
  fleet,
  member,
  env: { account: process.env.CDK_DEFAULT_ACCOUNT, region: member.region },
  stackName: `global-router-network-${member.id}-${mode}`,
  synthesizer: new CliCredentialsStackSynthesizer({
    fileAssetsBucketName: `global-router-cdk-assets-${process.env.CDK_DEFAULT_ACCOUNT}-${member.region}`,
  }),
};
if (mode === "peerings") new FleetPeeringsStack(app, "FleetPeerings", props);
else if (mode === "routes") new FleetRoutesStack(app, "FleetRoutes", props);
else if (mode === "member") new FleetMemberStack(app, "FleetMember", props);
else throw new Error("Choose networkMode=peerings, routes or member");
