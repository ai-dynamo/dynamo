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
  McsFleet,
  McsMemberStack,
  McsPeeringsStack,
  McsRoutesStack,
  validateMcsFleet,
} from "../lib/mcs-fleet";

const app = new App();
const file: unknown = app.node.tryGetContext("mcsConfigPath");
if (typeof file !== "string" || !file.endsWith(".local.json"))
  throw new Error("Select ignored mcsConfigPath=config/mcs-fleet.local.json");
const fleet: McsFleet = JSON.parse(readFileSync(file, "utf8"));
validateMcsFleet(fleet);
const member = fleet.members.find(
  (m) => m.id === app.node.tryGetContext("mcsMember"),
);
if (!member) throw new Error("Choose a configured mcsMember");
const mode = app.node.tryGetContext("mcsMode");
const props = {
  fleet,
  member,
  env: { account: process.env.CDK_DEFAULT_ACCOUNT, region: member.region },
  stackName: `global-router-mcs-${member.id}-${mode}`,
  synthesizer: new CliCredentialsStackSynthesizer({
    fileAssetsBucketName: `global-router-cdk-assets-${process.env.CDK_DEFAULT_ACCOUNT}-${member.region}`,
  }),
};
if (mode === "peerings") new McsPeeringsStack(app, "McsPeerings", props);
else if (mode === "routes") new McsRoutesStack(app, "McsRoutes", props);
else if (mode === "member") new McsMemberStack(app, "McsMember", props);
else throw new Error("Choose mcsMode=peerings, routes or member");
