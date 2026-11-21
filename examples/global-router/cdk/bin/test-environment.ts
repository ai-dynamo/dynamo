// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import { readFileSync } from "node:fs";
import { App } from "aws-cdk-lib";
import { testEnvironmentStack } from "../testing/environment";

const app = new App();
const file = app.node.tryGetContext("configPath");
if (typeof file !== "string" || !file) throw new Error("Select -c configPath=<test-environment.json>");
testEnvironmentStack(app, JSON.parse(readFileSync(file, "utf8")),
  app.node.tryGetContext("layer"), app.node.tryGetContext("member"), app.node.tryGetContext("pool"));
