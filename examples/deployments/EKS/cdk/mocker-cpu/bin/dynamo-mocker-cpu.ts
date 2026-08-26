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

import { US_EAST2_DEPLOYMENT } from "../lib/const";
import { DynamoMockerCpuStack } from "../lib/dynamo-mocker-cpu-stack";

const app = new App();
const config = US_EAST2_DEPLOYMENT;

new DynamoMockerCpuStack(app, "DynamoMockerCpu", {
  config,
  description:
    "Single-subnet NVIDIA Dynamo CPU Mocker pools on Amazon EKS",
  env: {
    account: process.env.CDK_DEFAULT_ACCOUNT,
    region: config.clusterConfig.region,
  },
  stackName: `${config.clusterConfig.clusterName}-cdk`,
});
