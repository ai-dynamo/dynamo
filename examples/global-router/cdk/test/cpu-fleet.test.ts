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

import { execFileSync } from "node:child_process";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import * as path from "node:path";
import * as assert from "node:assert/strict";
import { test } from "node:test";
import { VALID_CONFIG } from "./fixtures";

test("isolated CPU fleets bind admin instances to their own IMDS template and omit SSH/EIPs/GPUs", () => {
  const directory = mkdtempSync(path.join(tmpdir(), "cpu-fleet-contract-"));
  const root = path.resolve(__dirname, "../..");
  const region = "us-west-2";
  const vpc = {
    ...VALID_CONFIG.clusterConfig.vpcConfig,
    natInstanceType: "t3.small",
    natInstanceAmiId: "ami-00000000000000000",
  };
  const deployment = {
    ...VALID_CONFIG,
    clusterConfig: {
      ...VALID_CONFIG.clusterConfig,
      clusterName: "example-west",
      vpcConfig: vpc,
      bastionConfig: {
        ...VALID_CONFIG.clusterConfig.bastionConfig,
        iamRoleName: "example-west-bastion-role",
        amiId: "ami-00000000000000000",
        vpcConfig: vpc,
      },
    },
  };
  const config = path.join(directory, "fleet.local.json");
  writeFileSync(
    config,
    JSON.stringify({
      prefix: "example",
      accountId: "111111111111",
      deployments: { west: deployment },
    }),
  );
  try {
    execFileSync(
      process.execPath,
      ["--require", "ts-node/register", "bin/cpu-fleet.ts"],
      {
        cwd: root,
        env: {
          ...process.env,
          CDK_DEFAULT_ACCOUNT: "111111111111",
          CDK_DEFAULT_REGION: region,
          CDK_OUTDIR: path.join(directory, "out"),
          CDK_CONTEXT_JSON: JSON.stringify({
            fleetConfigPath: config,
            member: "west",
            mode: "foundation",
          }),
        },
        stdio: "pipe",
      },
    );
    const template = JSON.parse(
      readFileSync(path.join(directory, "out/CpuFleet.template.json"), "utf8"),
    );
    const resources = template.Resources as Record<
      string,
      { Type: string; Properties: any }
    >;
    const admin = Object.entries(resources).find(
      ([name, r]) =>
        name.startsWith("BastionInstance") && r.Type === "AWS::EC2::Instance",
    )![1];
    const specification = admin.Properties.LaunchTemplate;
    assert.equal(
      specification.LaunchTemplateName,
      "example-global-router-west-admin",
    );
    assert.equal(
      resources[specification.Version["Fn::GetAtt"][0]].Properties
        .LaunchTemplateName,
      specification.LaunchTemplateName,
    );
    assert.equal(admin.Properties.KeyName, undefined);
    assert.ok(
      !Object.values(resources).some((r) =>
        ["AWS::EC2::EIP", "AWS::EC2::NatGateway"].includes(r.Type),
      ),
    );
    const group = Object.entries(resources).find(
      ([name, r]) =>
        name.startsWith("BastionSecurityGroup") &&
        r.Type === "AWS::EC2::SecurityGroup",
    )![1];
    assert.deepEqual(group.Properties.SecurityGroupIngress ?? [], []);
    const node = Object.values(resources).find(
      (r) => r.Type === "AWS::EKS::Nodegroup",
    )!;
    assert.ok(
      node.Properties.InstanceTypes.every(
        (type: string) => !type.startsWith("g") && !type.startsWith("p"),
      ),
    );
  } finally {
    rmSync(directory, { recursive: true, force: true });
  }
});
