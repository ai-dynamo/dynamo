// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import { CfnOutput, Stack, StackProps, aws_iam as iam } from "aws-cdk-lib";
import * as eks from "aws-cdk-lib/aws-eks-v2";
import { Construct } from "constructs";
import { GlobalLayerConfig } from "../config";
import { importCluster } from "../foundation/target";
import { globalManifests, validateGlobal } from "./manifests";

export class GlobalLayerStack extends Stack {
  constructor(scope: Construct, id: string, props: StackProps & { config: GlobalLayerConfig }) {
    super(scope, id, props);
    const config = props.config;
    validateGlobal(config);
    const cluster = importCluster(this, config.cluster);
    // Namespace is a foundation prerequisite, so deleting this application cannot delete pools.
    const accounts = [
      { suffix: "router", table: config.stores.affinityTableName, actions: ["dynamodb:GetItem", "dynamodb:PutItem"] },
      { suffix: "discovery", table: config.stores.discoveryTableName, actions: ["dynamodb:DescribeTable", "dynamodb:Query", "dynamodb:UpdateItem"] },
    ].map(({ suffix, table, actions }) => {
      const account = new eks.ServiceAccount(this, `${suffix}Identity`, {
        cluster, name: `${config.name}-${suffix}`, namespace: config.namespace,
      });
      account.addToPrincipalPolicy(new iam.PolicyStatement({
        actions, resources: [this.formatArn({ service: "dynamodb", resource: "table", resourceName: table })],
      }));
      return account;
    });
    const workloads = new eks.KubernetesManifest(this, "GlobalServices", {
      cluster, manifest: globalManifests(config), overwrite: false,
    });
    workloads.node.addDependency(...accounts);
    new CfnOutput(this, "DiscoveryService", { value: `${config.name}-discovery.${config.namespace}.svc:8080` });
    new CfnOutput(this, "RouterService", { value: `${config.name}-router.${config.namespace}.svc:8000` });
  }
}
