// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import { CfnResource, Duration, RemovalPolicy, Stack, StackProps } from "aws-cdk-lib";
import * as eks from "aws-cdk-lib/aws-eks-v2";
import { Construct } from "constructs";
import { ClusterTarget, dnsName } from "../config";
import { importCluster } from "./target";

export interface ClusterServicesConfig {
  readonly cluster: ClusterTarget;
  readonly operatorNamespace: string;
  readonly dynamoVersion?: string;
  readonly manageOperator?: boolean;
  readonly workloadNamespaces: readonly string[];
}

/** Install once per cluster; pool and global stacks never install an operator or own namespaces. */
export class ClusterServicesStack extends Stack {
  constructor(scope: Construct, id: string, props: StackProps & { config: ClusterServicesConfig }) {
    super(scope, id, { ...props, terminationProtection: true });
    const c = props.config;
    dnsName(c.operatorNamespace, "operatorNamespace");
    const manageOperator = c.manageOperator !== false;
    if (manageOperator && (!c.dynamoVersion || !/^[0-9][A-Za-z0-9.+-]*$/.test(c.dynamoVersion))) throw new Error("Pin a Dynamo platform chart version");
    const cluster = importCluster(this, c.cluster);
    const names = [...new Set([...(manageOperator ? [c.operatorNamespace] : []), ...c.workloadNamespaces])];
    if (!names.length) throw new Error("Configure at least one namespace");
    names.forEach(name => dnsName(name, "namespace"));
    const namespaces = new eks.KubernetesManifest(this, "Namespaces", { cluster,
      manifest: names.map(name => ({ apiVersion: "v1", kind: "Namespace", metadata: { name } })),
      overwrite: false, prune: false,
    });
    for (const child of namespaces.node.findAll()) {
      if (child instanceof CfnResource) child.applyRemovalPolicy(RemovalPolicy.RETAIN);
    }
    if (!manageOperator) return;
    const platform = new eks.HelmChart(this, "DynamoPlatform", {
      cluster, chart: `https://helm.ngc.nvidia.com/nvidia/ai-dynamo/charts/dynamo-platform-${c.dynamoVersion}.tgz`,
      namespace: c.operatorNamespace, release: "dynamo-platform", createNamespace: false,
      atomic: true, wait: true, timeout: Duration.minutes(15),
      values: { global: { etcd: { install: false }, nats: { install: false } } },
    });
    platform.node.addDependency(namespaces);
  }
}
