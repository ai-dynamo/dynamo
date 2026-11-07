// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import { CfnOutput, Duration, Stack, StackProps } from "aws-cdk-lib";
import * as eks from "aws-cdk-lib/aws-eks-v2";
import { Construct } from "constructs";
import { PoolConfig } from "../config";
import { importCluster } from "../foundation/target";
import { poolManifests } from "./manifests";

/** One independently replaceable DGD and its operator-managed PoolRelay. */
export class PoolStack extends Stack {
  constructor(scope: Construct, id: string, props: StackProps & { config: PoolConfig }) {
    super(scope, id, props);
    const manifests = poolManifests(props.config);
    const cluster = importCluster(this, props.config.cluster);
    const dgd = new eks.KubernetesManifest(this, "Dgd", { cluster, manifest: [manifests.dgd], overwrite: false });
    if (manifests.endpoint) {
      const endpoint = new eks.KubernetesManifest(this, "FrontendEndpoint", { cluster, manifest: [manifests.endpoint], overwrite: false });
      endpoint.node.addDependency(dgd);
      const address = new eks.KubernetesObjectValue(this, "FrontendAddress", {
        cluster, objectType: "service", objectName: manifests.endpoint.metadata.name,
        objectNamespace: props.config.namespace, jsonPath: ".status.loadBalancer.ingress[0].hostname",
        timeout: Duration.minutes(10),
      });
      address.node.addDependency(endpoint);
      const url = `http://${address.value}:8000`;
      manifests.relay.spec.podTemplate.spec.containers[0].env.push({ name: "DYN_RELAY_ROUTING_POOL_FRONTENDS", value: url });
      new CfnOutput(this, "FrontendUrl", { value: url });
    }
    const relay = new eks.KubernetesManifest(this, "Relay", { cluster, manifest: [manifests.relay], overwrite: false });
    relay.node.addDependency(dgd);
  }
}
