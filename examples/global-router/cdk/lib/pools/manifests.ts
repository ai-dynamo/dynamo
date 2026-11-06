// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import { PoolConfig, dnsName, pinnedImage, positive, httpUrl } from "../config";
import { isValidIpv4Cidr } from "../foundation/cidr";

export function poolManifests(config: PoolConfig): { dgd: Record<string, any>; relay: Record<string, any>; endpoint?: Record<string, any> } {
  dnsName(config.name, "pool name"); dnsName(config.namespace, "namespace");
  dnsName(config.frontendServiceName, "frontendServiceName");
  // Leave room for the operator-generated -pool-relay Service name.
  if (config.name.length > 52) throw new Error("Pool name must be at most 52 characters");
  if (!config.clusterId || /[\s/]/.test(config.clusterId)) throw new Error("clusterId must be nonempty and contain no whitespace or slash");
  pinnedImage(config.relayImage); positive(config.relayReplicas, "relayReplicas");
  httpUrl(config.discoveryUrl, "discoveryUrl");
  if (config.discoveryMode !== "registrations") {
    throw new Error("Pool discovery must use registrations backed by DynamoDB");
  }
  if (config.frontendEndpoint) {
    if (config.frontendUrls?.length) throw new Error("Choose frontendUrls or a managed frontendEndpoint");
    if (!config.frontendEndpoint.sourceCidrs.length || config.frontendEndpoint.sourceCidrs.some(c => !isValidIpv4Cidr(c))) {
      throw new Error("frontendEndpoint requires explicit valid source CIDRs");
    }
  } else if (!config.frontendUrls?.length) {
    throw new Error("Registrations mode needs frontendUrls or a managed frontendEndpoint reachable from all Global Routers");
  }
  for (const url of config.frontendUrls ?? []) httpUrl(url, "frontendUrl");
  if (!Array.isArray(config.dgdSpec?.components) || config.dgdSpec.components.length === 0) {
    throw new Error("Supply a DGD spec with components");
  }
  const metadata = { name: config.name, namespace: config.namespace };
  const env = [
    { name: "DYN_RELAY_GLOBAL_DISCOVERY_URL", value: config.discoveryUrl },
    { name: "DYN_RELAY_ROUTING_POOL", value: `${config.clusterId}/${config.namespace}/${config.name}` },
    { name: "DYN_RELAY_ROUTING_POOL_REGION", value: config.cluster.region },
    ...(config.frontendUrls?.length ? [{ name: "DYN_RELAY_ROUTING_POOL_FRONTENDS", value: config.frontendUrls.join(",") }] : []),
  ];
  return {
    ...(config.frontendEndpoint ? { endpoint: {
      apiVersion: "v1", kind: "Service", metadata: { name: `${config.name}-private`, namespace: config.namespace,
        annotations: {
          "service.beta.kubernetes.io/aws-load-balancer-type": "external",
          "service.beta.kubernetes.io/aws-load-balancer-nlb-target-type": "ip",
          "service.beta.kubernetes.io/aws-load-balancer-scheme": "internal",
          // A pool can have one frontend replica while the NLB spans several AZs.
          "service.beta.kubernetes.io/aws-load-balancer-attributes": "load_balancing.cross_zone.enabled=true",
        } },
      spec: { type: "LoadBalancer", loadBalancerSourceRanges: [...config.frontendEndpoint.sourceCidrs],
        selector: { "nvidia.com/dynamo-graph-deployment-name": config.name, "nvidia.com/dynamo-component-type": "frontend" },
        ports: [{ name: "http", protocol: "TCP", port: 8000, targetPort: "http" }],
      },
    } } : {}),
    dgd: { apiVersion: "nvidia.com/v1beta1", kind: "DynamoGraphDeployment", metadata, spec: structuredClone(config.dgdSpec) },
    relay: { apiVersion: "nvidia.com/v1alpha1", kind: "DynamoPoolRelay", metadata,
      spec: { clusterId: config.clusterId, replicas: config.relayReplicas, podTemplate: { spec: {
        containers: [{ name: "main", image: config.relayImage, env,
          resources: { requests: { cpu: "100m", memory: "256Mi" }, limits: { memory: "1Gi" } } }],
      } } } },
  };
}
