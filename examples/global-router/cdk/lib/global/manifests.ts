// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import { GlobalLayerConfig, dnsName, pinnedImage, positive, validateStores } from "../config";
import { isValidIpv4Cidr } from "../foundation/cidr";

export function validateGlobal(config: GlobalLayerConfig): void {
  dnsName(config.name, "global name"); dnsName(config.namespace, "namespace");
  if (config.name.length > 40) throw new Error("Global name must be at most 40 characters");
  pinnedImage(config.frontendImage); pinnedImage(config.discoveryImage);
  positive(config.frontendReplicas, "frontendReplicas"); positive(config.discoveryReplicas, "discoveryReplicas");
  validateStores(config.stores);
  if (![config.stores.homeRegion, ...config.stores.replicaRegions].includes(config.cluster.region)) {
    throw new Error("Each global deployment needs a local replica of both DynamoDB tables");
  }
  if (!config.affinityScope?.trim()) throw new Error("affinityScope is required and must match across replicas");
  if (config.affinityTimeoutMs !== undefined) positive(config.affinityTimeoutMs, "affinityTimeoutMs");
  if (config.discoveryMode !== "registrations") {
    throw new Error("Global discovery must use registrations backed by DynamoDB; MCS/Karmada is not supported by this deployment path");
  }
  if (config.routers !== undefined) throw new Error("Remove routers: each replica registers its own endpoint through DynamoDB discovery");
  if (config.podAddressFamily !== undefined && !["ipv4", "ipv6"].includes(config.podAddressFamily)) {
    throw new Error("podAddressFamily must match the cluster: ipv4 or ipv6");
  }
  if (config.ingress && (!config.ingress.sourceCidrs.length || config.ingress.sourceCidrs.some(c => !isValidIpv4Cidr(c)))) {
    throw new Error("Ingress requires explicit valid source CIDRs");
  }
}

export function globalManifests(config: GlobalLayerConfig): Record<string, any>[] {
  validateGlobal(config);
  const { name, namespace } = config;
  const router = `${name}-router`, discovery = `${name}-discovery`;
  const metadata = (name: string) => ({ name, namespace });
  const labels = (name: string) => ({ "app.kubernetes.io/name": name });
  const stateHost = config.podAddressFamily === "ipv6" ? "[$(POD_IP)]" : "$(POD_IP)";
  const frontendArgs = ["--port", "8000", "--relay-view-port", "50051",
    "--global-discovery-url", `http://${discovery}.${namespace}.svc:8080`,
    "--router-replica-id", `${config.cluster.name}/${namespace}/$(POD_UID)`,
    "--router-advertise-url", `http://${stateHost}:50051`,
    "--router-policy", "affinity", "--region", config.cluster.region,
    "--affinity-store", "dynamodb", "--affinity-scope", config.affinityScope,
    "--affinity-ddb-table", config.stores.affinityTableName, "--affinity-ddb-region", config.cluster.region,
    ...(config.affinityTimeoutMs !== undefined ? ["--affinity-timeout-ms", String(config.affinityTimeoutMs)] : [])];
  const discoveryArgs = ["--listen", "0.0.0.0:8080", "--pool-discovery", config.discoveryMode,
    "--router-discovery", "registrations",
    "--registration-storage", "dynamodb", "--dynamodb-table", config.stores.discoveryTableName,
    "--dynamodb-region", config.cluster.region];

  const deployment = (name: string, image: string, command: string, args: string[], replicas: number, port: number, health: string) => ({
    apiVersion: "apps/v1", kind: "Deployment", metadata: metadata(name),
    spec: { replicas, selector: { matchLabels: labels(name) }, template: {
      metadata: { labels: labels(name) }, spec: {
        serviceAccountName: name, terminationGracePeriodSeconds: 40,
        containers: [{ name: "main", image, command: [command], args,
          ...(name === router ? { env: [
            { name: "POD_UID", valueFrom: { fieldRef: { fieldPath: "metadata.uid" } } },
            { name: "POD_IP", valueFrom: { fieldRef: { fieldPath: "status.podIP" } } },
          ] } : {}),
          ports: [{ name: "http", containerPort: port }, ...(name === router ? [{ name: "state", containerPort: 50051 }] : [])],
          readinessProbe: { httpGet: { path: health, port: "http" }, periodSeconds: 5 },
          livenessProbe: { httpGet: { path: health, port: "http" }, periodSeconds: 10 },
          startupProbe: { httpGet: { path: health, port: "http" }, periodSeconds: 5, failureThreshold: 60 },
          resources: { requests: { cpu: "250m", memory: "512Mi" }, limits: { memory: "2Gi" } },
          securityContext: { allowPrivilegeEscalation: false, capabilities: { drop: ["ALL"] } },
        }],
      },
    } },
  });
  const service = (name: string, port: number) => ({ apiVersion: "v1", kind: "Service", metadata: metadata(name),
    spec: { selector: labels(name), ports: [{ name: "http", port, targetPort: "http" },
      ...(name === router ? [{ name: "state", port: 50051, targetPort: "state" }] : [])] } });
  const manifests: Record<string, any>[] = [
    deployment(router, config.frontendImage, "global-frontend", frontendArgs, config.frontendReplicas, 8000, "/live"),
    deployment(discovery, config.discoveryImage, "global-discovery-service", discoveryArgs, config.discoveryReplicas, 8080, "/healthz"),
    service(router, 8000), service(discovery, 8080),
  ];
  if (config.ingress) {
    const tls = config.ingress.certificateArn;
    manifests.push({ apiVersion: "networking.k8s.io/v1", kind: "Ingress", metadata: {
      ...metadata(router), annotations: {
        "alb.ingress.kubernetes.io/scheme": "internal", "alb.ingress.kubernetes.io/target-type": "ip",
        "alb.ingress.kubernetes.io/inbound-cidrs": config.ingress.sourceCidrs.join(","),
        "alb.ingress.kubernetes.io/healthcheck-path": "/live",
        ...(tls ? { "alb.ingress.kubernetes.io/certificate-arn": tls,
          "alb.ingress.kubernetes.io/listen-ports": '[{"HTTPS":443}]' } : {}),
      },
    }, spec: { ingressClassName: "alb", rules: [{ http: { paths: [{ path: "/", pathType: "Prefix",
      backend: { service: { name: router, port: { number: 8000 } } } }] } }] } });
  }
  return manifests;
}
