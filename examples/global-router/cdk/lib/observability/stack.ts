// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import { createHash } from "node:crypto";
import { Stack, StackProps, Duration, RemovalPolicy, CfnOutput, aws_iam as iam, aws_logs as logs, aws_eks as eksCfn, aws_cloudwatch as cw } from "aws-cdk-lib";
import * as eks from "aws-cdk-lib/aws-eks-v2";
import { Construct } from "constructs";
import { importCluster } from "../foundation/target";
import { ObservabilityConfig } from "./config";
import { collectorConfig, metricNamespace } from "./collector";
import { dashboardBody } from "./dashboard";
export class ObservabilityStack extends Stack {
 constructor(scope: Construct, id: string, props: StackProps & { config: ObservabilityConfig }) {
  super(scope, id, props);
  const c = props.config;
  if (!c.fleet || !c.collectorImage.includes(":")) throw new Error("Supply fleet and a pinned collector image");
  const cluster = importCluster(this, c.cluster);
  const groups = ["application", "dataplane", "host", "performance"].map(kind => new logs.LogGroup(this, `ContainerLogs-${kind}`, {
    logGroupName: `/aws/containerinsights/${c.cluster.name}/${kind}`, retention: logs.RetentionDays.ONE_WEEK, removalPolicy: RemovalPolicy.RETAIN,
  }));
  const metricLogs = new logs.LogGroup(this, "PrometheusLogs", { logGroupName: `/dynamo/${c.fleet}/prometheus`, retention: logs.RetentionDays.ONE_WEEK, removalPolicy: RemovalPolicy.RETAIN });
  const oidc = iam.OpenIdConnectProvider.fromOpenIdConnectProviderArn(this, "AgentOidc", c.cluster.oidcProviderArn);
  const issuer = c.cluster.oidcProviderArn.split("oidc-provider/")[1];
  const agentRole = new iam.Role(this, "AgentRole", { assumedBy: new iam.OpenIdConnectPrincipal(oidc).withConditions({ StringEquals: {
    [`${issuer}:aud`]: "sts.amazonaws.com", [`${issuer}:sub`]: "system:serviceaccount:amazon-cloudwatch:cloudwatch-agent",
  } }), managedPolicies: [iam.ManagedPolicy.fromAwsManagedPolicyName("CloudWatchAgentServerPolicy")] });
  const addon = new eksCfn.CfnAddon(this, "ContainerObservability", {
    addonName: "amazon-cloudwatch-observability", addonVersion: c.addonVersion, clusterName: c.cluster.name,
    serviceAccountRoleArn: agentRole.roleArn, resolveConflicts: "NONE",
    configurationValues: JSON.stringify({
      applicationSignals: { enabled: false }, dcgmExporter: { enabled: false }, neuronMonitor: { enabled: false },
      agent: { config: { logs: { metrics_collected: { kubernetes: { cluster_name: c.cluster.name, metrics_collection_interval: 60, enhanced_container_insights: false } }, force_flush_interval: 5 } } },
    }),
  });
  groups.forEach(g => addon.node.addDependency(g));
  const ns = new eks.KubernetesManifest(this, "Namespace", { cluster, manifest: [{ apiVersion: "v1", kind: "Namespace", metadata: { name: "gr-observability" } }] });
  const sa = new eks.ServiceAccount(this, "CollectorIdentity", { cluster, namespace: "gr-observability", name: "gr-metrics" });
  sa.node.addDependency(ns); metricLogs.grantWrite(sa);
  const cfg = JSON.stringify(collectorConfig(c));
  const manifest = new eks.KubernetesManifest(this, "Collector", { cluster, manifest: [
    { apiVersion: "rbac.authorization.k8s.io/v1", kind: "ClusterRole", metadata: { name: "gr-metrics" }, rules: [{ apiGroups: [""], resources: ["pods"], verbs: ["get", "list", "watch"] }] },
    { apiVersion: "rbac.authorization.k8s.io/v1", kind: "ClusterRoleBinding", metadata: { name: "gr-metrics" }, roleRef: { apiGroup: "rbac.authorization.k8s.io", kind: "ClusterRole", name: "gr-metrics" }, subjects: [{ kind: "ServiceAccount", name: "gr-metrics", namespace: "gr-observability" }] },
    { apiVersion: "v1", kind: "ConfigMap", metadata: { name: "gr-metrics", namespace: "gr-observability" }, data: { "config.yaml": cfg } },
    { apiVersion: "apps/v1", kind: "Deployment", metadata: { name: "gr-metrics", namespace: "gr-observability" }, spec: {
      replicas: 1, strategy: { type: "Recreate" }, selector: { matchLabels: { app: "gr-metrics" } }, template: {
        metadata: { labels: { app: "gr-metrics" }, annotations: { "checksum/config": createHash("sha256").update(cfg).digest("hex") } },
        spec: { serviceAccountName: "gr-metrics", containers: [{ name: "collector", image: c.collectorImage,
          args: ["--config=/etc/otel/config.yaml"], ports: [{ name: "health", containerPort: 13133 }],
          resources: { requests: { cpu: "100m", memory: "128Mi" }, limits: { memory: "512Mi" } },
          readinessProbe: { httpGet: { path: "/", port: "health" }, initialDelaySeconds: 10 },
          livenessProbe: { httpGet: { path: "/", port: "health" }, initialDelaySeconds: 30 },
          securityContext: { allowPrivilegeEscalation: false, readOnlyRootFilesystem: true, capabilities: { drop: ["ALL"] } },
          volumeMounts: [{ name: "config", mountPath: "/etc/otel", readOnly: true }],
        }], volumes: [{ name: "config", configMap: { name: "gr-metrics" } }] },
      },
    } },
  ] }); manifest.node.addDependency(sa);
  const appMetric = (name: string, component = "global-router", statistic = "Sum") => new cw.Metric({ namespace: metricNamespace, metricName: name, dimensionsMap: { ClusterName: c.cluster.name, Component: component }, statistic, period: Duration.minutes(1) });
  const alarms: cw.Alarm[] = [];
  const alarmNames: string[] = [];
  const alarm = (name: string, metric: cw.IMetric, threshold: number, comparison = cw.ComparisonOperator.GREATER_THAN_THRESHOLD, missing = cw.TreatMissingData.NOT_BREACHING) => { const created = new cw.Alarm(this, name, { alarmName: `${c.cluster.name}-${name}`, alarmDescription: "On-call simulation threshold; inspect regional dashboard and application logs. No notification actions configured.", metric, threshold, comparisonOperator: comparison, evaluationPeriods: 3, datapointsToAlarm: 2, treatMissingData: missing }); alarms.push(created); alarmNames.push(name); return created; };
  for (const component of ["global-router", "pool-frontend"]) alarm(`${component}-scrape-unhealthy`, appMetric("up", component, "Minimum"), 1, cw.ComparisonOperator.LESS_THAN_THRESHOLD, cw.TreatMissingData.BREACHING);
  for (const [label, name] of [["affinity-degraded", "affinity_store_errors_total"], ["request-rejected", "rejected_requests_total"], ["response-interrupted", "response_errors_total"]]) alarm(label, appMetric(`dynamo_global_frontend_${name}`), 0);
  alarm("router-server-errors", appMetric("dynamo_global_frontend_requests_total").with({ dimensionsMap: { ClusterName: c.cluster.name, Component: "global-router", ResponseClass: "server_error" } }), 0);
  alarm("pool-request-errors", appMetric("dynamo_frontend_requests_total", "pool-frontend").with({ dimensionsMap: { ClusterName: c.cluster.name, Component: "pool-frontend", status: "error" } }), 0);
  for (const service of c.monitoredServices) alarm(`${service.name}-pods-missing`, new cw.Metric({ namespace: "ContainerInsights", metricName: "service_number_of_running_pods", dimensionsMap: { ClusterName: c.cluster.name, Namespace: service.namespace, Service: service.name }, statistic: "Minimum", period: Duration.minutes(1) }), service.minimumPods, cw.ComparisonOperator.LESS_THAN_THRESHOLD, cw.TreatMissingData.BREACHING);
  alarm("pool-model-unready", appMetric("dynamo_frontend_model_ready", "pool-frontend", "Minimum"), 1, cw.ComparisonOperator.LESS_THAN_THRESHOLD, cw.TreatMissingData.BREACHING);
  alarm("router-mean-latency-high", new cw.MathExpression({ expression: "IF(n>=10,s/n)", usingMetrics: { s: appMetric("dynamo_global_frontend_request_duration_seconds_sum"), n: appMetric("dynamo_global_frontend_request_duration_seconds_count") }, period: Duration.minutes(1) }), 5);
  alarm("node-memory-high", new cw.Metric({ namespace: "ContainerInsights", metricName: "node_memory_utilization", dimensionsMap: { ClusterName: c.cluster.name }, statistic: "Maximum", period: Duration.minutes(1) }), 85);
  for (const table of [c.stores.affinityTableName, c.stores.discoveryTableName]) for (const kind of ["ReadThrottleEvents", "WriteThrottleEvents"]) alarm(`${table}-${kind}`, new cw.Metric({ namespace: "AWS/DynamoDB", metricName: kind, dimensionsMap: { TableName: table }, statistic: "Sum", period: Duration.minutes(1) }), 0);
  const dashboardName = `${c.fleet}-${c.cluster.region}`;
  const body = (global: boolean) => {
    const dashboard = dashboardBody(c, global);
    for (const widget of dashboard.widgets) if ((widget.y as number) >= 3) widget.y = (widget.y as number) + 3;
    const regionalNames = alarmNames;
    const arns = global ? c.dashboardRegions.flatMap(r => regionalNames.map(name => `arn:${this.partition}:cloudwatch:${r.region}:${this.account}:alarm:${r.clusterName}-${name}`)) : alarms.map(a => a.alarmArn);
    dashboard.widgets.push({ type: "alarm", x: 0, y: 3, width: 24, height: 3, properties: { title: "On-call alarms — no notification actions", alarms: arns, sortBy: "stateUpdatedTimestamp" } });
    return JSON.stringify(dashboard);
  };
  new cw.CfnDashboard(this, "RegionalDashboard", { dashboardName, dashboardBody: body(false) });
  if (c.globalDashboard) new cw.CfnDashboard(this, "GlobalDashboard", { dashboardName: `${c.fleet}-global`, dashboardBody: body(true) });
  new CfnOutput(this, "Dashboard", { value: `https://${c.cluster.region}.console.aws.amazon.com/cloudwatch/home?region=${c.cluster.region}#dashboards:name=${dashboardName}` });
 }
}
