// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import { test } from "node:test";
import assert from "node:assert/strict";
import { collectorConfig } from "../lib/observability/collector";
import { dashboardBody } from "../lib/observability/dashboard";
import { ObservabilityConfig } from "../lib/observability/config";
const config: ObservabilityConfig = {
 fleet: "test", cluster: { name: "test-cluster", region: "us-east-2", securityGroupId: "sg-1234", kubectlServiceToken: "unused", oidcProviderArn: "unused" },
 monitoredServices: [{ namespace: "gr-global", name: "gr-router", minimumPods: 2 }],
 globalNamespace: "gr-global", poolNamespace: "gr-pools", routerName: "gr-router", addonVersion: "pinned", collectorImage: "otel:version",
 stores: { homeRegion: "us-east-2", replicaRegions: ["us-west-2"], affinityTableName: "affinity", discoveryTableName: "discovery" },
 dashboardRegions: [{ region: "us-east-2", clusterName: "test-cluster", pool: "blue" }, { region: "us-west-2", clusterName: "west-cluster", pool: "blue" }],
};
test("histogram counts and sums retain target identity until reset-aware delta conversion", () => {
 const c = collectorConfig(config);
 assert.deepEqual(c.service.pipelines.metrics.processors, ["memory_limiter", "transform/histograms", "filter/histograms", "cumulativetodelta", "transform/status", "batch"]);
 assert.equal(c.processors.cumulativetodelta.initial_value, "drop");
 assert.equal(c.exporters.awsemf.dimension_rollup_option, "NoDimensionRollup");
 assert.equal(c.receivers.prometheus.config.scrape_configs.length, 3);
 assert.ok(c.receivers.prometheus.config.scrape_configs[1].relabel_configs.some(r => "replacement" in r && r.replacement === "us-east-2"));
});
test("cache token panels count serving frontends only and keep reporting coverage visible", () => {
 const body = dashboardBody(config, true);
 const cachePanels = body.widgets.filter((w: any) => /cached input tokens|Input and cached tokens|Cached-token/i.test(w.properties.title ?? ""));
 assert.equal(cachePanels.length, 4);
 for (const panel of cachePanels) assert.ok(!JSON.stringify(panel).includes('"global-router"'));
 assert.ok(JSON.stringify(body).includes("dynamo_frontend_cached_tokens_count"));
 assert.ok(JSON.stringify(body).includes("not p95/p99"));
 for (const w of body.widgets as any[]) {
   assert.ok(w.x + w.width <= 24);
   for (const other of body.widgets as any[]) if (w !== other) assert.ok(!(w.x < other.x + other.width && w.x + w.width > other.x && w.y < other.y + other.height && w.y + w.height > other.y), "widgets overlap");
 }
});

test("unpaired token counters are never presented as a cache hit percentage", () => {
 // Actual Oregon 12:54 UTC deltas: individually valid, but different request cohorts.
 const input = 304048, cached = 330752;
 assert.ok(cached / input > 1);
 for (const global of [false, true]) {
   const body = dashboardBody(config, global);
   const panel = body.widgets.find((w: any) => w.properties.title?.includes("unpaired request stages")) as any;
   assert.ok(panel);
   assert.ok(panel.properties.metrics.some((m: any[]) => m[1] === "dynamo_frontend_input_tokens_total"));
   assert.ok(panel.properties.metrics.some((m: any[]) => m[1] === "dynamo_frontend_cached_tokens_sum"));
   assert.ok(panel.properties.metrics.every((m: any[]) => typeof m[0] === "string"));
   assert.ok(!body.widgets.some((w: any) => /KV-cache hit percentage/i.test(w.properties.title ?? "")));
   assert.ok(JSON.stringify(body).includes("same requests"));
 }
});

test("observability synthesis owns no fleet infrastructure or serving workloads", async () => {
 const { App } = await import("aws-cdk-lib");
 const { Template } = await import("aws-cdk-lib/assertions");
 const { ObservabilityStack } = await import("../lib/observability/stack");
 const app = new App();
 const stack = new ObservabilityStack(app, "Observe", { env: { account: "123456789012", region: "us-east-2" }, config: { ...config, globalDashboard: true, cluster: { ...config.cluster, kubectlServiceToken: "arn:aws:lambda:us-east-2:123456789012:function:provider", oidcProviderArn: "arn:aws:iam::123456789012:oidc-provider/oidc.eks.us-east-2.amazonaws.com/id/EXAMPLE" } } });
 const template = Template.fromStack(stack);
 for (const type of ["AWS::EC2::VPC", "AWS::EKS::Cluster", "AWS::EKS::Nodegroup", "AWS::DynamoDB::GlobalTable"]) template.resourceCountIs(type, 0);
 template.resourceCountIs("AWS::EKS::Addon", 1);
 const resources = template.toJSON().Resources;
 const manifests = JSON.stringify(Object.values(resources).filter((r: any) => r.Type === "Custom::AWSCDK-EKS-KubernetesResource"));
 assert.ok(!manifests.includes('"kind":"DynamoGraphDeployment"'));
 assert.ok(!manifests.includes('"kind":"DynamoPoolRelay"'));
 const global = JSON.stringify(resources.GlobalDashboard.Properties.DashboardBody);
 assert.ok(global.includes("alarm:west-cluster-router-server-errors"));
 assert.ok(!global.includes("alarm:west-cluster-test-cluster"));
});

test("worker discovery and gauge dimensions preserve independent targets and virtual engines", () => {
 const c = collectorConfig(config);
 const worker = c.receivers.prometheus.config.scrape_configs.find(j => j.job_name === "pool-worker")!;
 assert.ok(worker.relabel_configs.some(r => "regex" in r && r.regex === "system"));
 assert.ok(worker.relabel_configs.some(r => "regex" in r && r.regex === "decode|prefill|worker"));
 assert.ok(!JSON.stringify(worker).includes('"blue"'), "new DGD pools must be discovered dynamically");
 const declarations = c.exporters.awsemf.metric_declarations;
 const engine = declarations.find(d => d.dimensions[0].includes("engine"))!;
 for (const dimension of ["PodName", "worker_id", "model", "Pool"]) assert.ok(engine.dimensions[0].includes(dimension));
 const state = declarations.find(d => d.dimensions[0].includes("PodName") && d.dimensions[0].includes("state"))!;
 assert.ok(state.metric_name_selectors.includes("^dynamo_frontend_inflight_input_tokens$"));
});

test("gauge panels use target averages and dynamic discovery; occupancy is not a hit rate", () => {
 const body = dashboardBody(config, true);
 for (const title of ["Pool active requests", "Awaiting first token", "Worker scheduler waiting", "Worker scheduler running", "Global-router in-flight"]) {
   const panel = body.widgets.find((w: any) => w.properties.title?.startsWith(title)) as any;
   assert.ok(panel, title);
   assert.equal(panel.properties.stat, "Average");
   for (const metric of panel.properties.metrics) {
     assert.ok(metric[0].expression.startsWith("SUM(SEARCH("));
     assert.ok(metric[0].expression.includes("'Average', 60"));
     assert.ok(metric[0].expression.includes("PodName"));
   }
 }
 const routes = body.widgets.find((w: any) => w.properties.title?.startsWith("Requests / minute by destination")) as any;
 assert.ok(JSON.stringify(routes).includes("SEARCH("));
 assert.ok(!JSON.stringify(routes).includes('Pool="blue"'));
 const occupancy = body.widgets.find((w: any) => w.properties.title?.startsWith("Worker KV-cache occupancy")) as any;
 assert.equal(occupancy.properties.stat, "Maximum");
 assert.ok(!JSON.stringify(occupancy).includes("SUM("));
});
