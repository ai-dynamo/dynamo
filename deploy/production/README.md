<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Dynamo Production Profile

This profile is an opinionated production deployment path for Dynamo on Kubernetes. It keeps the Dynamo platform chart focused on Dynamo-owned resources and manages third-party platform add-ons as separate GitOps applications.

## Deployment Model

The production stack has two layers:

1. Platform add-ons managed by Argo CD as independent Helm releases.
2. Dynamo Platform from this repository, configured to integrate with those add-ons.

This keeps `deploy/helm/charts/platform` close to upstream Dynamo while still providing a repeatable full-stack deployment.

Each add-on under [`addons/`](addons/) has an individual README that records its Argo CD application, chart source, owned responsibilities, exclusions, verification commands, and upgrade notes.

## Baseline Add-ons

| Capability | Add-on | Why it is external |
| --- | --- | --- |
| GPU node lifecycle | NVIDIA GPU Operator | Owns drivers, device plugin, GFD, and DCGM exporter lifecycle. |
| Metrics and alerting | kube-prometheus-stack | Owns Prometheus, Grafana, Alertmanager, CRDs, and retention. |
| Logs | Loki + Fluentd | Owns collection, storage, retention, and log routing. |
| Image and cluster scanning | Trivy Operator | Owns vulnerability, SBOM, secret, and misconfiguration scans. |
| Runtime threat detection | Falco | Owns node-level runtime security sensors and rules. |
| Backup and restore | Velero | Owns cluster-resource and PV backup/restore. |
| External secrets | External Secrets Operator | Owns provider-specific secret synchronization. |
| Multinode scheduling | Grove + KAI Scheduler | Recommended Dynamo path for topology-aware gang scheduling. |
| HTTP gateway | SMG | Owns the OpenAI-compatible gateway in front of Dynamo Frontend. |
| GitOps | Argo CD | Owns delivery, drift correction, promotion, and audit trail. |

## Pinned Chart Versions

| Add-on | Chart source | Chart version |
| --- | --- | --- |
| NVIDIA GPU Operator | `https://helm.ngc.nvidia.com/nvidia/gpu-operator` | `v26.3.1` |
| kube-prometheus-stack | `https://prometheus-community.github.io/helm-charts` | `84.1.0` |
| Loki | `https://grafana.github.io/helm-charts` | `7.0.0` |
| Fluentd | `https://fluent.github.io/helm-charts` | `0.5.3` |
| Falco | `https://falcosecurity.github.io/charts` | `8.0.2` |
| Trivy Operator | `https://aquasecurity.github.io/helm-charts` | `0.32.1` |
| Velero | `https://vmware-tanzu.github.io/helm-charts` | `12.0.0` |
| External Secrets Operator | `https://charts.external-secrets.io` | `2.4.0` |
| KAI Scheduler | `ghcr.io/kai-scheduler/kai-scheduler` | `v0.14.0` |
| Grove | `ghcr.io/ai-dynamo/grove` | `v0.1.0-alpha.8` |
| SMG | `https://github.com/lightseekorg/smg.git` | `v1.4.1` |
| Dynamo PrometheusRules | Repository manifests | `main` |

## Optional Add-ons

| Capability | Add-on | When to use |
| --- | --- | --- |
| Event-driven scaling | KEDA | Use for scale-to-zero or trigger types beyond Planner/HPA. |
| Cluster-wide traces/log pipelines | OpenTelemetry Operator | Use when you need managed collectors or auto-instrumentation. |
| Kubernetes CI runners | Actions Runner Controller | Use only when GitHub Actions jobs must run inside the cluster. |
| Alternate multinode orchestration | LWS + Volcano | Use only when you intentionally choose the non-Grove path. |
| Continuous profiling | Parca Agent | Use when always-on CPU/GPU profiling is required. |
| Deprecated API checks | kube-no-trouble | Use before Kubernetes minor upgrades and before promoting GitOps changes. |

Pinned optional versions:

- KEDA chart `2.19.0`
- OpenTelemetry Operator chart `0.110.0`
- Actions Runner Controller chart `0.14.1`
- LWS chart `v0.8.0`
- Volcano chart `1.14.1`
- Parca chart `4.19.0`, Parca server `v0.27.1`, Parca Agent `v0.47.1`
- kube-no-trouble image `ghcr.io/doitintl/kube-no-trouble:0.7.3`

## Not Baseline

These projects remain useful in adjacent platform designs but are not part of the baseline Dynamo production profile:

- Kaniko, because `GoogleContainerTools/kaniko` is archived.
- OME, KubeAI, LLMKube, and llm-d, because Dynamo now provides its own deployment, model, routing, and scaling APIs.
- Kueue, JobSet, Kubeflow Trainer, MPI Operator, and Scheduler Plugins, because they target broader batch or training platforms rather than Dynamo serving by default.
- SkyPilot, dstack, Slurm operators, Plural, Yoke, Hoptimator, Canine, LiteIO, and CRI-O, because they are cluster-platform choices outside Dynamo's serving runtime.

## Install Flow

For the active A4 REAP lane, use the `ai-blaise/infrastructure` wrapper:

```bash
scripts/dynamo-reap/deploy-a4-production.sh
```

That wrapper applies this production profile, optional production integrations, the private Hugging Face model cache setup, and the REAP SGLang `DynamoGraphDeployment`.

1. Install Argo CD in the target cluster.
2. Register this repository with Argo CD.
3. Apply `gitops/project.yaml`.
4. Apply `gitops/root-app.yaml`.
5. Wait for the baseline add-ons to become healthy.
6. Run:

```bash
deploy/pre-deployment/pre-deployment-check.sh --profile production
```

7. Deploy or sync `dynamo-platform` from the Argo CD root app.
8. Verify Dynamo-created resources:

```bash
deploy/pre-deployment/pre-deployment-check.sh --require dynamo-crds,dynamo-webhooks,kai-queue
```

The root app assumes this repository is available at `https://github.com/ai-blaise/dynamo-prod-k8s.git` and reads the `main` branch. Override `spec.source.targetRevision` when validating a feature branch.

Optional integrations are stored under `gitops/optional` and are not deployed by the root app.

Integration examples are stored under `examples/`, including KEDA scaling through `DynamoGraphDeploymentScalingAdapter`, External Secrets for Hugging Face tokens, an AWS `ClusterSecretStore`, kube-no-trouble upgrade checks, an OpenTelemetry collector for Dynamo traces, and the DeepSeek REAP SGLang deployment described in `runbooks/deepseek-reap-sglang.md`.

The `addons/gpu-operator/values-k3s.yaml` overlay is available for k3s validation clusters. Keep it out of the baseline root app unless the target fleet standardizes on k3s.

## Dynamo Integration Values

The Dynamo Platform app uses `addons/dynamo-platform/values.yaml`. Key defaults:

- Kubernetes-native discovery remains enabled.
- Webhook failure policy remains `Fail`.
- Grove and KAI integration are enabled but not installed by the Dynamo chart.
- The operator receives the in-cluster Prometheus URL for Planner and metrics consumers.

## Verification

Run verification from the deployment host:

```bash
deploy/pre-deployment/pre-deployment-check.sh --profile production --output json
kubectl get applications -n argocd
kubectl get pods -A
kubectl get servicemonitor -A
kubectl get queues
```

After Dynamo is installed, verify Dynamo-specific resources:

```bash
deploy/pre-deployment/pre-deployment-check.sh --require dynamo-crds,dynamo-webhooks,kai-queue
kubectl get dgd,dgdr,dgdsa,dm -A
```

Cluster-specific checks are intentionally opt-in:

```bash
deploy/pre-deployment/pre-deployment-check.sh --require network-policies,trivy-reports,kubent
```

Optional add-ons expose their own checks:

```bash
deploy/pre-deployment/pre-deployment-check.sh --require keda,opentelemetry,parca,lws-volcano
```
