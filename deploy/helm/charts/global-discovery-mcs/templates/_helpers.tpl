# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

{{/*
Expand the name of the chart.
*/}}
{{- define "global-discovery-mcs.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Prefix of the resource names. The components are <prefix>-registry and
<prefix>-controller; the default prefix "mcs" names the Deployments
mcs-registry and mcs-controller, the names the documentation uses. The
names do not depend on the release name: one release per namespace, and one
controller per cluster.
*/}}
{{- define "global-discovery-mcs.fullname" -}}
{{- default "mcs" .Values.fullnameOverride | trunc 52 | trimSuffix "-" }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "global-discovery-mcs.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "global-discovery-mcs.labels" -}}
helm.sh/chart: {{ include "global-discovery-mcs.chart" . }}
{{ include "global-discovery-mcs.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/part-of: dynamo-global-router
{{- end }}

{{/*
Selector labels
*/}}
{{- define "global-discovery-mcs.selectorLabels" -}}
app.kubernetes.io/name: {{ include "global-discovery-mcs.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Labels and selector labels of one component (registry or controller).
Usage: include "global-discovery-mcs.componentLabels" (dict "context" . "component" "registry")
*/}}
{{- define "global-discovery-mcs.componentLabels" -}}
{{ include "global-discovery-mcs.labels" .context }}
app.kubernetes.io/component: {{ .component }}
{{- end }}

{{- define "global-discovery-mcs.componentSelectorLabels" -}}
{{ include "global-discovery-mcs.selectorLabels" .context }}
app.kubernetes.io/component: {{ .component }}
{{- end }}

{{/*
Resource names of the two components.
*/}}
{{- define "global-discovery-mcs.registry.fullname" -}}
{{- printf "%s-registry" (include "global-discovery-mcs.fullname" .) | trunc 63 | trimSuffix "-" }}
{{- end }}

{{- define "global-discovery-mcs.controller.fullname" -}}
{{- printf "%s-controller" (include "global-discovery-mcs.fullname" .) | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
ServiceAccount names of the two components.
*/}}
{{- define "global-discovery-mcs.registry.serviceAccountName" -}}
{{- if .Values.registry.serviceAccount.create }}
{{- default (include "global-discovery-mcs.registry.fullname" .) .Values.registry.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.registry.serviceAccount.name }}
{{- end }}
{{- end }}

{{- define "global-discovery-mcs.controller.serviceAccountName" -}}
{{- if .Values.controller.serviceAccount.create }}
{{- default (include "global-discovery-mcs.controller.fullname" .) .Values.controller.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.controller.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
The global-discovery-mcs image. An empty image.tag uses the chart appVersion.
*/}}
{{- define "global-discovery-mcs.image" -}}
{{- printf "%s:%s" .Values.image.repository (.Values.image.tag | default .Chart.AppVersion | toString) }}
{{- end }}

{{/*
Registry URLs of the controller as a JSON list: controller.registryUrls, or
the registry Service of this release when that list is empty and
registry.enabled. The in-release URL uses https when registry.tls.secretName
is set; its certificate must then name mcs-registry.<namespace>.svc.
*/}}
{{- define "global-discovery-mcs.controller.registryURLs" -}}
{{- $urls := .Values.controller.registryUrls | default list }}
{{- if and (not $urls) .Values.registry.enabled }}
{{- $scheme := ternary "https" "http" (ne (.Values.registry.tls.secretName | default "") "") }}
{{- $urls = list (printf "%s://%s.%s.svc:%d" $scheme (include "global-discovery-mcs.registry.fullname" .) .Release.Namespace (.Values.registry.ports.grpc | int)) }}
{{- end }}
{{- toJson $urls }}
{{- end }}

{{/*
Mount path of the registry CA bundle in the controller.
*/}}
{{- define "global-discovery-mcs.controller.caDir" -}}
/etc/mcs-controller/registry-ca
{{- end }}

{{/*
Mount path of the registry TLS key pair.
*/}}
{{- define "global-discovery-mcs.registry.tlsDir" -}}
/etc/mcs-registry/tls
{{- end }}

{{/*
Fail on registry values the binary or the deployment would reject.
*/}}
{{- define "global-discovery-mcs.registry.validate" -}}
{{- $r := .Values.registry }}
{{- if and (eq $r.store "memory") (gt (int $r.replicas) 1) }}
{{- fail (printf "registry.store=memory requires registry.replicas=1, got %d: every replica would keep its own state. Use registry.store=dynamodb to run more replicas." (int $r.replicas)) }}
{{- end }}
{{- if and (eq $r.store "dynamodb") (not $r.dynamodb.table) }}
{{- fail "registry.dynamodb.table is required with registry.store=dynamodb" }}
{{- end }}
{{- end }}

{{/*
Fail on controller values the binary would reject at startup, so that a
broken configuration fails at install time instead of crash-looping.
*/}}
{{- define "global-discovery-mcs.controller.validate" -}}
{{- $c := .Values.controller }}
{{- $urls := include "global-discovery-mcs.controller.registryURLs" . | fromJsonArray }}
{{- if not $urls }}
{{- fail "controller.registryUrls is required unless registry.enabled deploys the registry in this release" }}
{{- end }}
{{- $schemes := dict }}
{{- range $urls }}
{{- $scheme := (urlParse .).scheme }}
{{- if not (has $scheme (list "http" "https")) }}
{{- fail (printf "controller.registryUrls: %q must be an https:// or http:// URL" .) }}
{{- end }}
{{- if and (eq $scheme "http") (not $c.registryAllowInsecureHttp) }}
{{- fail (printf "controller.registryUrls: %q is plaintext; set controller.registryAllowInsecureHttp=true or serve the registry over TLS (registry.tls.secretName)" .) }}
{{- end }}
{{- $_ := set $schemes $scheme true }}
{{- end }}
{{- if gt (len $schemes) 1 }}
{{- fail "controller.registryUrls: every URL must use the same scheme" }}
{{- end }}
{{- if and $c.registryTlsCa.configMapName $c.registryTlsCa.secretName }}
{{- fail "controller.registryTlsCa: set configMapName or secretName, not both" }}
{{- end }}
{{- if and (eq $c.clusterIdentitySource "flags") (or (not $c.clustersetId) (not $c.clusterId)) }}
{{- fail "controller.clusterIdentitySource=flags requires controller.clustersetId and controller.clusterId" }}
{{- end }}
{{- if and (gt (int $c.replicas) 1) (not $c.leaderElect) }}
{{- fail (printf "controller.replicas=%d requires controller.leaderElect=true" (int $c.replicas)) }}
{{- end }}
{{- end }}
