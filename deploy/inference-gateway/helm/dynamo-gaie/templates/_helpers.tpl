# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

{{/*
Expand the name of the chart.
*/}}
{{- define "dynamo-gaie.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
If release name contains chart name it will be used as a full name.
*/}}
{{- define "dynamo-gaie.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "dynamo-gaie.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "dynamo-gaie.labels" -}}
helm.sh/chart: {{ include "dynamo-gaie.chart" . }}
{{ include "dynamo-gaie.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "dynamo-gaie.selectorLabels" -}}
app.kubernetes.io/name: {{ include "dynamo-gaie.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Resolve the fully qualified Dynamo namespace as "<release namespace>-<dynamoGraphDeploymentName>"
*/}}
{{- define "dynamo-gaie.dynamoNamespace" -}}
{{- $dgdName := (.Values.dynamoGraphDeploymentName | default "") | trim -}}
{{- if not $dgdName }}
{{- fail "set dynamoGraphDeploymentName to derive the Dynamo namespace" }}
{{- end }}
{{- $releaseNamespace := (.Release.Namespace | default "") | trim -}}
{{- if not $releaseNamespace }}
{{- fail "Release.Namespace must be set to derive the Dynamo namespace" }}
{{- end }}
{{- printf "%s-%s" $releaseNamespace $dgdName }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "dynamo-gaie.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "dynamo-gaie.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}
