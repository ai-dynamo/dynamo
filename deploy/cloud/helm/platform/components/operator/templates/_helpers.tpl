# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
{{- define "dynamo-operator.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
If release name contains chart name it will be used as a full name.
*/}}
{{- define "dynamo-operator.fullname" -}}
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
{{- define "dynamo-operator.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{- define "dynamo-operator.dynamo.envname" -}}
{{ include "dynamo-operator.fullname" . }}-dynamo-env
{{- end }}


{{/*
Common labels
*/}}
{{- define "dynamo-operator.labels" -}}
helm.sh/chart: {{ include "dynamo-operator.chart" . }}
{{ include "dynamo-operator.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "dynamo-operator.selectorLabels" -}}
app.kubernetes.io/name: {{ include "dynamo-operator.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "dynamo-operator.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "dynamo-operator.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Generate docker config json for registry credentials
Supports both direct credentials and existing secrets
*/}}
{{- define "dynamo-operator.dockerconfig" -}}
{{- $server := required "docker registry server is required" .Values.dynamo.dockerRegistry.server -}}
{{- if or (eq $server "docker.io") (hasPrefix "docker.io/" $server) -}}
  {{- $server = "https://index.docker.io/v1/" -}}
{{- end -}}

{{- if .Values.dynamo.dockerRegistry.existingSecretName -}}
  {{/* Use existing secret */}}
  {{- $secretName := .Values.dynamo.dockerRegistry.existingSecretName -}}
  {{- $secret := lookup "v1" "Secret" .Release.Namespace $secretName -}}
  {{- if not $secret -}}
    {{- fail (printf "Secret %s not found in namespace %s" $secretName .Release.Namespace) -}}
  {{- end -}}
  {{- if ne $secret.type "kubernetes.io/dockerconfigjson" -}}
    {{- fail (printf "Secret %s in namespace %s is not of type kubernetes.io/dockerconfigjson" $secretName .Release.Namespace) -}}
  {{- end -}}
  {{- index $secret.data ".dockerconfigjson" | b64dec -}}
{{- else -}}
  {{/* Use direct credentials */}}
  {{- $username := required "docker registry username is required when not using existing secret" .Values.dynamo.dockerRegistry.username -}}
  {{- $password := required "docker registry password is required when not using existing secret" .Values.dynamo.dockerRegistry.password -}}
  {
    "auths": {
      "{{ $server }}": {
        "username": "{{ $username }}",
        "password": "{{ $password }}",
        "auth": "{{ printf "%s:%s" $username $password | b64enc }}"
      }
    }
  }
{{- end -}}
{{- end -}}

{{/*
Validate docker config secret has auth for server (extracted from full repository path)
Usage:
  {{ include "dynamo-operator.validateDockerConfigSecret" (dict "secretName" "my-secret" "namespace" .Release.Namespace "repository" "myserver.com/myrepo") }}
Returns: Error if invalid or missing auth, empty string if valid
*/}}
{{- define "dynamo-operator.validateDockerConfigSecret" -}}
{{- $server := regexSplit "/" .repository 2 | first -}}
{{- $secret := lookup "v1" "Secret" .namespace .secretName -}}
{{- if not $secret -}}
  {{- fail (printf "Secret %s not found in namespace %s" .secretName .namespace) -}}
{{- end -}}
{{- if ne $secret.type "kubernetes.io/dockerconfigjson" -}}
  {{- fail (printf "Secret %s in namespace %s is not of type kubernetes.io/dockerconfigjson" .secretName .namespace) -}}
{{- end -}}
{{- $dockerConfig := index $secret.data ".dockerconfigjson" | b64dec | fromJson -}}
{{- if not (hasKey $dockerConfig "auths") -}}
  {{- fail (printf "Secret %s in namespace %s does not contain auths field" .secretName .namespace) -}}
{{- end -}}
{{- if not (hasKey $dockerConfig.auths $server) -}}
  {{- fail (printf "Secret %s in namespace %s does not contain auth for server %s (extracted from %s)" .secretName .namespace $server .repository) -}}
{{- end -}}
{{- end -}}

{{/*
Retrieve components docker registry secret name
*/}}
{{- define "dynamo-operator.componentsDockerRegistrySecretName" -}}
{{- if .Values.dynamo.dockerRegistry.existingSecretName -}}
  {{- .Values.dynamo.dockerRegistry.existingSecretName -}}
{{- else -}}
  {{- printf "%s-regcred" (include "dynamo-operator.fullname" .) -}}
{{- end -}}
{{- end -}}