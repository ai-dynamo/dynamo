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
*/}}
{{- define "dynamo-operator.dockerconfig" -}}
{{- $server := .Values.dynamo.dockerRegistry.server -}}
{{- $username := .Values.dynamo.dockerRegistry.username -}}
{{- $password := .Values.dynamo.dockerRegistry.password -}}
{{- if .Values.dynamo.dockerRegistry.passwordExistingSecretName -}}
  {{- $secretName := .Values.dynamo.dockerRegistry.passwordExistingSecretName -}}
  {{- $secretKey := .Values.dynamo.dockerRegistry.passwordExistingSecretKey -}}
  {{- $dockerconfigjson := lookup "v1" "Secret" .Release.Namespace $secretName }}

  {{- if $dockerconfigjson -}}
    {{- if eq $dockerconfigjson.type "kubernetes.io/dockerconfigjson" -}}
      {{/* If the secret is already of type kubernetes.io/dockerconfigjson, use its .dockerconfigjson value directly */}}
      {{- index $dockerconfigjson.data ".dockerconfigjson" | b64dec }}
    {{- else -}}
      {{/* If the secret is not of the correct type, extract password from the secret and build a new one */}}
      {{- $password = index $dockerconfigjson.data $secretKey | b64dec }}
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
  {{- else -}}
    {{/* If no secret is found, use the default password */}}
    {{- $password = .Values.dynamo.dockerRegistry.password }}
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
{{- else -}}
  {{/* Build a new dockerconfigjson if passwordExistingSecretName is not set */}}
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