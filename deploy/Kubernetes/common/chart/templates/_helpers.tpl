# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Annotation Groups
{{- define "nvidia.annotations.default" }}
dynamo: "{{ .Release.Name }}.{{ .Chart.AppVersion | default "0.0" }}"
{{-   with .Values.kubernetes }}
{{-     with .annotations }}
{{        toYaml . }}
{{-     end }}
{{-   end }}
{{- end -}}

{{- define "nvidia.annotations.chart" }}
helm.sh/chart: {{ .Chart.Name | quote }}
{{-   template "nvidia.annotations.default" . }}
{{- end -}}

# Label Groups
{{- define "nvidia.labels.default" }}
{{-   template "nvidia.label.appInstance" . }}
{{-   template "nvidia.label.appName" . }}
{{-   template "nvidia.label.appPartOf" . }}
{{-   template "nvidia.label.appVersion" . }}
{{- end -}}

{{- define "nvidia.labels.chart" }}
{{-   template "nvidia.labels.default" . }}
{{-   template "nvidia.label.appManagedBy" . }}
{{-   template "nvidia.label.chart" . }}
{{-   with .Values.kubernetes }}
{{-     with .labels }}
{{        toYaml . }}
{{-     end }}
{{-   end }}
{{-   template "nvidia.label.release" . }}
{{- end -}}

# Label Values
{{- define "nvidia.label.appInstance" }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{- define "nvidia.label.appManagedBy" }}
{{-   $service_name := "dynamo" }}
{{-   with .Release.Service }}
{{-     $service_name = . }}
{{-   end }}
app.kubernetes.io/managed-by: {{ $service_name }}
{{- end }}

{{- define "nvidia.label.appName" }}
app.kubernetes.io/name: {{ required "Property '.component.name' is required." .Values.component.name }}
{{- end }}

{{- define "nvidia.label.appPartOf" }}
{{-   $part_of := "dynamo" }}
{{-   with .Values.kubernetes }}
{{-     with .partOf }}
{{-       $part_of = . }}
{{-     end }}
{{-   end }}
app.kubernetes.io/part-of: {{ $part_of }}
{{- end }}

{{- define "nvidia.label.appVersion" }}
app.kubernetes.io/version: {{ .Chart.Version | default "0.0" | quote }}
{{- end }}

{{- define "nvidia.label.chart" }}
helm.sh/chart: {{ .Chart.Name | quote }}
helm.sh/version: {{ .Chart.Version | default "0.0" | quote }}
{{- end }}

{{- define "nvidia.label.release" }}
release: "{{ .Chart.Name }}_v{{ .Chart.Version | default "0.0" }}"
{{- end }}
