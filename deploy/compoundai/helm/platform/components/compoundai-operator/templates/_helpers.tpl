{{/*
Expand the name of the chart.
*/}}
{{- define "compoundai-operator.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
If release name contains chart name it will be used as a full name.
*/}}
{{- define "compoundai-operator.fullname" -}}
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
{{- define "compoundai-operator.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{- define "compoundai-operator.compoundai.envname" -}}
{{ include "compoundai-operator.fullname" . }}-compoundai-env
{{- end }}

{{/*
Generate k8s robot token
*/}}
{{- define "compoundai-operator.yataiApiToken" -}}
    {{- $secretObj := (lookup "v1" "Secret" .Release.Namespace (include "compoundai-operator.compoundai.envname" .)) | default dict }}
    {{- $secretData := (get $secretObj "data") | default dict }}
    {{- (get $secretData "YATAI_API_TOKEN") | default (randAlphaNum 16 | nospace | b64enc) | b64dec }}
{{- end -}}

{{/*
Common labels
*/}}
{{- define "compoundai-operator.labels" -}}
helm.sh/chart: {{ include "compoundai-operator.chart" . }}
{{ include "compoundai-operator.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "compoundai-operator.selectorLabels" -}}
app.kubernetes.io/name: {{ include "compoundai-operator.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "compoundai-operator.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "compoundai-operator.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Generate docker config json for registry credentials
*/}}
{{- define "compoundai-operator.dockerconfig" -}}
{{- $server := .Values.compoundai.dockerRegistry.server -}}
{{- $username := .Values.compoundai.dockerRegistry.username -}}
{{- $password := default .Values.global.NGC_API_KEY .Values.compoundai.dockerRegistry.password -}}
{{- if .Values.compoundai.dockerRegistry.passwordExistingSecretName -}}
{{- $password = .Values.compoundai.dockerRegistry.passwordExistingSecretKey -}}
{{- end -}}
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
