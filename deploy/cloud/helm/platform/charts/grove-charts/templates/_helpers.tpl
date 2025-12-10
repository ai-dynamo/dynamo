{{- define "operator.config.data" -}}
config.yaml: |
  apiVersion: operator.config.grove.io/v1alpha1
  kind: OperatorConfiguration
  runtimeClientConnection:
    qps: {{ .Values.config.runtimeClientConnection.qps }}
    burst: {{ .Values.config.runtimeClientConnection.burst }}
  leaderElection:
    enabled: {{ .Values.config.leaderElection.enabled }}
    leaseDuration: {{ .Values.config.leaderElection.leaseDuration }}
    renewDeadline: {{ .Values.config.leaderElection.renewDeadline }}
    retryPeriod: {{ .Values.config.leaderElection.retryPeriod }}
    resourceLock: {{ .Values.config.leaderElection.resourceLock }}
    resourceName: {{ .Values.config.leaderElection.resourceName }}
    resourceNamespace: {{ .Release.Namespace }}
  server:
    webhooks:
      port: {{ .Values.config.server.webhooks.port }}
    healthProbes:
      port: {{ .Values.config.server.healthProbes.port }}
    metrics:
      port: {{ .Values.config.server.metrics.port }}
  controllers:
    podCliqueSet:
      concurrentSyncs: {{ .Values.config.controllers.podCliqueSet.concurrentSyncs }}
    podClique:
      concurrentSyncs: {{ .Values.config.controllers.podClique.concurrentSyncs }}
  {{- if .Values.config.debugging }}
  debugging:
    enableProfiling: {{ .Values.config.debugging.enableProfiling }}
  {{- end }}
  logLevel: {{ .Values.config.logLevel | default "info" }}
  logFormat: {{ .Values.config.logFormat | default "json" }}
  {{- if .Values.config.authorizer.enabled }}
  authorizer:
    enabled: {{ .Values.config.authorizer.enabled }}
    reconcilerServiceAccountUserName: {{ printf "system:serviceaccount:%s:%s" .Release.Namespace .Values.serviceAccount.name }}
    exemptServiceAccountUserNames:
    {{- if .Values.config.authorizer.exemptServiceAccountUserNames }}
    {{- range $idx, $name := .Values.config.authorizer.exemptServiceAccountUserNames }}
      - {{ $name }}
    {{- end }}
    {{- else }}
      []
    {{- end }}
  {{- end }}

{{- end -}}

{{- define "operator.config.name" -}}
grove-operator-cm-{{ include "operator.config.data" . | sha256sum | trunc 8 }}
{{- end -}}

{{- define "common.chart.labels" -}}
chart: "{{ .Chart.Name }}-{{ .Chart.Version }}"
release: "{{ .Release.Name }}"
{{- end -}}

{{- define "operator.config.labels" -}}
{{- include "common.chart.labels" . }}
{{- range $key, $val := .Values.configMap.labels }}
{{ $key }}: {{ $val }}
{{- end }}
{{- end -}}

{{- define "operator.deployment.matchLabels" -}}
{{- range $key, $val := .Values.deployment.labels }}
{{ $key }}: {{ $val }}
{{- end }}
{{- end -}}

{{- define "operator.deployment.labels" -}}
{{- include "common.chart.labels" . }}
{{- include "operator.deployment.matchLabels" . }}
{{- end -}}

{{- define "operator.serviceaccount.labels" -}}
{{- include "common.chart.labels" . }}
{{- range $key, $val := .Values.serviceAccount.labels }}
{{ $key }}: {{ $val }}
{{- end }}
{{- end -}}

{{- define "operator.service.labels" -}}
{{- include "common.chart.labels" . }}
{{- range $key, $val := .Values.service.labels }}
{{ $key }}: {{ $val }}
{{- end }}
{{- end -}}

{{- define "operator.clusterrole.labels" -}}
{{- include "common.chart.labels" . }}
{{- range $key, $val := .Values.clusterRole.labels }}
{{ $key }}: {{ $val }}
{{- end }}
{{- end -}}

{{- define "operator.clusterrolebinding.labels" -}}
{{- include "common.chart.labels" . }}
{{- range $key, $val := .Values.clusterRoleBinding.labels }}
{{ $key }}: {{ $val }}
{{- end }}
{{- end -}}

{{- define "image" -}}
{{- if hasPrefix "sha256:" (required "$.tag is required" $.tag) -}}
{{ required "$.repository is required" $.repository }}@{{ required "$.tag" $.tag }}
{{- else -}}
{{ required "$.repository is required" $.repository }}:{{ required "$.tag" $.tag }}
{{- end -}}
{{- end -}}

{{- define "operator.pcs.validating.webhook.labels" -}}
{{- include "common.chart.labels" . }}
{{- range $key, $val := .Values.webhooks.podCliqueSetValidationWebhook.labels }}
{{ $key }}: {{ $val }}
{{- end }}
{{- end -}}

{{- define "operator.pcs.defaulting.webhook.labels" -}}
{{- include "common.chart.labels" . }}
{{- range $key, $val := .Values.webhooks.podCliqueSetDefaultingWebhook.labels }}
{{ $key }}: {{ $val }}
{{- end }}
{{- end -}}

{{- define "operator.authorizer.webhook.labels" -}}
{{- include "common.chart.labels" . }}
{{- range $key, $val := .Values.webhooks.authorizerWebhook.labels }}
{{ $key }}: {{ $val }}
{{- end }}
{{- end -}}

{{- define "operator.server.secret.labels" -}}
{{- include "common.chart.labels" . }}
{{- range $key, $val := .Values.webhookServerSecret.labels }}
{{ $key }}: {{ $val }}
{{- end }}
{{- end -}}