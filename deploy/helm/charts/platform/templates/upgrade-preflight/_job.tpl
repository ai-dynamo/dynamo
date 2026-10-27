# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

{{- define "dynamo-platform.upgradePreflight.serviceAccount" -}}
{{- $root := .root -}}
{{- $annotations := dict "helm.sh/hook" "pre-install,pre-upgrade" "helm.sh/hook-weight" "-20" "helm.sh/hook-delete-policy" "before-hook-creation" -}}
apiVersion: v1
kind: ServiceAccount
metadata:
  name: {{ .name }}
  namespace: {{ $root.Release.Namespace }}
  annotations: {{- toYaml $annotations | nindent 4 }}
{{- end -}}

{{- define "dynamo-platform.upgradePreflight.job" -}}
{{- $root := .root -}}
{{- $operator := .operator -}}
{{- $config := $operator.upgradePreflight.job -}}
{{- $podSecurityContext := $config.podSecurityContext -}}
{{- if eq $podSecurityContext nil -}}
{{- $podSecurityContext = dict "runAsNonRoot" true "runAsUser" 65532 "seccompProfile" (dict "type" "RuntimeDefault") -}}
{{- end -}}
apiVersion: batch/v1
kind: Job
metadata:
  name: {{ .name }}
  namespace: {{ $root.Release.Namespace }}
  annotations:
    helm.sh/hook: pre-install,pre-upgrade
    helm.sh/hook-weight: "-10"
    helm.sh/hook-delete-policy: before-hook-creation,hook-succeeded
    helm.sh/hook-output-log-policy: hook-failed
spec:
  activeDeadlineSeconds: {{ $config.activeDeadlineSeconds }}
  backoffLimit: 2
  template:
    spec:
      serviceAccountName: {{ .name }}
      restartPolicy: Never
      {{- with $config.imagePullSecrets }}
      imagePullSecrets: {{- toYaml . | nindent 8 }}
      {{- end }}
      {{- with $operator.controllerManager.tolerations }}
      tolerations: {{- toYaml . | nindent 8 }}
      {{- end }}
      {{- with $config.affinity }}
      affinity: {{- toYaml . | nindent 8 }}
      {{- end }}
      {{- with $podSecurityContext }}
      securityContext: {{- toYaml . | nindent 8 }}
      {{- end }}
      containers:
        - name: {{ .checkID }}
          image: {{ $config.image | quote }}
          command: [python3, -c]
          args:
            - |
              {{- .script | nindent 14 }}
            {{- range .args }}
            - {{ . | quote }}
            {{- end }}
          securityContext:
            allowPrivilegeEscalation: false
            readOnlyRootFilesystem: true
            capabilities:
              drop: [ALL]
          resources:
            requests: {cpu: 10m, memory: 32Mi}
            limits: {memory: 128Mi}
{{- end -}}

