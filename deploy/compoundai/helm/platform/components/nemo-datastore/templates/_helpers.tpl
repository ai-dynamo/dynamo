{{/*
Expand the name of the chart.
*/}}
{{- define "datastore.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
If release name contains chart name it will be used as a full name.
*/}}
{{- define "datastore.fullname" -}}
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
Gitea service name
*/}}
{{- define "datastore.gitea.fullname" -}}
{{- printf "%s-%s" .Release.Name "gitea-http" }}
{{- end -}}


{{/*
Gitea internal endpoint accessible from datastore service
Note: we are using default gitea port: 3000
*/}}
{{- define "datastore.gitea.inernal-endpoint" -}}
{{- printf "http://%s:3000" (include "datastore.gitea.fullname" .) }}
{{- end -}}

{{/*
Service endpoint for gitea postgresql server deployed in cluster
*/}}
{{- define "datastore.gitea.inernal-postgres.host" -}}
{{- printf "%s-%s" .Release.Name "postgresqlgitea" }}
{{- end -}}


{{/*
API PostgreSQL endpoint to create Database Url
*/}}
{{- define "datastore.api.database.endpoint" -}}
{{- if .Values.postgresqlapi.enabled -}}
{{- printf "%s-postgresqlapi:5432/%s" .Release.Name .Values.postgresqlapi.auth.database }}
{{- else -}}
{{- printf "%s:%v/%s" .Values.externalPostgresqlApi.host .Values.externalPostgresqlApi.port .Values.externalPostgresqlApi.auth.database }}
{{- end -}}
{{- end -}}

{{/*
API PostgreSQL User
*/}}
{{- define "datastore.database.user" -}}
{{- if .Values.postgresqlapi.enabled -}}
{{- print .Values.postgresqlapi.auth.username -}}
{{- else -}}
{{- print .Values.externalPostgresqlApi.auth.username -}}
{{- end -}}
{{- end -}}


{{/*
API PostgreSQL DB Secret Name
*/}}
{{- define "datastore.api.database.secretName" -}}
{{- if and (.Values.postgresqlapi.enabled) (.Values.postgresqlapi.auth.existingSecret) -}}
{{- print .Values.postgresqlapi.auth.existingSecret -}}
{{- else if and (not .Values.postgresqlapi.enabled) (.Values.externalPostgresqlApi.auth.existingSecret) -}}
{{- print .Values.externalPostgresqlApi.auth.existingSecret -}}
{{- else -}}
{{- printf "%s-postgresqlapi" .Release.Name -}}
{{- end -}}
{{- end -}}

{{/*
Service endpoint for minio deployed in cluster
Note: we are using default minio port: 9000
*/}}
{{- define "datastore.gitea.inernal-minio.host" -}}
{{- if and (.Values.minio.apiIngress.enabled) (.Values.minio.apiIngress.hostname) -}}
{{- print .Values.minio.apiIngress.hostname  -}}
{{- else -}}
{{- printf "%s-minio.%s.svc.cluster.local:9000" .Release.Name .Release.Namespace }}
{{- end -}}
{{- end -}}


{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "datastore.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "datastore.labels" -}}
helm.sh/chart: {{ include "datastore.chart" . }}
{{ include "datastore.selectorLabels" . }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "datastore.selectorLabels" -}}
app.kubernetes.io/name: {{ include "datastore.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}


{{/*
Generates a random LFS_JWT_SECRET for gitea server
https://github.com/go-gitea/gitea/issues/22727
*/}}
{{- define "datastore.gitea.random.lfs_jwt_token" -}}
{{- randAlphaNum 32 | b64enc | replace "=" "" | replace "/" "+" | replace "=" "" | replace "_" "-" -}}
{{- end }}


{{/*
Create the name of the service account to use
*/}}
{{- define "datastore.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "datastore.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{- define "datastore.service.name" -}}
{{- default (include "datastore.fullname" .) .Values.serviceNameOverride }}
{{- end }}

{{/*
Additional secret provided to configure gitea
*/}}
{{- define "datastore.gitea.additional.secret.name" -}}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- $secretName := "gitea-service-secret" -}}
{{- range $idx, $value := .Values.gitea.gitea.additionalConfigSources }}
{{- if .secret -}}
    {{- $secretName = .secret.secretName  -}}
{{- end -}}
{{- end }}
{{- print $secretName -}}
{{- end -}}

{{/*
Additional gitea configuration
*/}}
{{- define "datastore.gitea.additional.config.name" -}}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- $configName := "gitea-config" -}}
{{- range $idx, $value := .Values.gitea.gitea.additionalConfigSources }}
{{- if .configMap -}}
    {{- $configName = .configMap.name -}}
{{- end -}}
{{- end }}
{{- print $configName -}}
{{- end -}}

{{/*
Image pull secret data for NGC
*/}}
{{- define "datastore.nvcr.imagepull.secret.data" -}}
{{- $username := "$oauthtoken" -}}
{{- $password := .Values.demo.ngcApiKey -}}
{{- $registry := "nvcr.io" -}}
{{- printf "{\"auths\":{\"%s\":{\"username\":\"%s\",\"password\":\"%s\",\"auth\":\"%s\"}}}" $registry $username $password (printf "%s:%s" $username $password | b64enc) | b64enc }}
{{- end }}


{{- define "datastore.api.database.serviceName" -}}
{{ printf "%s-postgresqlapi" .Release.Name }}
{{- end -}}