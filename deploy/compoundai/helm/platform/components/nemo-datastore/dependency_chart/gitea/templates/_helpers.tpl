{{/* vim: set filetype=mustache: */}}
{{/*
Expand the name of the chart.
*/}}

{{- /* multiple replicas assertions */ -}}
{{- if gt .Values.replicaCount 1.0 -}}
  {{- fail "When using multiple replicas, a RWX file system is required" -}}
  {{- if eq (get (.Values.persistence.accessModes 0) "ReadWriteOnce") -}}
    {{- fail "When using multiple replicas, a RWX file system is required" -}}
  {{- end }}
  
  {{- if eq (get .Values.gitea.config.indexer "ISSUE_INDEXER_TYPE") "bleve" -}}
    {{- fail "When using multiple replicas, the repo indexer must be set to 'meilisearch' or 'elasticsearch'" -}}
  {{- end }}
  
  {{- if and (eq .Values.gitea.config.indexer.REPO_INDEXER_TYPE "bleve") (eq .Values.gitea.config.indexer.REPO_INDEXER_ENABLED "true") -}}
    {{- fail "When using multiple replicas, the repo indexer must be set to 'meilisearch' or 'elasticsearch'" -}}
  {{- end }}
  
  {{- if eq .Values.gitea.config.indexer.ISSUE_INDEXER_TYPE "bleve" -}}
    {{- (printf "DEBUG: When using multiple replicas, the repo indexer must be set to 'meilisearch' or 'elasticsearch'") | fail -}}
  {{- end }}
{{- end }}

{{- define "gitea.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/*
Create a default fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
If release name contains chart name it will be used as a full name.
*/}}
{{- define "gitea.fullname" -}}
{{- if .Values.fullnameOverride -}}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- $name := default .Chart.Name .Values.nameOverride -}}
{{- if contains $name .Release.Name -}}
{{- .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}
{{- end -}}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "gitea.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/*
Create image name and tag used by the deployment.
*/}}
{{- define "gitea.image" -}}
{{- $fullOverride := .Values.image.fullOverride | default "" -}}
{{- $registry := .Values.global.imageRegistry | default .Values.image.registry -}}
{{- $repository := .Values.image.repository -}}
{{- $separator := ":" -}}
{{- $tag := .Values.image.tag | default .Chart.AppVersion | toString -}}
{{- $rootless := ternary "-rootless" "" (.Values.image.rootless) -}}
{{- $digest := "" -}}
{{- if .Values.image.digest }}
    {{- $digest = (printf "@%s" (.Values.image.digest | toString)) -}}
{{- end -}}
{{- if $fullOverride }}
    {{- printf "%s" $fullOverride -}}
{{- else if $registry }}
    {{- printf "%s/%s%s%s%s%s" $registry $repository $separator $tag $rootless $digest -}}
{{- else -}}
    {{- printf "%s%s%s%s%s" $repository $separator $tag $rootless $digest -}}
{{- end -}}
{{- end -}}

{{/*
Docker Image Registry Secret Names evaluating values as templates
*/}}
{{- define "gitea.images.pullSecrets" -}}
{{- $pullSecrets := .Values.imagePullSecrets -}}
{{- range .Values.global.imagePullSecrets -}}
    {{- $pullSecrets = append $pullSecrets (dict "name" .) -}}
{{- end -}}
{{- if (not (empty $pullSecrets)) }}
imagePullSecrets:
{{ toYaml $pullSecrets }}
{{- end }}
{{- end -}}


{{/*
Storage Class
*/}}
{{- define "gitea.persistence.storageClass" -}}
{{- $storageClass := .Values.persistence.storageClass | default .Values.global.storageClass }}
{{- if $storageClass }}
storageClassName: {{ $storageClass | quote }}
{{- end }}
{{- end -}}

{{/*
Common labels
*/}}
{{- define "gitea.labels" -}}
helm.sh/chart: {{ include "gitea.chart" . }}
app: {{ include "gitea.name" . }}
{{ include "gitea.selectorLabels" . }}
app.kubernetes.io/version: {{ .Values.image.tag | default .Chart.AppVersion | quote }}
version: {{ .Values.image.tag | default .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end -}}

{{/*
Selector labels
*/}}
{{- define "gitea.selectorLabels" -}}
app.kubernetes.io/name: {{ include "gitea.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}

{{- define "postgresql-ha.dns" -}}
{{- if (index .Values "postgresql-ha").enabled -}}
{{- printf "%s-postgresql-ha-pgpool.%s.svc.%s:%g" .Release.Name .Release.Namespace .Values.clusterDomain (index .Values "postgresql-ha" "service" "ports" "postgresql") -}}
{{- end -}}
{{- end -}}

{{- define "postgresql.dns" -}}
{{- if (index .Values "postgresql").enabled -}}
{{- printf "%s-postgresql.%s.svc.%s:%g" .Release.Name .Release.Namespace .Values.clusterDomain .Values.postgresql.global.postgresql.service.ports.postgresql -}}
{{- end -}}
{{- end -}}

{{- define "redis.dns" -}}
{{- if (index .Values "redis-cluster").enabled -}}
{{- printf "redis+cluster://:%s@%s-redis-cluster-headless.%s.svc.%s:%g/0?pool_size=100&idle_timeout=180s&" (index .Values "redis-cluster").global.redis.password .Release.Name .Release.Namespace .Values.clusterDomain (index .Values "redis-cluster").service.ports.redis -}}
{{- end -}}
{{- end -}}

{{- define "redis.port" -}}
{{- if (index .Values "redis-cluster").enabled -}}
{{ (index .Values "redis-cluster").service.ports.redis }}
{{- end -}}
{{- end -}}

{{- define "redis.servicename" -}}
{{- if (index .Values "redis-cluster").enabled -}}
{{- printf "%s-redis-cluster-headless.%s.svc.%s" .Release.Name .Release.Namespace .Values.clusterDomain -}}
{{- end -}}
{{- end -}}

{{- define "gitea.default_domain" -}}
{{- printf "%s-http.%s.svc.%s" (include "gitea.fullname" .) .Release.Namespace .Values.clusterDomain -}}
{{- end -}}

{{- define "gitea.ldap_settings" -}}
{{- $idx := index . 0 }}
{{- $values := index . 1 }}

{{- if not (hasKey $values "bindDn") -}}
{{- $_ := set $values "bindDn" "" -}}
{{- end -}}

{{- if not (hasKey $values "bindPassword") -}}
{{- $_ := set $values "bindPassword" "" -}}
{{- end -}}

{{- $flags := list "notActive" "skipTlsVerify" "allowDeactivateAll" "synchronizeUsers" "attributesInBind" -}}
{{- range $key, $val := $values -}}
{{- if and (ne $key "enabled") (ne $key "existingSecret") -}}
{{- if eq $key "bindDn" -}}
{{- printf "--%s \"${GITEA_LDAP_BIND_DN_%d}\" " ($key | kebabcase) ($idx) -}}
{{- else if eq $key "bindPassword" -}}
{{- printf "--%s \"${GITEA_LDAP_PASSWORD_%d}\" " ($key | kebabcase) ($idx) -}}
{{- else if eq $key "port" -}}
{{- printf "--%s %d " $key ($val | int) -}}
{{- else if has $key $flags -}}
{{- printf "--%s " ($key | kebabcase) -}}
{{- else -}}
{{- printf "--%s %s " ($key | kebabcase) ($val | squote) -}}
{{- end -}}
{{- end -}}
{{- end -}}
{{- end -}}

{{- define "gitea.oauth_settings" -}}
{{- $idx := index . 0 }}
{{- $values := index . 1 }}

{{- if not (hasKey $values "key") -}}
{{- $_ := set $values "key" (printf "${GITEA_OAUTH_KEY_%d}" $idx) -}}
{{- end -}}

{{- if not (hasKey $values "secret") -}}
{{- $_ := set $values "secret" (printf "${GITEA_OAUTH_SECRET_%d}" $idx) -}}
{{- end -}}

{{- range $key, $val := $values -}}
{{- if ne $key "existingSecret" -}}
{{- printf "--%s %s " ($key | kebabcase) ($val | quote) -}}
{{- end -}}
{{- end -}}
{{- end -}}

{{- define "gitea.public_protocol" -}}
{{- if and .Values.ingress.enabled (gt (len .Values.ingress.tls) 0) -}}
https
{{- else -}}
{{ .Values.gitea.config.server.PROTOCOL }}
{{- end -}}
{{- end -}}

{{- define "gitea.inline_configuration" -}}
  {{- include "gitea.inline_configuration.init" . -}}
  {{- include "gitea.inline_configuration.defaults" . -}}

  {{- $generals := list -}}
  {{- $inlines := dict -}}

  {{- range $key, $value := .Values.gitea.config  }}
    {{- if kindIs "map" $value }}
      {{- if gt (len $value) 0 }}
        {{- $section := default list (get $inlines $key) -}}
        {{- range $n_key, $n_value := $value }}
          {{- $section = append $section (printf "%s=%v" $n_key $n_value) -}}
        {{- end }}
        {{- $_ := set $inlines $key (join "\n" $section) -}}
      {{- end -}}
    {{- else }}
      {{- if or (eq $key "APP_NAME") (eq $key "RUN_USER") (eq $key "RUN_MODE") -}}
        {{- $generals = append $generals (printf "%s=%s" $key $value) -}}
      {{- else -}}
        {{- (printf "Key %s cannot be on top level of configuration" $key) | fail -}}
      {{- end -}}

    {{- end }}
  {{- end }}

  {{- $_ := set $inlines "_generals_" (join "\n" $generals) -}}
  {{- toYaml $inlines -}}
{{- end -}}

{{- define "gitea.inline_configuration.init" -}}
  {{- if not (hasKey .Values.gitea.config "cache") -}}
    {{- $_ := set .Values.gitea.config "cache" dict -}}
  {{- end -}}
  {{- if not (hasKey .Values.gitea.config "server") -}}
    {{- $_ := set .Values.gitea.config "server" dict -}}
  {{- end -}}
  {{- if not (hasKey .Values.gitea.config "metrics") -}}
    {{- $_ := set .Values.gitea.config "metrics" dict -}}
  {{- end -}}
  {{- if not (hasKey .Values.gitea.config "database") -}}
    {{- $_ := set .Values.gitea.config "database" dict -}}
  {{- end -}}
  {{- if not (hasKey .Values.gitea.config "security") -}}
    {{- $_ := set .Values.gitea.config "security" dict -}}
  {{- end -}}
  {{- if not .Values.gitea.config.repository -}}
    {{- $_ := set .Values.gitea.config "repository" dict -}}
  {{- end -}}
  {{- if not (hasKey .Values.gitea.config "oauth2") -}}
    {{- $_ := set .Values.gitea.config "oauth2" dict -}}
  {{- end -}}
  {{- if not (hasKey .Values.gitea.config "session") -}}
    {{- $_ := set .Values.gitea.config "session" dict -}}
  {{- end -}}
  {{- if not (hasKey .Values.gitea.config "queue") -}}
    {{- $_ := set .Values.gitea.config "queue" dict -}}
  {{- end -}}
  {{- if not (hasKey .Values.gitea.config "queue.issue_indexer") -}}
    {{- $_ := set .Values.gitea.config "queue.issue_indexer" dict -}}
  {{- end -}}
  {{- if not (hasKey .Values.gitea.config "indexer") -}}
    {{- $_ := set .Values.gitea.config "indexer" dict -}}
  {{- end -}}
{{- end -}}

{{- define "gitea.inline_configuration.defaults" -}}
  {{- include "gitea.inline_configuration.defaults.server" . -}}
  {{- include "gitea.inline_configuration.defaults.database" . -}}

  {{- if not .Values.gitea.config.repository.ROOT -}}
    {{- $_ := set .Values.gitea.config.repository "ROOT" "/data/git/gitea-repositories" -}}
  {{- end -}}
  {{- if not .Values.gitea.config.security.INSTALL_LOCK -}}
    {{- $_ := set .Values.gitea.config.security "INSTALL_LOCK" "true" -}}
  {{- end -}}
  {{- if not (hasKey .Values.gitea.config.metrics "ENABLED") -}}
    {{- $_ := set .Values.gitea.config.metrics "ENABLED" .Values.gitea.metrics.enabled -}}
  {{- end -}}
  {{- /* redis queue */ -}}
  {{- if (index .Values "redis-cluster").enabled -}}
    {{- $_ := set .Values.gitea.config.queue "TYPE" "redis" -}}
    {{- $_ := set .Values.gitea.config.queue "CONN_STR" (include "redis.dns" .) -}}
    {{- $_ := set .Values.gitea.config.session "PROVIDER" "redis" -}}
    {{- $_ := set .Values.gitea.config.session "PROVIDER_CONFIG" (include "redis.dns" .) -}}
    {{- $_ := set .Values.gitea.config.cache "ADAPTER" "redis" -}}
    {{- $_ := set .Values.gitea.config.cache "HOST" (include "redis.dns" .) -}}
  {{- else -}}
    {{- if not (get .Values.gitea.config.session "PROVIDER") -}}
      {{- $_ := set .Values.gitea.config.session "PROVIDER" "memory" -}}
    {{- end -}}
    {{- if not (get .Values.gitea.config.session "PROVIDER_CONFIG") -}}
      {{- $_ := set .Values.gitea.config.session "PROVIDER_CONFIG" "" -}}
    {{- end -}}
    {{- if not (get .Values.gitea.config.queue "TYPE") -}}
      {{- $_ := set .Values.gitea.config.queue "TYPE" "level" -}}
    {{- end -}}
    {{- if not (get .Values.gitea.config.queue "CONN_STR") -}}
      {{- $_ := set .Values.gitea.config.queue "CONN_STR" "" -}}
    {{- end -}}
    {{- if not (get .Values.gitea.config.cache "ADAPTER") -}}
      {{- $_ := set .Values.gitea.config.cache "ADAPTER" "memory" -}}
    {{- end -}}
    {{- if not (get .Values.gitea.config.cache "HOST") -}}
      {{- $_ := set .Values.gitea.config.cache "HOST" "" -}}
    {{- end -}}
  {{- end -}}
  {{- if not .Values.gitea.config.indexer.ISSUE_INDEXER_TYPE -}}
     {{- $_ := set .Values.gitea.config.indexer "ISSUE_INDEXER_TYPE" "db" -}}
  {{- end -}}
{{- end -}}

{{- define "gitea.inline_configuration.defaults.server" -}}
  {{- if not (hasKey .Values.gitea.config.server "HTTP_PORT") -}}
    {{- $_ := set .Values.gitea.config.server "HTTP_PORT" .Values.service.http.port -}}
  {{- end -}}
  {{- if not .Values.gitea.config.server.PROTOCOL -}}
    {{- $_ := set .Values.gitea.config.server "PROTOCOL" "http" -}}
  {{- end -}}
  {{- if not (.Values.gitea.config.server.DOMAIN) -}}
    {{- if gt (len .Values.ingress.hosts) 0 -}}
      {{- $_ := set .Values.gitea.config.server "DOMAIN" ( tpl (index .Values.ingress.hosts 0).host $) -}}
    {{- else -}}
      {{- $_ := set .Values.gitea.config.server "DOMAIN" (include "gitea.default_domain" .) -}}
    {{- end -}}
  {{- end -}}
  {{- if not .Values.gitea.config.server.ROOT_URL -}}
    {{- $_ := set .Values.gitea.config.server "ROOT_URL" (printf "%s://%s" (include "gitea.public_protocol" .) .Values.gitea.config.server.DOMAIN) -}}
  {{- end -}}
  {{- if not .Values.gitea.config.server.SSH_DOMAIN -}}
    {{- $_ := set .Values.gitea.config.server "SSH_DOMAIN" .Values.gitea.config.server.DOMAIN -}}
  {{- end -}}
  {{- if not .Values.gitea.config.server.SSH_PORT -}}
    {{- $_ := set .Values.gitea.config.server "SSH_PORT" .Values.service.ssh.port -}}
  {{- end -}}
  {{- if not (hasKey .Values.gitea.config.server "SSH_LISTEN_PORT") -}}
    {{- if not .Values.image.rootless -}}
      {{- $_ := set .Values.gitea.config.server "SSH_LISTEN_PORT" .Values.gitea.config.server.SSH_PORT -}}
    {{- else -}}
      {{- $_ := set .Values.gitea.config.server "SSH_LISTEN_PORT" "2222" -}}
    {{- end -}}
  {{- end -}}
  {{- if not (hasKey .Values.gitea.config.server "START_SSH_SERVER") -}}
    {{- if .Values.image.rootless -}}
      {{- $_ := set .Values.gitea.config.server "START_SSH_SERVER" "true" -}}
    {{- end -}}
  {{- end -}}
  {{- if not (hasKey .Values.gitea.config.server "APP_DATA_PATH") -}}
    {{- $_ := set .Values.gitea.config.server "APP_DATA_PATH" "/data" -}}
  {{- end -}}
  {{- if not (hasKey .Values.gitea.config.server "ENABLE_PPROF") -}}
    {{- $_ := set .Values.gitea.config.server "ENABLE_PPROF" false -}}
  {{- end -}}
{{- end -}}

{{- define "gitea.inline_configuration.defaults.database" -}}
  {{- if (index .Values "postgresql-ha" "enabled") -}}
    {{- $_ := set .Values.gitea.config.database "DB_TYPE"   "postgres" -}}
    {{- if not (.Values.gitea.config.database.HOST) -}}
      {{- $_ := set .Values.gitea.config.database "HOST"      (include "postgresql-ha.dns" .) -}}
    {{- end -}}
    {{- $_ := set .Values.gitea.config.database "NAME"      (index .Values "postgresql-ha" "global" "postgresql" "database") -}}
    {{- $_ := set .Values.gitea.config.database "USER"      (index .Values "postgresql-ha" "global" "postgresql" "username") -}}
    {{- $_ := set .Values.gitea.config.database "PASSWD"    (index .Values "postgresql-ha" "global" "postgresql" "password") -}}
  {{- end -}}
  {{- if (index .Values "postgresql" "enabled") -}}
    {{- $_ := set .Values.gitea.config.database "DB_TYPE"   "postgres" -}}
    {{- if not (.Values.gitea.config.database.HOST) -}}
      {{- $_ := set .Values.gitea.config.database "HOST"      (include "postgresql.dns" .) -}}
    {{- end -}}
    {{- $_ := set .Values.gitea.config.database "NAME"      .Values.postgresql.global.postgresql.auth.database -}}
    {{- $_ := set .Values.gitea.config.database "USER"      .Values.postgresql.global.postgresql.auth.username -}}
    {{- $_ := set .Values.gitea.config.database "PASSWD"    .Values.postgresql.global.postgresql.auth.password -}}
  {{- end -}}
{{- end -}}

{{- define "gitea.init-additional-mounts" -}}
  {{- /* Honor the deprecated extraVolumeMounts variable when defined */ -}}
  {{- if gt (len .Values.extraInitVolumeMounts) 0 -}}
    {{- toYaml .Values.extraInitVolumeMounts -}}
  {{- else if gt (len .Values.extraVolumeMounts) 0 -}}
    {{- toYaml .Values.extraVolumeMounts -}}
  {{- end -}}
{{- end -}}

{{- define "gitea.container-additional-mounts" -}}
  {{- /* Honor the deprecated extraVolumeMounts variable when defined */ -}}
  {{- if gt (len .Values.extraContainerVolumeMounts) 0 -}}
    {{- toYaml .Values.extraContainerVolumeMounts -}}
  {{- else if gt (len .Values.extraVolumeMounts) 0 -}}
    {{- toYaml .Values.extraVolumeMounts -}}
  {{- end -}}
{{- end -}}

{{- define "gitea.gpg-key-secret-name" -}}
{{ default (printf "%s-gpg-key" (include "gitea.fullname" .)) .Values.signing.existingSecret }}
{{- end -}}

{{- define "gitea.serviceAccountName" -}}
{{ .Values.serviceAccount.name | default (include "gitea.fullname" .) }}
{{- end -}}
