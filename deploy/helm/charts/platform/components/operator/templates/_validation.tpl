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

{{/* Validate the explicit, cluster-wide API ownership contract. */}}
{{- define "dynamo-operator.validateAPIOwnership" -}}
{{- if and .Values.upgradeCRD (not .Values.apiOwnership.enabled) -}}
  {{- fail "VALIDATION ERROR: upgradeCRD=true requires apiOwnership.enabled=true. A non-owner must not install or update cluster-wide CRD schemas." -}}
{{- end -}}

{{- if and (not .Values.namespaceRestriction.enabled) (not .Values.apiOwnership.enabled) -}}
  {{- fail "VALIDATION ERROR: A cluster-wide operator must set apiOwnership.enabled=true. API non-owners are supported only for namespace-restricted installations." -}}
{{- end -}}

{{- if not (empty .Values.webhook.namespaceSelector) -}}
  {{- fail "VALIDATION ERROR: webhook.namespaceSelector must be empty. The single API owner must serve admission cluster-wide so it stays aligned with the cluster-wide CRD schema and conversion implementation." -}}
{{- end -}}

{{- if .Values.apiOwnership.enabled -}}
  {{- $ownerRoleName := include "dynamo-operator.apiOwnerRoleName" . -}}
  {{- $existingOwner := lookup "rbac.authorization.k8s.io/v1" "ClusterRole" "" $ownerRoleName -}}
  {{- if $existingOwner -}}
    {{- $annotations := $existingOwner.metadata.annotations | default dict -}}
    {{- $existingRelease := index $annotations "meta.helm.sh/release-name" | default "unknown" -}}
    {{- $existingNamespace := index $annotations "meta.helm.sh/release-namespace" | default "unknown" -}}
    {{- if or (ne $existingRelease .Release.Name) (ne $existingNamespace .Release.Namespace) -}}
      {{- fail (printf "VALIDATION ERROR: API ownership is already held by Helm release '%s' in namespace '%s' (ClusterRole: %s). Exactly one operator installation in the cluster may set apiOwnership.enabled=true. Running parallel operators at the same version is strongly recommended. If versions differ, the owner must be one and only one installation running the newest version." $existingRelease $existingNamespace $ownerRoleName) -}}
    {{- end -}}
  {{- end -}}
{{- end -}}
{{- end -}}

{{/* Validation to prevent overlapping cluster-wide reconciliation. */}}
{{- define "dynamo-operator.validateClusterWideInstallation" -}}
{{- $currentReleaseName := .Release.Name -}}

{{/* Check for existing ClusterRoles that would indicate other cluster-wide installations */}}
{{- $existingClusterRoles := lookup "rbac.authorization.k8s.io/v1" "ClusterRole" "" "" -}}
{{- $foundExistingClusterWideOperator := false -}}
{{- $existingOperatorRelease := "" -}}
{{- $existingOperatorRoleName := "" -}}
{{- $existingOperatorNamespace := "" -}}

{{- if $existingClusterRoles -}}
  {{- range $cr := $existingClusterRoles.items -}}
    {{- if and (contains "-dynamo-operator-" $cr.metadata.name) (hasSuffix "-manager-role" $cr.metadata.name) -}}
      {{- $currentRoleName := printf "%s-dynamo-operator-manager-role" $currentReleaseName -}}
      {{- if ne $cr.metadata.name $currentRoleName -}}
        {{- $foundExistingClusterWideOperator = true -}}
        {{- $existingOperatorRoleName = $cr.metadata.name -}}
        {{- if $cr.metadata.labels -}}
          {{- if $cr.metadata.labels.release -}}
            {{- $existingOperatorRelease = $cr.metadata.labels.release -}}
          {{- else if index $cr.metadata.labels "app.kubernetes.io/instance" -}}
            {{- $existingOperatorRelease = index $cr.metadata.labels "app.kubernetes.io/instance" -}}
          {{- end -}}
        {{- end -}}

        {{/* Find the namespace by looking at ClusterRoleBinding subjects */}}
        {{- $clusterRoleBindings := lookup "rbac.authorization.k8s.io/v1" "ClusterRoleBinding" "" "" -}}
        {{- if $clusterRoleBindings -}}
          {{- range $crb := $clusterRoleBindings.items -}}
            {{- if eq $crb.roleRef.name $cr.metadata.name -}}
              {{- range $subject := $crb.subjects -}}
                {{- if and (eq $subject.kind "ServiceAccount") $subject.namespace -}}
                  {{- $existingOperatorNamespace = $subject.namespace -}}
                {{- end -}}
              {{- end -}}
            {{- end -}}
          {{- end -}}
        {{- end -}}
      {{- end -}}
    {{- end -}}
  {{- end -}}
{{- end -}}

{{- if $foundExistingClusterWideOperator -}}
  {{- $uninstallCmd := printf "helm uninstall %s" $existingOperatorRelease -}}
  {{- if $existingOperatorNamespace -}}
    {{- $uninstallCmd = printf "helm uninstall %s -n %s" $existingOperatorRelease $existingOperatorNamespace -}}
  {{- end -}}

  {{- if $existingOperatorNamespace -}}
    {{- fail (printf "VALIDATION ERROR: Found existing cluster-wide Dynamo operator from release '%s' in namespace '%s' (ClusterRole: %s). Cluster-wide and namespace-restricted operators must not be mixed because the cluster-wide reconcilers do not exclude namespaces managed by other operators. Uninstall the existing cluster-wide operator first: %s" $existingOperatorRelease $existingOperatorNamespace $existingOperatorRoleName $uninstallCmd) -}}
  {{- else -}}
    {{- fail (printf "VALIDATION ERROR: Found existing cluster-wide Dynamo operator from release '%s' (ClusterRole: %s). Cluster-wide and namespace-restricted operators must not be mixed because the cluster-wide reconcilers do not exclude namespaces managed by other operators. Uninstall the existing cluster-wide operator first: %s" $existingOperatorRelease $existingOperatorRoleName $uninstallCmd) -}}
  {{- end -}}
{{- end -}}

{{/* Additional validation for cluster-wide mode */}}
{{- if not .Values.namespaceRestriction.enabled -}}
  {{/* Warn if using different leader election IDs */}}
  {{- $leaderElectionId := default "dynamo.nvidia.com" .Values.controllerManager.leaderElection.id -}}
  {{- if ne $leaderElectionId "dynamo.nvidia.com" -}}
    {{- fail (printf "VALIDATION WARNING: Using custom leader election ID '%s' in cluster-wide mode. For proper coordination, all cluster-wide Dynamo operators should use the SAME leader election ID. Different IDs will allow multiple leaders simultaneously (split-brain scenario)." $leaderElectionId) -}}
  {{- end -}}
{{- end -}}
{{- end -}}

{{/*
Validation for configuration consistency
*/}}
{{- define "dynamo-operator.validateConfiguration" -}}
{{/* Validate leader election namespace setting */}}
{{- if and (not .Values.namespaceRestriction.enabled) .Values.controllerManager.leaderElection.namespace -}}
  {{- if eq .Values.controllerManager.leaderElection.namespace .Release.Namespace -}}
    {{- printf "\nWARNING: Leader election namespace is set to the same as release namespace (%s) in cluster-wide mode. This may prevent proper coordination between multiple releases. Consider using 'kube-system' or leaving empty for default.\n" .Release.Namespace | fail -}}
  {{- end -}}
{{- end -}}
{{- end -}}

{{/*
Validation for discoveryBackend configuration
*/}}
{{- define "dynamo-operator.validateDiscoveryBackend" -}}
{{- $discoveryBackend := .Values.discoveryBackend -}}
{{- if and (ne $discoveryBackend "kubernetes") (ne $discoveryBackend "etcd") -}}
  {{- fail (printf "VALIDATION ERROR: discoveryBackend must be 'kubernetes' (default) or 'etcd'. Got: '%s'" $discoveryBackend) -}}
{{- end -}}
{{- end -}}
