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
Validation to prevent operator conflicts
Prevents all conflict scenarios:
1. Multiple cluster-wide operators (multiple cluster managers)
2. Namespace-restricted operator when cluster-wide exists (both would manage same resources)
3. Cluster-wide operator when namespace-restricted exist (both would manage same resources)
*/}}
{{- define "dynamo-operator.validateClusterWideInstallation" -}}
{{- $currentReleaseName := .Release.Name -}}

{{/* Check for existing namespace-restricted operators (only when installing cluster-wide) */}}
{{- if not .Values.namespaceRestriction.enabled -}}
  {{- $allRoles := lookup "rbac.authorization.k8s.io/v1" "Role" "" "" -}}
  {{- $namespaceRestrictedOperators := list -}}

  {{- if $allRoles -}}
    {{- range $role := $allRoles.items -}}
      {{- if and (contains "-dynamo-operator-" $role.metadata.name) (hasSuffix "-manager-role" $role.metadata.name) -}}
        {{- $namespaceRestrictedOperators = append $namespaceRestrictedOperators $role.metadata.namespace -}}
      {{- end -}}
    {{- end -}}
  {{- end -}}

  {{- if $namespaceRestrictedOperators -}}
    {{- fail (printf "VALIDATION ERROR: Cannot install cluster-wide Dynamo operator. Found existing namespace-restricted Dynamo operators in namespaces: %s. This would create resource conflicts as both the cluster-wide operator and namespace-restricted operators would manage the same DGDs/DCDs. Either:\n1. Use one of the existing namespace-restricted operators for your specific namespace, or\n2. Uninstall all existing namespace-restricted operators first, or\n3. Install this operator in namespace-restricted mode: --set dynamo-operator.namespaceRestriction.enabled=true" (join ", " ($namespaceRestrictedOperators | uniq))) -}}
  {{- end -}}
{{- end -}}

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

  {{- if .Values.namespaceRestriction.enabled -}}
    {{- if $existingOperatorNamespace -}}
      {{- fail (printf "VALIDATION ERROR: Found existing cluster-wide Dynamo operator from release '%s' in namespace '%s' (ClusterRole: %s). Cannot install namespace-restricted operator because the cluster-wide operator already manages resources in all namespaces, including the target namespace. This would create resource conflicts. Either:\n1. Use the existing cluster-wide operator, or\n2. Uninstall the existing cluster-wide operator first: %s" $existingOperatorRelease $existingOperatorNamespace $existingOperatorRoleName $uninstallCmd) -}}
    {{- else -}}
      {{- fail (printf "VALIDATION ERROR: Found existing cluster-wide Dynamo operator from release '%s' (ClusterRole: %s). Cannot install namespace-restricted operator because the cluster-wide operator already manages resources in all namespaces, including the target namespace. This would create resource conflicts. Either:\n1. Use the existing cluster-wide operator, or\n2. Uninstall the existing cluster-wide operator first: %s" $existingOperatorRelease $existingOperatorRoleName $uninstallCmd) -}}
    {{- end -}}
  {{- else -}}
    {{- if $existingOperatorNamespace -}}
      {{- fail (printf "VALIDATION ERROR: Found existing cluster-wide Dynamo operator from release '%s' in namespace '%s' (ClusterRole: %s). Only one cluster-wide Dynamo operator should be deployed per cluster. Either:\n1. Use the existing cluster-wide operator (no need to install another), or\n2. Uninstall the existing cluster-wide operator first: %s" $existingOperatorRelease $existingOperatorNamespace $existingOperatorRoleName $uninstallCmd) -}}
    {{- else -}}
      {{- fail (printf "VALIDATION ERROR: Found existing cluster-wide Dynamo operator from release '%s' (ClusterRole: %s). Only one cluster-wide Dynamo operator should be deployed per cluster. Either:\n1. Use the existing cluster-wide operator (no need to install another), or\n2. Uninstall the existing cluster-wide operator first: %s" $existingOperatorRelease $existingOperatorRoleName $uninstallCmd) -}}
    {{- end -}}
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
