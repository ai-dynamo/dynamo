/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package discovery

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"

	corev1 "k8s.io/api/core/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

const (
	kindServiceAccount = "ServiceAccount"
	apiGroupRBAC       = "rbac.authorization.k8s.io"
	apiGroupCore       = ""
	apiGroupNvidia     = "nvidia.com"
	maxLabelValueLen   = 63
	hashLen            = 8
)

func GetK8sDiscoveryServiceAccountName(dgdName string) string {
	return fmt.Sprintf("%s-k8s-service-discovery", dgdName)
}

func GetK8sDiscoveryServiceAccount(dgdName string, namespace string) *corev1.ServiceAccount {
	name := GetK8sDiscoveryServiceAccountName(dgdName)
	labelValue := getK8sDiscoveryLabelValue(name)
	return &corev1.ServiceAccount{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
			Labels: map[string]string{
				"app.kubernetes.io/managed-by": "dynamo-operator",
				"app.kubernetes.io/component":  "rbac",
				"app.kubernetes.io/name":       labelValue,
			},
		},
	}
}

func GetK8sDiscoveryRole(dgdName string, namespace string) *rbacv1.Role {
	name := GetK8sDiscoveryServiceAccountName(dgdName)
	labelValue := getK8sDiscoveryLabelValue(name)
	roleName := name + "-role"
	return &rbacv1.Role{
		ObjectMeta: metav1.ObjectMeta{
			Name:      roleName,
			Namespace: namespace,
			Labels: map[string]string{
				"app.kubernetes.io/managed-by": "dynamo-operator",
				"app.kubernetes.io/component":  "rbac",
				"app.kubernetes.io/name":       labelValue,
			},
		},
		Rules: []rbacv1.PolicyRule{
			{
				APIGroups: []string{apiGroupCore},
				Resources: []string{"endpoints", "pods"},
				Verbs:     []string{"get", "list", "watch"},
			},
			{
				APIGroups: []string{"discovery.k8s.io"},
				Resources: []string{"endpointslices"},
				Verbs:     []string{"get", "list", "watch"},
			},
			{
				APIGroups: []string{apiGroupNvidia},
				Resources: []string{"dynamoworkermetadatas"},
				Verbs:     []string{"create", "get", "list", "watch", "update", "patch", "delete"},
			},
		},
	}
}

func GetK8sDiscoveryRoleBinding(dgdName, namespace string) *rbacv1.RoleBinding {
	name := GetK8sDiscoveryServiceAccountName(dgdName)
	labelValue := getK8sDiscoveryLabelValue(name)
	roleName := name + "-role"
	bindingName := name + "-binding"
	return &rbacv1.RoleBinding{
		ObjectMeta: metav1.ObjectMeta{
			Name:      bindingName,
			Namespace: namespace,
			Labels: map[string]string{
				"app.kubernetes.io/managed-by": "dynamo-operator",
				"app.kubernetes.io/component":  "rbac",
				"app.kubernetes.io/name":       labelValue,
			},
		},
		Subjects: []rbacv1.Subject{
			{
				Kind:      kindServiceAccount,
				Name:      name,
				Namespace: namespace,
			},
		},
		RoleRef: rbacv1.RoleRef{
			APIGroup: apiGroupRBAC,
			Kind:     "Role",
			Name:     roleName,
		},
	}
}

// getK8sDiscoveryLabelValue returns the app.kubernetes.io/name value shared by
// the discovery ServiceAccount, Role, and RoleBinding. It is derived from the
// SA name (not each resource's own name) so all three carry the same "app"
// identifier, and is capped to Kubernetes' 63-char label-value limit via
// deterministic SHA-256 suffix truncation when necessary.
func getK8sDiscoveryLabelValue(serviceAccountName string) string {
	if len(serviceAccountName) <= maxLabelValueLen {
		return serviceAccountName
	}

	sum := sha256.Sum256([]byte(serviceAccountName))
	hash := hex.EncodeToString(sum[:hashLen/2])
	keep := maxLabelValueLen - len(hash) - 1
	if keep <= 0 {
		return hash
	}

	return serviceAccountName[:keep] + "-" + hash
}
