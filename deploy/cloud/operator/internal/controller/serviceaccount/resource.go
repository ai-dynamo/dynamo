/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package serviceaccount

import (
	corev1 "k8s.io/api/core/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

const (
	kindServiceAccount = "ServiceAccount"
	apiGroupRBAC       = "rbac.authorization.k8s.io"
	apiGroupCore       = ""
)

// KubernetesDiscoveryResources contains all RBAC resources needed for Kubernetes endpoint discovery
type KubernetesDiscoveryResources struct {
	ServiceAccount *corev1.ServiceAccount
	Role           *rbacv1.Role
	RoleBinding    *rbacv1.RoleBinding
}

// GetKubernetesDiscoveryServiceAccount returns a ServiceAccount with associated Role and RoleBinding
// that allows listing/reading of endpoints resources in the specified namespace.
//
// Parameters:
//   - name: the name of the ServiceAccount to create
//   - namespace: the namespace to create the resources in
//
// Returns:
//   - KubernetesDiscoveryResources containing the ServiceAccount, Role, and RoleBinding
func GetKubernetesDiscoveryServiceAccount(name, namespace string) *KubernetesDiscoveryResources {
	return &KubernetesDiscoveryResources{
		ServiceAccount: getServiceAccount(name, namespace),
		Role:           getEndpointsDiscoveryRole(name, namespace),
		RoleBinding:    getRoleBinding(name, namespace),
	}
}

// getServiceAccount creates a ServiceAccount resource
func getServiceAccount(name, namespace string) *corev1.ServiceAccount {
	return &corev1.ServiceAccount{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
			Labels: map[string]string{
				"app.kubernetes.io/managed-by": "dynamo-operator",
				"app.kubernetes.io/component":  "rbac",
				"app.kubernetes.io/name":       name,
			},
		},
	}
}

// getEndpointsDiscoveryRole creates a Role that allows listing and reading endpoints
func getEndpointsDiscoveryRole(name, namespace string) *rbacv1.Role {
	roleName := name + "-role"
	return &rbacv1.Role{
		ObjectMeta: metav1.ObjectMeta{
			Name:      roleName,
			Namespace: namespace,
			Labels: map[string]string{
				"app.kubernetes.io/managed-by": "dynamo-operator",
				"app.kubernetes.io/component":  "rbac",
				"app.kubernetes.io/name":       name,
			},
		},
		Rules: []rbacv1.PolicyRule{
			{
				APIGroups: []string{apiGroupCore},
				Resources: []string{"endpoints"},
				Verbs:     []string{"get", "list", "watch"},
			},
			{
				APIGroups: []string{"discovery.k8s.io"},
				Resources: []string{"endpointslices"},
				Verbs:     []string{"get", "list", "watch"},
			},
		},
	}
}

// getRoleBinding creates a RoleBinding that binds the ServiceAccount to the Role
func getRoleBinding(name, namespace string) *rbacv1.RoleBinding {
	roleName := name + "-role"
	bindingName := name + "-binding"
	return &rbacv1.RoleBinding{
		ObjectMeta: metav1.ObjectMeta{
			Name:      bindingName,
			Namespace: namespace,
			Labels: map[string]string{
				"app.kubernetes.io/managed-by": "dynamo-operator",
				"app.kubernetes.io/component":  "rbac",
				"app.kubernetes.io/name":       name,
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
