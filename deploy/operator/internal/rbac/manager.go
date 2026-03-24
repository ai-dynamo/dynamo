/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package rbac

import (
	"context"
	"fmt"

	corev1 "k8s.io/api/core/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

const (
	// RBAC resource kind constants
	kindClusterRole    = "ClusterRole"
	kindServiceAccount = "ServiceAccount"
	apiGroupRBAC       = "rbac.authorization.k8s.io"
)

// Manager handles dynamic RBAC creation for cluster-wide operator installations.
type Manager struct {
	client client.Client
}

// NewManager creates a new RBAC manager.
func NewManager(client client.Client) *Manager {
	return &Manager{client: client}
}

// needsRoleRefRecreate checks if the RoleRef has changed, which requires
// deleting and recreating the RoleBinding since RoleRef is immutable.
func needsRoleRefRecreate(existing *rbacv1.RoleBinding, clusterRoleName string) bool {
	return existing.RoleRef.Name != clusterRoleName ||
		existing.RoleRef.Kind != kindClusterRole ||
		existing.RoleRef.APIGroup != apiGroupRBAC
}

// needsSubjectUpdate checks if the Subjects field needs updating.
// Subjects are mutable so they can be updated in-place.
func needsSubjectUpdate(existing *rbacv1.RoleBinding, serviceAccountName, targetNamespace string) bool {
	return len(existing.Subjects) != 1 ||
		existing.Subjects[0].Kind != kindServiceAccount ||
		existing.Subjects[0].Name != serviceAccountName ||
		existing.Subjects[0].Namespace != targetNamespace
}

// EnsureServiceAccountWithRBAC creates or updates a ServiceAccount and RoleBinding
// in the target namespace. This should ONLY be called in cluster-wide mode.
//
// In cluster-wide mode, the operator dynamically creates:
//   - ServiceAccount in the target namespace
//   - RoleBinding in the target namespace that binds the SA to a ClusterRole
//
// The ClusterRole must already exist (created by Helm).
//
// Parameters:
//   - ctx: context
//   - targetNamespace: namespace to create RBAC resources in
//   - serviceAccountName: name of the ServiceAccount to create
//   - clusterRoleName: name of the ClusterRole to bind to (must exist)
func (m *Manager) EnsureServiceAccountWithRBAC(
	ctx context.Context,
	targetNamespace string,
	serviceAccountName string,
	clusterRoleName string,
) error {
	logger := log.FromContext(ctx)

	if targetNamespace == "" {
		return fmt.Errorf("target namespace is required")
	}
	if serviceAccountName == "" {
		return fmt.Errorf("service account name is required")
	}
	if clusterRoleName == "" {
		return fmt.Errorf("cluster role name is required")
	}

	// Verify ClusterRole exists before creating RoleBinding
	clusterRole := &rbacv1.ClusterRole{}
	if err := m.client.Get(ctx, client.ObjectKey{Name: clusterRoleName}, clusterRole); err != nil {
		if apierrors.IsNotFound(err) {
			return fmt.Errorf("cluster role %q does not exist: ensure it is created by Helm before deploying components", clusterRoleName)
		}
		return fmt.Errorf("failed to verify cluster role %q: %w", clusterRoleName, err)
	}
	logger.V(1).Info("ClusterRole verified",
		"clusterRole", clusterRoleName,
		"rules", len(clusterRole.Rules))

	// Create/update ServiceAccount
	sa := &corev1.ServiceAccount{
		ObjectMeta: metav1.ObjectMeta{
			Name:      serviceAccountName,
			Namespace: targetNamespace,
			Labels: map[string]string{
				"app.kubernetes.io/managed-by": "dynamo-operator",
				"app.kubernetes.io/component":  "rbac",
				"app.kubernetes.io/name":       serviceAccountName,
			},
		},
	}

	if err := m.client.Get(ctx, client.ObjectKeyFromObject(sa), sa); err != nil {
		if !apierrors.IsNotFound(err) {
			return fmt.Errorf("failed to get service account: %w", err)
		}
		// ServiceAccount doesn't exist, create it
		if err := m.client.Create(ctx, sa); err != nil {
			return fmt.Errorf("failed to create service account: %w", err)
		}
		logger.V(1).Info("ServiceAccount created",
			"serviceAccount", serviceAccountName,
			"namespace", targetNamespace)
	} else {
		logger.V(1).Info("ServiceAccount already exists",
			"serviceAccount", serviceAccountName,
			"namespace", targetNamespace)
	}

	// Create/update RoleBinding
	roleBindingName := fmt.Sprintf("%s-binding", serviceAccountName)
	rb := &rbacv1.RoleBinding{
		ObjectMeta: metav1.ObjectMeta{
			Name:      roleBindingName,
			Namespace: targetNamespace,
			Labels: map[string]string{
				"app.kubernetes.io/managed-by": "dynamo-operator",
				"app.kubernetes.io/component":  "rbac",
				"app.kubernetes.io/name":       serviceAccountName,
			},
		},
		Subjects: []rbacv1.Subject{{
			Kind:      kindServiceAccount,
			Name:      serviceAccountName,
			Namespace: targetNamespace,
		}},
		RoleRef: rbacv1.RoleRef{
			APIGroup: apiGroupRBAC,
			Kind:     kindClusterRole,
			Name:     clusterRoleName,
		},
	}

	existingRB := &rbacv1.RoleBinding{}
	if err := m.client.Get(ctx, client.ObjectKeyFromObject(rb), existingRB); err != nil {
		if !apierrors.IsNotFound(err) {
			return fmt.Errorf("failed to get role binding: %w", err)
		}
		// RoleBinding doesn't exist, create it
		if err := m.client.Create(ctx, rb); err != nil {
			return fmt.Errorf("failed to create role binding: %w", err)
		}
		logger.V(1).Info("RoleBinding created",
			"roleBinding", roleBindingName,
			"clusterRole", clusterRoleName,
			"namespace", targetNamespace)
	} else {
		// RoleBinding exists, check if it needs updating
		needsRecreate := needsRoleRefRecreate(existingRB, clusterRoleName)
		needsUpdate := needsSubjectUpdate(existingRB, serviceAccountName, targetNamespace)

		if needsRecreate {
			// RoleRef is immutable, so delete and recreate the RoleBinding
			if err := m.client.Delete(ctx, existingRB); err != nil {
				return fmt.Errorf("failed to delete role binding for recreation: %w", err)
			}
			logger.V(1).Info("RoleBinding deleted for recreation due to RoleRef change",
				"roleBinding", roleBindingName,
				"oldClusterRole", existingRB.RoleRef.Name,
				"newClusterRole", clusterRoleName,
				"namespace", targetNamespace)

			// Recreate with new RoleRef
			if err := m.client.Create(ctx, rb); err != nil {
				return fmt.Errorf("failed to recreate role binding: %w", err)
			}
			logger.V(1).Info("RoleBinding recreated",
				"roleBinding", roleBindingName,
				"clusterRole", clusterRoleName,
				"namespace", targetNamespace)
		} else if needsUpdate {
			// Only Subjects changed, can update in-place
			existingRB.Subjects = rb.Subjects
			if err := m.client.Update(ctx, existingRB); err != nil {
				return fmt.Errorf("failed to update role binding: %w", err)
			}
			logger.V(1).Info("RoleBinding subjects updated",
				"roleBinding", roleBindingName,
				"namespace", targetNamespace)
		} else {
			logger.V(1).Info("RoleBinding already up-to-date",
				"roleBinding", roleBindingName,
				"clusterRole", clusterRoleName,
				"namespace", targetNamespace)
		}
	}

	return nil
}
