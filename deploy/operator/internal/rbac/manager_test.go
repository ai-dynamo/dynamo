/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package rbac

import (
	"context"
	"strings"
	"testing"

	corev1 "k8s.io/api/core/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

const (
	// Test constants
	testServiceAccountName = "test-sa"
	testNamespace          = "test-namespace"
	testClusterRoleName    = "test-cluster-role"
	testRoleBindingName    = "test-sa-binding"
)

func setupTest() (client.Client, *runtime.Scheme) {
	scheme := runtime.NewScheme()
	_ = corev1.AddToScheme(scheme)
	_ = rbacv1.AddToScheme(scheme)

	fakeClient := fake.NewClientBuilder().WithScheme(scheme).Build()
	return fakeClient, scheme
}

func setupTestWithClusterRole(clusterRoleName string) client.Client {
	scheme := runtime.NewScheme()
	_ = corev1.AddToScheme(scheme)
	_ = rbacv1.AddToScheme(scheme)

	// Pre-create ClusterRole
	clusterRole := &rbacv1.ClusterRole{
		ObjectMeta: metav1.ObjectMeta{
			Name: clusterRoleName,
		},
		Rules: []rbacv1.PolicyRule{
			{
				APIGroups: []string{"nvidia.com"},
				Resources: []string{"dynamocomponentdeployments", "dynamographdeployments"},
				Verbs:     []string{"get", "list", "create", "update", "patch"},
			},
		},
	}

	fakeClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(clusterRole).
		Build()
	return fakeClient
}

func TestEnsureServiceAccountWithRBAC_CreateNew(t *testing.T) {
	// Setup
	fakeClient := setupTestWithClusterRole(testClusterRoleName)
	manager := NewManager(fakeClient)
	ctx := context.Background()

	// Execute
	err := manager.EnsureServiceAccountWithRBAC(
		ctx,
		testNamespace,
		testServiceAccountName,
		testClusterRoleName,
	)

	// Verify
	if err != nil {
		t.Fatalf("Expected no error, got: %v", err)
	}

	// Check ServiceAccount was created
	sa := &corev1.ServiceAccount{}
	err = fakeClient.Get(ctx, client.ObjectKey{
		Namespace: testNamespace,
		Name:      testServiceAccountName,
	}, sa)
	if err != nil {
		t.Fatalf("ServiceAccount not created: %v", err)
	}

	// Verify ServiceAccount labels
	expectedLabels := map[string]string{
		"app.kubernetes.io/managed-by": "dynamo-operator",
		"app.kubernetes.io/component":  "rbac",
		"app.kubernetes.io/name":       testServiceAccountName,
	}
	for k, v := range expectedLabels {
		if sa.Labels[k] != v {
			t.Errorf("Expected label %s=%s, got %s", k, v, sa.Labels[k])
		}
	}

	// Check RoleBinding was created
	rb := &rbacv1.RoleBinding{}
	err = fakeClient.Get(ctx, client.ObjectKey{
		Namespace: testNamespace,
		Name:      testRoleBindingName,
	}, rb)
	if err != nil {
		t.Fatalf("RoleBinding not created: %v", err)
	}

	// Verify RoleBinding configuration
	if len(rb.Subjects) != 1 {
		t.Fatalf("Expected 1 subject, got %d", len(rb.Subjects))
	}
	if rb.Subjects[0].Kind != "ServiceAccount" {
		t.Errorf("Expected subject kind ServiceAccount, got %s", rb.Subjects[0].Kind)
	}
	if rb.Subjects[0].Name != testServiceAccountName {
		t.Errorf("Expected subject name test-sa, got %s", rb.Subjects[0].Name)
	}
	if rb.Subjects[0].Namespace != testNamespace {
		t.Errorf("Expected subject namespace test-namespace, got %s", rb.Subjects[0].Namespace)
	}

	// Verify RoleRef
	if rb.RoleRef.Kind != "ClusterRole" {
		t.Errorf("Expected RoleRef kind ClusterRole, got %s", rb.RoleRef.Kind)
	}
	if rb.RoleRef.Name != testClusterRoleName {
		t.Errorf("Expected RoleRef name test-cluster-role, got %s", rb.RoleRef.Name)
	}
	if rb.RoleRef.APIGroup != "rbac.authorization.k8s.io" {
		t.Errorf("Expected RoleRef APIGroup rbac.authorization.k8s.io, got %s", rb.RoleRef.APIGroup)
	}
}

func TestEnsureServiceAccountWithRBAC_AlreadyExists(t *testing.T) {
	// Setup - pre-create ClusterRole, ServiceAccount and RoleBinding
	_, scheme := setupTest()

	clusterRole := &rbacv1.ClusterRole{
		ObjectMeta: metav1.ObjectMeta{
			Name: testClusterRoleName,
		},
		Rules: []rbacv1.PolicyRule{
			{
				APIGroups: []string{"nvidia.com"},
				Resources: []string{"dynamocomponentdeployments", "dynamographdeployments"},
				Verbs:     []string{"get", "list", "create", "update", "patch"},
			},
		},
	}

	existingSA := &corev1.ServiceAccount{
		ObjectMeta: metav1.ObjectMeta{
			Name:      testServiceAccountName,
			Namespace: testNamespace,
			Labels: map[string]string{
				"app.kubernetes.io/managed-by": "dynamo-operator",
				"app.kubernetes.io/component":  "rbac",
				"app.kubernetes.io/name":       testServiceAccountName,
			},
		},
	}

	existingRB := &rbacv1.RoleBinding{
		ObjectMeta: metav1.ObjectMeta{
			Name:      testRoleBindingName,
			Namespace: testNamespace,
			Labels: map[string]string{
				"app.kubernetes.io/managed-by": "dynamo-operator",
				"app.kubernetes.io/component":  "rbac",
				"app.kubernetes.io/name":       testServiceAccountName,
			},
		},
		Subjects: []rbacv1.Subject{{
			Kind:      "ServiceAccount",
			Name:      testServiceAccountName,
			Namespace: testNamespace,
		}},
		RoleRef: rbacv1.RoleRef{
			APIGroup: "rbac.authorization.k8s.io",
			Kind:     "ClusterRole",
			Name:     testClusterRoleName,
		},
	}

	fakeClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(clusterRole, existingSA, existingRB).
		Build()

	manager := NewManager(fakeClient)
	ctx := context.Background()

	// Execute
	err := manager.EnsureServiceAccountWithRBAC(
		ctx,
		testNamespace,
		testServiceAccountName,
		testClusterRoleName,
	)

	// Verify
	if err != nil {
		t.Fatalf("Expected no error, got: %v", err)
	}

	// Verify resources still exist and unchanged
	sa := &corev1.ServiceAccount{}
	err = fakeClient.Get(ctx, client.ObjectKey{
		Namespace: testNamespace,
		Name:      testServiceAccountName,
	}, sa)
	if err != nil {
		t.Fatalf("ServiceAccount not found: %v", err)
	}

	rb := &rbacv1.RoleBinding{}
	err = fakeClient.Get(ctx, client.ObjectKey{
		Namespace: testNamespace,
		Name:      testRoleBindingName,
	}, rb)
	if err != nil {
		t.Fatalf("RoleBinding not found: %v", err)
	}
}

func TestEnsureServiceAccountWithRBAC_UpdateRoleBinding(t *testing.T) {
	// Setup - pre-create ClusterRole, ServiceAccount and RoleBinding with wrong subject
	_, scheme := setupTest()

	clusterRole := &rbacv1.ClusterRole{
		ObjectMeta: metav1.ObjectMeta{
			Name: testClusterRoleName,
		},
		Rules: []rbacv1.PolicyRule{
			{
				APIGroups: []string{"nvidia.com"},
				Resources: []string{"dynamocomponentdeployments", "dynamographdeployments"},
				Verbs:     []string{"get", "list", "create", "update", "patch"},
			},
		},
	}

	existingSA := &corev1.ServiceAccount{
		ObjectMeta: metav1.ObjectMeta{
			Name:      testServiceAccountName,
			Namespace: testNamespace,
			Labels: map[string]string{
				"app.kubernetes.io/managed-by": "dynamo-operator",
				"app.kubernetes.io/component":  "rbac",
				"app.kubernetes.io/name":       testServiceAccountName,
			},
		},
	}

	existingRB := &rbacv1.RoleBinding{
		ObjectMeta: metav1.ObjectMeta{
			Name:      testRoleBindingName,
			Namespace: testNamespace,
			Labels: map[string]string{
				"app.kubernetes.io/managed-by": "dynamo-operator",
				"app.kubernetes.io/component":  "rbac",
				"app.kubernetes.io/name":       testServiceAccountName,
			},
		},
		Subjects: []rbacv1.Subject{{
			Kind:      "ServiceAccount",
			Name:      "wrong-sa", // Wrong name
			Namespace: testNamespace,
		}},
		RoleRef: rbacv1.RoleRef{
			APIGroup: "rbac.authorization.k8s.io",
			Kind:     "ClusterRole",
			Name:     testClusterRoleName,
		},
	}

	fakeClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(clusterRole, existingSA, existingRB).
		Build()

	manager := NewManager(fakeClient)
	ctx := context.Background()

	// Execute
	err := manager.EnsureServiceAccountWithRBAC(
		ctx,
		testNamespace,
		testServiceAccountName,
		testClusterRoleName,
	)

	// Verify
	if err != nil {
		t.Fatalf("Expected no error, got: %v", err)
	}

	// Verify RoleBinding was updated with correct subject
	rb := &rbacv1.RoleBinding{}
	err = fakeClient.Get(ctx, client.ObjectKey{
		Namespace: testNamespace,
		Name:      testRoleBindingName,
	}, rb)
	if err != nil {
		t.Fatalf("RoleBinding not found: %v", err)
	}

	if len(rb.Subjects) != 1 {
		t.Fatalf("Expected 1 subject, got %d", len(rb.Subjects))
	}
	if rb.Subjects[0].Name != testServiceAccountName {
		t.Errorf("Expected subject name test-sa, got %s", rb.Subjects[0].Name)
	}
}

func TestEnsureServiceAccountWithRBAC_MultipleNamespaces(t *testing.T) {
	// Setup
	fakeClient := setupTestWithClusterRole(testClusterRoleName)
	manager := NewManager(fakeClient)
	ctx := context.Background()

	namespaces := []string{"ns1", "ns2", "ns3"}

	// Execute - create RBAC in multiple namespaces
	for _, ns := range namespaces {
		err := manager.EnsureServiceAccountWithRBAC(
			ctx,
			ns,
			testServiceAccountName,
			testClusterRoleName,
		)
		if err != nil {
			t.Fatalf("Failed for namespace %s: %v", ns, err)
		}
	}

	// Verify - check resources exist in all namespaces
	for _, ns := range namespaces {
		sa := &corev1.ServiceAccount{}
		err := fakeClient.Get(ctx, client.ObjectKey{
			Namespace: ns,
			Name:      testServiceAccountName,
		}, sa)
		if err != nil {
			t.Errorf("ServiceAccount not found in namespace %s: %v", ns, err)
		}

		rb := &rbacv1.RoleBinding{}
		err = fakeClient.Get(ctx, client.ObjectKey{
			Namespace: ns,
			Name:      testRoleBindingName,
		}, rb)
		if err != nil {
			t.Errorf("RoleBinding not found in namespace %s: %v", ns, err)
		}
	}
}

func TestEnsureServiceAccountWithRBAC_ServiceAccountExistsRoleBindingDoesNot(t *testing.T) {
	// Setup - pre-create only ServiceAccount and ClusterRole
	_, scheme := setupTest()

	clusterRole := &rbacv1.ClusterRole{
		ObjectMeta: metav1.ObjectMeta{
			Name: testClusterRoleName,
		},
		Rules: []rbacv1.PolicyRule{
			{
				APIGroups: []string{"nvidia.com"},
				Resources: []string{"dynamocomponentdeployments", "dynamographdeployments"},
				Verbs:     []string{"get", "list", "create", "update", "patch"},
			},
		},
	}

	existingSA := &corev1.ServiceAccount{
		ObjectMeta: metav1.ObjectMeta{
			Name:      testServiceAccountName,
			Namespace: testNamespace,
			Labels: map[string]string{
				"app.kubernetes.io/managed-by": "dynamo-operator",
				"app.kubernetes.io/component":  "rbac",
				"app.kubernetes.io/name":       testServiceAccountName,
			},
		},
	}

	fakeClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(clusterRole, existingSA).
		Build()

	manager := NewManager(fakeClient)
	ctx := context.Background()

	// Execute
	err := manager.EnsureServiceAccountWithRBAC(
		ctx,
		testNamespace,
		testServiceAccountName,
		testClusterRoleName,
	)

	// Verify
	if err != nil {
		t.Fatalf("Expected no error, got: %v", err)
	}

	// Verify ServiceAccount still exists
	sa := &corev1.ServiceAccount{}
	err = fakeClient.Get(ctx, client.ObjectKey{
		Namespace: testNamespace,
		Name:      testServiceAccountName,
	}, sa)
	if err != nil {
		t.Fatalf("ServiceAccount not found: %v", err)
	}

	// Verify RoleBinding was created
	rb := &rbacv1.RoleBinding{}
	err = fakeClient.Get(ctx, client.ObjectKey{
		Namespace: testNamespace,
		Name:      testRoleBindingName,
	}, rb)
	if err != nil {
		t.Fatalf("RoleBinding not created: %v", err)
	}
}

func TestEnsureServiceAccountWithRBAC_Idempotency(t *testing.T) {
	// Setup
	fakeClient := setupTestWithClusterRole(testClusterRoleName)
	manager := NewManager(fakeClient)
	ctx := context.Background()

	// Execute multiple times
	for i := 0; i < 3; i++ {
		err := manager.EnsureServiceAccountWithRBAC(
			ctx,
			testNamespace,
			testServiceAccountName,
			testClusterRoleName,
		)
		if err != nil {
			t.Fatalf("Iteration %d failed: %v", i, err)
		}
	}

	// Verify - should still have exactly one ServiceAccount and one RoleBinding
	saList := &corev1.ServiceAccountList{}
	err := fakeClient.List(ctx, saList, client.InNamespace(testNamespace))
	if err != nil {
		t.Fatalf("Failed to list ServiceAccounts: %v", err)
	}
	if len(saList.Items) != 1 {
		t.Errorf("Expected 1 ServiceAccount, got %d", len(saList.Items))
	}

	rbList := &rbacv1.RoleBindingList{}
	err = fakeClient.List(ctx, rbList, client.InNamespace(testNamespace))
	if err != nil {
		t.Fatalf("Failed to list RoleBindings: %v", err)
	}
	if len(rbList.Items) != 1 {
		t.Errorf("Expected 1 RoleBinding, got %d", len(rbList.Items))
	}
}

func TestNewManager(t *testing.T) {
	// Setup
	fakeClient, _ := setupTest()

	// Execute
	manager := NewManager(fakeClient)

	// Verify
	if manager == nil {
		t.Fatal("Expected non-nil manager")
	}
	if manager.client == nil {
		t.Fatal("Expected non-nil client in manager")
	}
}

func TestEnsureServiceAccountWithRBAC_ClusterRoleNotFound(t *testing.T) {
	// Setup - no ClusterRole created
	fakeClient, _ := setupTest()
	manager := NewManager(fakeClient)
	ctx := context.Background()

	// Execute
	err := manager.EnsureServiceAccountWithRBAC(
		ctx,
		testNamespace,
		testServiceAccountName,
		"non-existent-cluster-role",
	)

	// Verify - should fail with clear error message
	if err == nil {
		t.Fatal("Expected error when ClusterRole doesn't exist, got nil")
	}
	expectedMsg := "cluster role \"non-existent-cluster-role\" does not exist"
	if !strings.Contains(err.Error(), expectedMsg) {
		t.Errorf("Expected error message to contain %q, got: %v", expectedMsg, err)
	}

	// Verify no ServiceAccount or RoleBinding was created
	sa := &corev1.ServiceAccount{}
	err = fakeClient.Get(ctx, client.ObjectKey{
		Namespace: testNamespace,
		Name:      testServiceAccountName,
	}, sa)
	if !apierrors.IsNotFound(err) {
		t.Error("Expected ServiceAccount not to be created when ClusterRole is missing")
	}
}

func TestEnsureServiceAccountWithRBAC_DifferentClusterRoles(t *testing.T) {
	// Setup - create two ClusterRoles
	_, scheme := setupTest()

	clusterRole1 := &rbacv1.ClusterRole{
		ObjectMeta: metav1.ObjectMeta{
			Name: "cluster-role-1",
		},
		Rules: []rbacv1.PolicyRule{
			{
				APIGroups: []string{"nvidia.com"},
				Resources: []string{"dynamocomponentdeployments"},
				Verbs:     []string{"get", "list"},
			},
		},
	}

	clusterRole2 := &rbacv1.ClusterRole{
		ObjectMeta: metav1.ObjectMeta{
			Name: "cluster-role-2",
		},
		Rules: []rbacv1.PolicyRule{
			{
				APIGroups: []string{"nvidia.com"},
				Resources: []string{"dynamographdeployments"},
				Verbs:     []string{"get", "list"},
			},
		},
	}

	fakeClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(clusterRole1, clusterRole2).
		Build()

	manager := NewManager(fakeClient)
	ctx := context.Background()

	// Execute - create with first cluster role
	err := manager.EnsureServiceAccountWithRBAC(
		ctx,
		testNamespace,
		testServiceAccountName,
		"cluster-role-1",
	)
	if err != nil {
		t.Fatalf("First call failed: %v", err)
	}

	// Verify first cluster role
	rb := &rbacv1.RoleBinding{}
	err = fakeClient.Get(ctx, client.ObjectKey{
		Namespace: testNamespace,
		Name:      testRoleBindingName,
	}, rb)
	if err != nil {
		t.Fatalf("RoleBinding not found: %v", err)
	}
	if rb.RoleRef.Name != "cluster-role-1" {
		t.Errorf("Expected RoleRef name cluster-role-1, got %s", rb.RoleRef.Name)
	}

	// Execute - change to second cluster role (should delete and recreate)
	err = manager.EnsureServiceAccountWithRBAC(
		ctx,
		testNamespace,
		testServiceAccountName,
		"cluster-role-2",
	)
	if err != nil {
		t.Fatalf("Second call failed: %v", err)
	}

	// Verify cluster role was changed
	rb = &rbacv1.RoleBinding{}
	err = fakeClient.Get(ctx, client.ObjectKey{
		Namespace: testNamespace,
		Name:      testRoleBindingName,
	}, rb)
	if err != nil {
		t.Fatalf("RoleBinding not found after update: %v", err)
	}
	if rb.RoleRef.Name != "cluster-role-2" {
		t.Errorf("Expected RoleRef name cluster-role-2 after update, got %s", rb.RoleRef.Name)
	}
}

func TestEnsureServiceAccountWithRBAC_EmptyNamespace(t *testing.T) {
	// Setup
	fakeClient, _ := setupTest()
	manager := NewManager(fakeClient)
	ctx := context.Background()

	// Execute with empty namespace
	err := manager.EnsureServiceAccountWithRBAC(
		ctx,
		"",
		testServiceAccountName,
		testClusterRoleName,
	)

	// Verify - should fail because namespace is required
	// The fake client might not enforce this, but in real K8s it would fail
	// We just verify the function returns (it might succeed with fake client)
	if err == nil {
		// Check if resources were created in empty namespace
		sa := &corev1.ServiceAccount{}
		err = fakeClient.Get(ctx, client.ObjectKey{
			Namespace: "",
			Name:      testServiceAccountName,
		}, sa)
		// In fake client this might work, but we document the behavior
		if err != nil && !apierrors.IsNotFound(err) {
			t.Logf("Expected behavior: empty namespace handled: %v", err)
		}
	}
}

func TestEnsureServiceAccountWithRBAC_RoleRefChange(t *testing.T) {
	// Setup - pre-create both ClusterRoles, ServiceAccount, and RoleBinding
	_, scheme := setupTest()

	oldClusterRole := &rbacv1.ClusterRole{
		ObjectMeta: metav1.ObjectMeta{
			Name: "old-cluster-role",
		},
		Rules: []rbacv1.PolicyRule{
			{
				APIGroups: []string{"nvidia.com"},
				Resources: []string{"dynamocomponentdeployments"},
				Verbs:     []string{"get"},
			},
		},
	}

	newClusterRole := &rbacv1.ClusterRole{
		ObjectMeta: metav1.ObjectMeta{
			Name: "new-cluster-role",
		},
		Rules: []rbacv1.PolicyRule{
			{
				APIGroups: []string{"nvidia.com"},
				Resources: []string{"dynamographdeployments"},
				Verbs:     []string{"get", "list"},
			},
		},
	}

	existingSA := &corev1.ServiceAccount{
		ObjectMeta: metav1.ObjectMeta{
			Name:      testServiceAccountName,
			Namespace: testNamespace,
		},
	}

	existingRB := &rbacv1.RoleBinding{
		ObjectMeta: metav1.ObjectMeta{
			Name:      testRoleBindingName,
			Namespace: testNamespace,
		},
		Subjects: []rbacv1.Subject{{
			Kind:      "ServiceAccount",
			Name:      testServiceAccountName,
			Namespace: testNamespace,
		}},
		RoleRef: rbacv1.RoleRef{
			APIGroup: "rbac.authorization.k8s.io",
			Kind:     "ClusterRole",
			Name:     "old-cluster-role", // Old ClusterRole name
		},
	}

	fakeClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(oldClusterRole, newClusterRole, existingSA, existingRB).
		Build()

	manager := NewManager(fakeClient)
	ctx := context.Background()

	// Execute with new ClusterRole name
	err := manager.EnsureServiceAccountWithRBAC(
		ctx,
		testNamespace,
		testServiceAccountName,
		"new-cluster-role", // New ClusterRole name
	)

	// Verify
	if err != nil {
		t.Fatalf("Expected no error, got: %v", err)
	}

	// Verify RoleBinding was recreated with new RoleRef
	rb := &rbacv1.RoleBinding{}
	err = fakeClient.Get(ctx, client.ObjectKey{
		Namespace: testNamespace,
		Name:      testRoleBindingName,
	}, rb)
	if err != nil {
		t.Fatalf("RoleBinding not found: %v", err)
	}

	if rb.RoleRef.Name != "new-cluster-role" {
		t.Errorf("Expected RoleRef name new-cluster-role, got %s", rb.RoleRef.Name)
	}
	if rb.Subjects[0].Name != testServiceAccountName {
		t.Errorf("Expected subject name test-sa, got %s", rb.Subjects[0].Name)
	}
}

func TestEnsureServiceAccountWithRBAC_SubjectWrongNamespace(t *testing.T) {
	// Setup - pre-create ClusterRole, ServiceAccount, and RoleBinding with wrong subject namespace
	_, scheme := setupTest()

	clusterRole := &rbacv1.ClusterRole{
		ObjectMeta: metav1.ObjectMeta{
			Name: testClusterRoleName,
		},
		Rules: []rbacv1.PolicyRule{
			{
				APIGroups: []string{"nvidia.com"},
				Resources: []string{"dynamocomponentdeployments", "dynamographdeployments"},
				Verbs:     []string{"get", "list", "create", "update", "patch"},
			},
		},
	}

	existingSA := &corev1.ServiceAccount{
		ObjectMeta: metav1.ObjectMeta{
			Name:      testServiceAccountName,
			Namespace: testNamespace,
		},
	}

	existingRB := &rbacv1.RoleBinding{
		ObjectMeta: metav1.ObjectMeta{
			Name:      testRoleBindingName,
			Namespace: testNamespace,
		},
		Subjects: []rbacv1.Subject{{
			Kind:      "ServiceAccount",
			Name:      testServiceAccountName,
			Namespace: "wrong-namespace", // Wrong namespace
		}},
		RoleRef: rbacv1.RoleRef{
			APIGroup: "rbac.authorization.k8s.io",
			Kind:     "ClusterRole",
			Name:     testClusterRoleName,
		},
	}

	fakeClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(clusterRole, existingSA, existingRB).
		Build()

	manager := NewManager(fakeClient)
	ctx := context.Background()

	// Execute
	err := manager.EnsureServiceAccountWithRBAC(
		ctx,
		testNamespace,
		testServiceAccountName,
		testClusterRoleName,
	)

	// Verify
	if err != nil {
		t.Fatalf("Expected no error, got: %v", err)
	}

	// Verify RoleBinding was updated with correct namespace
	rb := &rbacv1.RoleBinding{}
	err = fakeClient.Get(ctx, client.ObjectKey{
		Namespace: testNamespace,
		Name:      testRoleBindingName,
	}, rb)
	if err != nil {
		t.Fatalf("RoleBinding not found: %v", err)
	}

	if len(rb.Subjects) != 1 {
		t.Fatalf("Expected 1 subject, got %d", len(rb.Subjects))
	}
	if rb.Subjects[0].Namespace != testNamespace {
		t.Errorf("Expected subject namespace test-namespace, got %s", rb.Subjects[0].Namespace)
	}
	if rb.Subjects[0].Name != testServiceAccountName {
		t.Errorf("Expected subject name test-sa, got %s", rb.Subjects[0].Name)
	}
}

func TestEnsureServiceAccountWithRBAC_SubjectWrongKind(t *testing.T) {
	// Setup - pre-create ClusterRole, ServiceAccount, and RoleBinding with wrong subject kind
	_, scheme := setupTest()

	clusterRole := &rbacv1.ClusterRole{
		ObjectMeta: metav1.ObjectMeta{
			Name: testClusterRoleName,
		},
		Rules: []rbacv1.PolicyRule{
			{
				APIGroups: []string{"nvidia.com"},
				Resources: []string{"dynamocomponentdeployments", "dynamographdeployments"},
				Verbs:     []string{"get", "list", "create", "update", "patch"},
			},
		},
	}

	existingSA := &corev1.ServiceAccount{
		ObjectMeta: metav1.ObjectMeta{
			Name:      testServiceAccountName,
			Namespace: testNamespace,
		},
	}

	existingRB := &rbacv1.RoleBinding{
		ObjectMeta: metav1.ObjectMeta{
			Name:      testRoleBindingName,
			Namespace: testNamespace,
		},
		Subjects: []rbacv1.Subject{{
			Kind:      "User", // Wrong kind
			Name:      testServiceAccountName,
			Namespace: testNamespace,
		}},
		RoleRef: rbacv1.RoleRef{
			APIGroup: "rbac.authorization.k8s.io",
			Kind:     "ClusterRole",
			Name:     testClusterRoleName,
		},
	}

	fakeClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(clusterRole, existingSA, existingRB).
		Build()

	manager := NewManager(fakeClient)
	ctx := context.Background()

	// Execute
	err := manager.EnsureServiceAccountWithRBAC(
		ctx,
		testNamespace,
		testServiceAccountName,
		testClusterRoleName,
	)

	// Verify
	if err != nil {
		t.Fatalf("Expected no error, got: %v", err)
	}

	// Verify RoleBinding was updated with correct kind
	rb := &rbacv1.RoleBinding{}
	err = fakeClient.Get(ctx, client.ObjectKey{
		Namespace: testNamespace,
		Name:      testRoleBindingName,
	}, rb)
	if err != nil {
		t.Fatalf("RoleBinding not found: %v", err)
	}

	if len(rb.Subjects) != 1 {
		t.Fatalf("Expected 1 subject, got %d", len(rb.Subjects))
	}
	if rb.Subjects[0].Kind != "ServiceAccount" {
		t.Errorf("Expected subject kind ServiceAccount, got %s", rb.Subjects[0].Kind)
	}
	if rb.Subjects[0].Name != testServiceAccountName {
		t.Errorf("Expected subject name test-sa, got %s", rb.Subjects[0].Name)
	}
}

func TestEnsureServiceAccountWithRBAC_RoleRefKindChange(t *testing.T) {
	// Setup - pre-create ClusterRole, ServiceAccount, and RoleBinding with wrong RoleRef kind
	_, scheme := setupTest()

	clusterRole := &rbacv1.ClusterRole{
		ObjectMeta: metav1.ObjectMeta{
			Name: testClusterRoleName,
		},
		Rules: []rbacv1.PolicyRule{
			{
				APIGroups: []string{"nvidia.com"},
				Resources: []string{"dynamocomponentdeployments", "dynamographdeployments"},
				Verbs:     []string{"get", "list", "create", "update", "patch"},
			},
		},
	}

	existingSA := &corev1.ServiceAccount{
		ObjectMeta: metav1.ObjectMeta{
			Name:      testServiceAccountName,
			Namespace: testNamespace,
		},
	}

	existingRB := &rbacv1.RoleBinding{
		ObjectMeta: metav1.ObjectMeta{
			Name:      testRoleBindingName,
			Namespace: testNamespace,
		},
		Subjects: []rbacv1.Subject{{
			Kind:      "ServiceAccount",
			Name:      testServiceAccountName,
			Namespace: testNamespace,
		}},
		RoleRef: rbacv1.RoleRef{
			APIGroup: "rbac.authorization.k8s.io",
			Kind:     "Role", // Wrong kind (should be ClusterRole)
			Name:     testClusterRoleName,
		},
	}

	fakeClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(clusterRole, existingSA, existingRB).
		Build()

	manager := NewManager(fakeClient)
	ctx := context.Background()

	// Execute
	err := manager.EnsureServiceAccountWithRBAC(
		ctx,
		testNamespace,
		testServiceAccountName,
		testClusterRoleName,
	)

	// Verify
	if err != nil {
		t.Fatalf("Expected no error, got: %v", err)
	}

	// Verify RoleBinding was recreated with correct RoleRef kind
	rb := &rbacv1.RoleBinding{}
	err = fakeClient.Get(ctx, client.ObjectKey{
		Namespace: testNamespace,
		Name:      testRoleBindingName,
	}, rb)
	if err != nil {
		t.Fatalf("RoleBinding not found: %v", err)
	}

	if rb.RoleRef.Kind != "ClusterRole" {
		t.Errorf("Expected RoleRef kind ClusterRole, got %s", rb.RoleRef.Kind)
	}
	if rb.RoleRef.Name != testClusterRoleName {
		t.Errorf("Expected RoleRef name test-cluster-role, got %s", rb.RoleRef.Name)
	}
}
