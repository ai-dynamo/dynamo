/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package rbac

import (
	"context"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	corev1 "k8s.io/api/core/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

const (
	testServiceAccountName = "test-sa"
	testNamespace          = "test-namespace"
	testClusterRoleName    = "test-cluster-role"
)

func testScheme(t *testing.T) *runtime.Scheme {
	t.Helper()

	scheme := runtime.NewScheme()
	if err := corev1.AddToScheme(scheme); err != nil {
		t.Fatalf("add core API to test scheme: %v", err)
	}
	if err := rbacv1.AddToScheme(scheme); err != nil {
		t.Fatalf("add RBAC API to test scheme: %v", err)
	}
	return scheme
}

func testClusterRole(name string) *rbacv1.ClusterRole {
	return &rbacv1.ClusterRole{
		ObjectMeta: metav1.ObjectMeta{Name: name},
		Rules: []rbacv1.PolicyRule{{
			APIGroups: []string{"nvidia.com"},
			Resources: []string{"dynamocomponentdeployments", "dynamographdeployments"},
			Verbs:     []string{"get", "list", "create", "update", "patch"},
		}},
	}
}

func managedLabels(serviceAccountName string) map[string]string {
	return map[string]string{
		"app.kubernetes.io/managed-by": "dynamo-operator",
		"app.kubernetes.io/component":  "rbac",
		"app.kubernetes.io/name":       serviceAccountName,
	}
}

func testServiceAccount(namespace, name string, labels map[string]string) *corev1.ServiceAccount {
	return &corev1.ServiceAccount{
		ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: namespace, Labels: labels},
	}
}

func testRoleBinding(
	namespace string,
	serviceAccountName string,
	subjects []rbacv1.Subject,
	roleRef rbacv1.RoleRef,
	labels map[string]string,
) *rbacv1.RoleBinding {
	return &rbacv1.RoleBinding{
		ObjectMeta: metav1.ObjectMeta{
			Name:      serviceAccountName + "-binding",
			Namespace: namespace,
			Labels:    labels,
		},
		Subjects: subjects,
		RoleRef:  roleRef,
	}
}

func desiredSubjects(namespace, serviceAccountName string) []rbacv1.Subject {
	return []rbacv1.Subject{{
		Kind:      kindServiceAccount,
		Name:      serviceAccountName,
		Namespace: namespace,
	}}
}

func desiredRoleRef(clusterRoleName string) rbacv1.RoleRef {
	return rbacv1.RoleRef{
		APIGroup: apiGroupRBAC,
		Kind:     kindClusterRole,
		Name:     clusterRoleName,
	}
}

func TestEnsureServiceAccountWithRBAC(t *testing.T) {
	existingLabels := map[string]string{"existing": "true"}
	testCases := []struct {
		name               string
		targetNamespace    string
		serviceAccountName string
		clusterRoleName    string
		initialObjects     []client.Object
		expectedError      string
		wantServiceAccount *corev1.ServiceAccount
		wantRoleBinding    *rbacv1.RoleBinding
	}{
		{
			name:               "creates new resources",
			targetNamespace:    testNamespace,
			serviceAccountName: testServiceAccountName,
			clusterRoleName:    testClusterRoleName,
			initialObjects:     []client.Object{testClusterRole(testClusterRoleName)},
			wantServiceAccount: testServiceAccount(testNamespace, testServiceAccountName, managedLabels(testServiceAccountName)),
			wantRoleBinding: testRoleBinding(
				testNamespace, testServiceAccountName, desiredSubjects(testNamespace, testServiceAccountName),
				desiredRoleRef(testClusterRoleName), managedLabels(testServiceAccountName),
			),
		},
		{
			name:               "preserves existing resources",
			targetNamespace:    testNamespace,
			serviceAccountName: testServiceAccountName,
			clusterRoleName:    testClusterRoleName,
			initialObjects: []client.Object{
				testClusterRole(testClusterRoleName),
				testServiceAccount(testNamespace, testServiceAccountName, existingLabels),
				testRoleBinding(testNamespace, testServiceAccountName, desiredSubjects(testNamespace, testServiceAccountName), desiredRoleRef(testClusterRoleName), existingLabels),
			},
			wantServiceAccount: testServiceAccount(testNamespace, testServiceAccountName, existingLabels),
			wantRoleBinding:    testRoleBinding(testNamespace, testServiceAccountName, desiredSubjects(testNamespace, testServiceAccountName), desiredRoleRef(testClusterRoleName), existingLabels),
		},
		{
			name:               "updates an incorrect subject name",
			targetNamespace:    testNamespace,
			serviceAccountName: testServiceAccountName,
			clusterRoleName:    testClusterRoleName,
			initialObjects: []client.Object{
				testClusterRole(testClusterRoleName),
				testServiceAccount(testNamespace, testServiceAccountName, existingLabels),
				testRoleBinding(testNamespace, testServiceAccountName, []rbacv1.Subject{{Kind: kindServiceAccount, Name: "wrong-sa", Namespace: testNamespace}}, desiredRoleRef(testClusterRoleName), existingLabels),
			},
			wantServiceAccount: testServiceAccount(testNamespace, testServiceAccountName, existingLabels),
			wantRoleBinding:    testRoleBinding(testNamespace, testServiceAccountName, desiredSubjects(testNamespace, testServiceAccountName), desiredRoleRef(testClusterRoleName), existingLabels),
		},
		{
			name:               "rejects a missing cluster role",
			targetNamespace:    testNamespace,
			serviceAccountName: testServiceAccountName,
			clusterRoleName:    "non-existent-cluster-role",
			expectedError:      "cluster role \"non-existent-cluster-role\" does not exist",
		},
		{
			name:               "rejects an empty namespace",
			targetNamespace:    "",
			serviceAccountName: testServiceAccountName,
			clusterRoleName:    testClusterRoleName,
			expectedError:      "target namespace is required",
		},
		{
			name:               "rejects an empty service account name",
			targetNamespace:    testNamespace,
			serviceAccountName: "",
			clusterRoleName:    testClusterRoleName,
			expectedError:      "service account name is required",
		},
		{
			name:               "rejects an empty cluster role name",
			targetNamespace:    testNamespace,
			serviceAccountName: testServiceAccountName,
			clusterRoleName:    "",
			expectedError:      "cluster role name is required",
		},
		{
			name:               "changes the referenced cluster role",
			targetNamespace:    testNamespace,
			serviceAccountName: testServiceAccountName,
			clusterRoleName:    "new-cluster-role",
			initialObjects: []client.Object{
				testClusterRole("old-cluster-role"),
				testClusterRole("new-cluster-role"),
				testServiceAccount(testNamespace, testServiceAccountName, existingLabels),
				testRoleBinding(testNamespace, testServiceAccountName, desiredSubjects(testNamespace, testServiceAccountName), desiredRoleRef("old-cluster-role"), existingLabels),
			},
			wantServiceAccount: testServiceAccount(testNamespace, testServiceAccountName, existingLabels),
			wantRoleBinding:    testRoleBinding(testNamespace, testServiceAccountName, desiredSubjects(testNamespace, testServiceAccountName), desiredRoleRef("new-cluster-role"), managedLabels(testServiceAccountName)),
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			t.Log("Arrange the scenario's Kubernetes state")
			fakeClient := fake.NewClientBuilder().
				WithScheme(testScheme(t)).
				WithObjects(tc.initialObjects...).
				Build()
			ctx := context.Background()

			t.Log("Ensure the requested ServiceAccount and RoleBinding")
			err := EnsureServiceAccountWithRBAC(
				ctx,
				fakeClient,
				tc.targetNamespace,
				tc.serviceAccountName,
				tc.clusterRoleName,
			)

			if tc.expectedError != "" {
				t.Log("Verify the error and infer that target resources remain absent")
				if err == nil || !strings.Contains(err.Error(), tc.expectedError) {
					t.Fatalf("expected error containing %q, got %v", tc.expectedError, err)
				}
				serviceAccountKey := client.ObjectKey{Namespace: tc.targetNamespace, Name: tc.serviceAccountName}
				if getErr := fakeClient.Get(ctx, serviceAccountKey, &corev1.ServiceAccount{}); !apierrors.IsNotFound(getErr) {
					t.Errorf("expected ServiceAccount %s to be absent, got %v", serviceAccountKey, getErr)
				}
				roleBindingKey := client.ObjectKey{Namespace: tc.targetNamespace, Name: tc.serviceAccountName + "-binding"}
				if getErr := fakeClient.Get(ctx, roleBindingKey, &rbacv1.RoleBinding{}); !apierrors.IsNotFound(getErr) {
					t.Errorf("expected RoleBinding %s to be absent, got %v", roleBindingKey, getErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("Ensure returned an unexpected error: %v", err)
			}

			t.Log("Compare the resulting ServiceAccount")
			gotServiceAccount := &corev1.ServiceAccount{}
			if err := fakeClient.Get(ctx, client.ObjectKeyFromObject(tc.wantServiceAccount), gotServiceAccount); err != nil {
				t.Fatalf("get ServiceAccount: %v", err)
			}
			gotServiceAccount.ResourceVersion = ""
			if diff := cmp.Diff(tc.wantServiceAccount, gotServiceAccount); diff != "" {
				t.Errorf("ServiceAccount mismatch (-want +got):\n%s", diff)
			}

			t.Log("Compare the resulting RoleBinding")
			gotRoleBinding := &rbacv1.RoleBinding{}
			if err := fakeClient.Get(ctx, client.ObjectKeyFromObject(tc.wantRoleBinding), gotRoleBinding); err != nil {
				t.Fatalf("get RoleBinding: %v", err)
			}
			gotRoleBinding.ResourceVersion = ""
			if diff := cmp.Diff(tc.wantRoleBinding, gotRoleBinding); diff != "" {
				t.Errorf("RoleBinding mismatch (-want +got):\n%s", diff)
			}
		})
	}
}
