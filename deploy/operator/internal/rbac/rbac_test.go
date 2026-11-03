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
	"k8s.io/apimachinery/pkg/runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

const (
	testServiceAccountName = "test-sa"
	testNamespace          = "test-namespace"
	testClusterRoleName    = "test-cluster-role"
	testOtherNamespace     = "other-namespace"
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
		Name: name,
		Rules: []rbacv1.PolicyRule{{
			APIGroups: []string{"nvidia.com"},
			Resources: []string{"dynamocomponentdeployments", "dynamographdeployments"},
			Verbs:     []string{"get", "list", "create", "update", "patch"},
		}},
	}
}

func managedLabels() map[string]string {
	return map[string]string{
		"app.kubernetes.io/managed-by": "dynamo-operator",
		"app.kubernetes.io/component":  "rbac",
		"app.kubernetes.io/name":       testServiceAccountName,
	}
}

func testServiceAccount(labels map[string]string) *corev1.ServiceAccount {
	return &corev1.ServiceAccount{
		Name: testServiceAccountName, Namespace: testNamespace, Labels: labels,
	}
}

func testRoleBinding(
	subjects []rbacv1.Subject,
	roleRef rbacv1.RoleRef,
	labels map[string]string,
) *rbacv1.RoleBinding {
	return &rbacv1.RoleBinding{
		Name:      testServiceAccountName + "-binding",
		Namespace: testNamespace,
		Labels:    labels,
		Subjects:  subjects,
		RoleRef:   roleRef,
	}
}

func desiredSubjects() []rbacv1.Subject {
	return []rbacv1.Subject{{
		Kind:      kindServiceAccount,
		Name:      testServiceAccountName,
		Namespace: testNamespace,
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
		ensureAgain        bool
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
			wantServiceAccount: testServiceAccount(managedLabels()),
			wantRoleBinding: testRoleBinding(
				desiredSubjects(),
				desiredRoleRef(testClusterRoleName), managedLabels()),
		},
		{
			name:               "converges when ensured again",
			targetNamespace:    testNamespace,
			serviceAccountName: testServiceAccountName,
			clusterRoleName:    testClusterRoleName,
			initialObjects:     []client.Object{testClusterRole(testClusterRoleName)},
			ensureAgain:        true,
			wantServiceAccount: testServiceAccount(managedLabels()),
			wantRoleBinding:    testRoleBinding(desiredSubjects(), desiredRoleRef(testClusterRoleName), managedLabels()),
		},
		{
			name:               "creates a missing binding for an existing service account",
			targetNamespace:    testNamespace,
			serviceAccountName: testServiceAccountName,
			clusterRoleName:    testClusterRoleName,
			initialObjects: []client.Object{
				testClusterRole(testClusterRoleName),
				testServiceAccount(existingLabels),
			},
			wantServiceAccount: testServiceAccount(existingLabels),
			wantRoleBinding:    testRoleBinding(desiredSubjects(), desiredRoleRef(testClusterRoleName), managedLabels()),
		},
		{
			name:               "creates resources without touching another namespace",
			targetNamespace:    testNamespace,
			serviceAccountName: testServiceAccountName,
			clusterRoleName:    testClusterRoleName,
			initialObjects: []client.Object{
				testClusterRole(testClusterRoleName),
				&corev1.ServiceAccount{Name: testServiceAccountName, Namespace: testOtherNamespace, Labels: managedLabels()},
				&rbacv1.RoleBinding{
					Name:      testServiceAccountName + "-binding",
					Namespace: testOtherNamespace,
					Labels:    managedLabels(),
					Subjects:  []rbacv1.Subject{{Kind: kindServiceAccount, Name: testServiceAccountName, Namespace: testOtherNamespace}},
					RoleRef:   desiredRoleRef(testClusterRoleName),
				},
			},
			wantServiceAccount: testServiceAccount(managedLabels()),
			wantRoleBinding:    testRoleBinding(desiredSubjects(), desiredRoleRef(testClusterRoleName), managedLabels()),
		},
		{
			name:               "preserves existing resources",
			targetNamespace:    testNamespace,
			serviceAccountName: testServiceAccountName,
			clusterRoleName:    testClusterRoleName,
			initialObjects: []client.Object{
				testClusterRole(testClusterRoleName),
				testServiceAccount(existingLabels),
				testRoleBinding(desiredSubjects(), desiredRoleRef(testClusterRoleName), existingLabels),
			},
			wantServiceAccount: testServiceAccount(existingLabels),
			wantRoleBinding:    testRoleBinding(desiredSubjects(), desiredRoleRef(testClusterRoleName), existingLabels),
		},
		{
			name:               "updates an incorrect subject name",
			targetNamespace:    testNamespace,
			serviceAccountName: testServiceAccountName,
			clusterRoleName:    testClusterRoleName,
			initialObjects: []client.Object{
				testClusterRole(testClusterRoleName),
				testServiceAccount(existingLabels),
				testRoleBinding([]rbacv1.Subject{{Kind: kindServiceAccount, Name: "wrong-sa", Namespace: testNamespace}}, desiredRoleRef(testClusterRoleName), existingLabels),
			},
			wantServiceAccount: testServiceAccount(existingLabels),
			wantRoleBinding:    testRoleBinding(desiredSubjects(), desiredRoleRef(testClusterRoleName), existingLabels),
		},
		{
			name:               "updates an incorrect subject namespace",
			targetNamespace:    testNamespace,
			serviceAccountName: testServiceAccountName,
			clusterRoleName:    testClusterRoleName,
			initialObjects: []client.Object{
				testClusterRole(testClusterRoleName),
				testServiceAccount(existingLabels),
				testRoleBinding([]rbacv1.Subject{{Kind: kindServiceAccount, Name: testServiceAccountName, Namespace: testOtherNamespace}}, desiredRoleRef(testClusterRoleName), existingLabels),
			},
			wantServiceAccount: testServiceAccount(existingLabels),
			wantRoleBinding:    testRoleBinding(desiredSubjects(), desiredRoleRef(testClusterRoleName), existingLabels),
		},
		{
			name:               "updates an incorrect subject kind",
			targetNamespace:    testNamespace,
			serviceAccountName: testServiceAccountName,
			clusterRoleName:    testClusterRoleName,
			initialObjects: []client.Object{
				testClusterRole(testClusterRoleName),
				testServiceAccount(existingLabels),
				testRoleBinding([]rbacv1.Subject{{Kind: rbacv1.UserKind, Name: testServiceAccountName, Namespace: testNamespace}}, desiredRoleRef(testClusterRoleName), existingLabels),
			},
			wantServiceAccount: testServiceAccount(existingLabels),
			wantRoleBinding:    testRoleBinding(desiredSubjects(), desiredRoleRef(testClusterRoleName), existingLabels),
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
				testServiceAccount(existingLabels),
				testRoleBinding(desiredSubjects(), desiredRoleRef("old-cluster-role"), existingLabels),
			},
			wantServiceAccount: testServiceAccount(existingLabels),
			wantRoleBinding:    testRoleBinding(desiredSubjects(), desiredRoleRef("new-cluster-role"), managedLabels()),
		},
		{
			name:               "replaces a binding that references a namespaced Role",
			targetNamespace:    testNamespace,
			serviceAccountName: testServiceAccountName,
			clusterRoleName:    testClusterRoleName,
			initialObjects: []client.Object{
				testClusterRole(testClusterRoleName),
				testServiceAccount(existingLabels),
				testRoleBinding(desiredSubjects(), rbacv1.RoleRef{APIGroup: apiGroupRBAC, Kind: "Role", Name: testClusterRoleName}, existingLabels),
			},
			wantServiceAccount: testServiceAccount(existingLabels),
			wantRoleBinding:    testRoleBinding(desiredSubjects(), desiredRoleRef(testClusterRoleName), managedLabels()),
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

			if tc.ensureAgain {
				t.Log("Ensure again against the resources created by the first call")
				if err := EnsureServiceAccountWithRBAC(
					ctx,
					fakeClient,
					tc.targetNamespace,
					tc.serviceAccountName,
					tc.clusterRoleName,
				); err != nil {
					t.Fatalf("Ensure again returned an unexpected error: %v", err)
				}
			}

			t.Log("Compare the ServiceAccounts in the target namespace")
			gotServiceAccounts := &corev1.ServiceAccountList{}
			if err := fakeClient.List(ctx, gotServiceAccounts, client.InNamespace(tc.targetNamespace)); err != nil {
				t.Fatalf("list ServiceAccounts: %v", err)
			}
			for i := range gotServiceAccounts.Items {
				gotServiceAccounts.Items[i].ResourceVersion = ""
			}
			if diff := cmp.Diff([]corev1.ServiceAccount{*tc.wantServiceAccount}, gotServiceAccounts.Items); diff != "" {
				t.Errorf("ServiceAccounts mismatch (-want +got):\n%s", diff)
			}

			t.Log("Compare the RoleBindings in the target namespace")
			gotRoleBindings := &rbacv1.RoleBindingList{}
			if err := fakeClient.List(ctx, gotRoleBindings, client.InNamespace(tc.targetNamespace)); err != nil {
				t.Fatalf("list RoleBindings: %v", err)
			}
			for i := range gotRoleBindings.Items {
				gotRoleBindings.Items[i].ResourceVersion = ""
			}
			if diff := cmp.Diff([]rbacv1.RoleBinding{*tc.wantRoleBinding}, gotRoleBindings.Items); diff != "" {
				t.Errorf("RoleBindings mismatch (-want +got):\n%s", diff)
			}

			t.Log("Verify objects outside the target namespace are unchanged")
			for _, initial := range tc.initialObjects {
				if initial.GetNamespace() == tc.targetNamespace {
					continue
				}
				want := initial.DeepCopyObject().(client.Object)
				want.SetResourceVersion("")
				got := initial.DeepCopyObject().(client.Object)
				if err := fakeClient.Get(ctx, client.ObjectKeyFromObject(initial), got); err != nil {
					t.Fatalf("get %T %s: %v", initial, client.ObjectKeyFromObject(initial), err)
				}
				got.SetResourceVersion("")
				if diff := cmp.Diff(want, got); diff != "" {
					t.Errorf("%T %s changed (-want +got):\n%s", initial, client.ObjectKeyFromObject(initial), diff)
				}
			}
		})
	}
}
