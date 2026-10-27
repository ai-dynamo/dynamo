//go:build !clustertest

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package controller

import (
	"errors"
	"testing"

	corev1 "k8s.io/api/core/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/envtest"
)

func TestDGDRRunPublisherRoleDoesNotEscalateManagerPermissions(t *testing.T) {
	env := sharedEnv.ForTest(t)
	admin := env.Client()
	ctx := t.Context()
	const managerName = "dgdr-run-manager"

	t.Log("Create a persisted Run and a restricted identity with every delegated permission except Candidate access")
	run := dgdrRunForTest(t, map[string]any{})
	run.Name, run.Namespace, run.UID, run.Generation = "rbac-run", env.Namespace(), "", 0
	if err := admin.Create(ctx, run); err != nil {
		t.Fatalf("create Run: %v", err)
	}
	if err := admin.Get(ctx, client.ObjectKeyFromObject(run), run); err != nil {
		t.Fatalf("get Run: %v", err)
	}

	managerRules := []rbacv1.PolicyRule{
		{APIGroups: []string{""}, Resources: []string{"serviceaccounts"}, Verbs: []string{"get", "list", "watch", "create", "update", "patch"}},
		{APIGroups: []string{rbacv1.GroupName}, Resources: []string{"roles", "rolebindings"}, Verbs: []string{"get", "list", "watch", "create", "update", "patch"}},
	}
	for _, rule := range dgdrRunPublisherRules(run.Name) {
		if len(rule.Resources) == 1 && (rule.Resources[0] == "dynamographdeploymentcandidates" || rule.Resources[0] == "dynamographdeploymentcandidates/status") {
			continue
		}
		managerRules = append(managerRules, rule)
	}
	managerRole := &rbacv1.Role{ObjectMeta: metav1.ObjectMeta{Name: managerName, Namespace: run.Namespace}, Rules: managerRules}
	if err := admin.Create(ctx, managerRole); err != nil {
		t.Fatalf("create manager Role: %v", err)
	}
	managerBinding := &rbacv1.RoleBinding{
		ObjectMeta: metav1.ObjectMeta{Name: managerName, Namespace: run.Namespace},
		RoleRef:    rbacv1.RoleRef{APIGroup: rbacv1.GroupName, Kind: "Role", Name: managerRole.Name},
		Subjects:   []rbacv1.Subject{{APIGroup: rbacv1.GroupName, Kind: rbacv1.UserKind, Name: managerName}},
	}
	if err := admin.Create(ctx, managerBinding); err != nil {
		t.Fatalf("create manager RoleBinding: %v", err)
	}
	managerConfig, err := env.AddUser(envtest.User{Name: managerName, Groups: []string{"system:authenticated"}})
	if err != nil {
		t.Fatalf("create restricted manager identity: %v", err)
	}
	managerClient, err := client.New(managerConfig, client.Options{Scheme: env.Scheme()})
	if err != nil {
		t.Fatalf("create restricted manager client: %v", err)
	}
	reconciler := &DynamoGraphDeploymentRunV1beta2Reconciler{Client: managerClient}

	t.Log("Kubernetes rejects the publisher Role while the manager lacks its Candidate permissions")
	err = reconciler.reconcilePublisherAccess(ctx, run, dgdrRunResourceName(run.Name))
	var statusErr *apierrors.StatusError
	if !errors.As(err, &statusErr) || statusErr.ErrStatus.Reason != metav1.StatusReasonForbidden {
		t.Fatalf("reconcile publisher access error = %v, want RBAC escalation rejection", err)
	}

	t.Log("Grant the manager every delegated permission and retry the same reconciliation")
	managerRole.Rules = append(append([]rbacv1.PolicyRule{}, managerRules[:2]...), dgdrRunPublisherRules(run.Name)...)
	if err := admin.Update(ctx, managerRole); err != nil {
		t.Fatalf("grant publisher permissions to manager: %v", err)
	}
	if err := reconciler.reconcilePublisherAccess(ctx, run, dgdrRunResourceName(run.Name)); err != nil {
		t.Fatalf("reconcile publisher access: %v", err)
	}

	t.Log("The restricted manager created the complete publisher access bundle")
	for _, object := range []client.Object{&corev1.ServiceAccount{}, &rbacv1.Role{}, &rbacv1.RoleBinding{}} {
		object.SetName(dgdrRunResourceName(run.Name))
		object.SetNamespace(run.Namespace)
		if err := admin.Get(ctx, client.ObjectKeyFromObject(object), object); err != nil {
			t.Fatalf("get %T: %v", object, err)
		}
	}
	createdRole := &rbacv1.Role{}
	key := client.ObjectKey{Namespace: run.Namespace, Name: dgdrRunResourceName(run.Name)}
	if err := admin.Get(ctx, key, createdRole); err != nil {
		t.Fatalf("get publisher Role: %v", err)
	}
	if want := dgdrRunPublisherRules(run.Name); !apiequality.Semantic.DeepEqual(createdRole.Rules, want) {
		t.Fatalf("publisher Role rules = %#v, want %#v", createdRole.Rules, want)
	}
}
