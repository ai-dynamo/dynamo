//go:build !clustertest

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package controller

import (
	"context"
	"errors"
	"testing"

	nvidiacomv1beta2 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta2"
	corev1 "k8s.io/api/core/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
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
	managerRole := &rbacv1.Role{Name: managerName, Namespace: run.Namespace, Rules: managerRules}
	if err := admin.Create(ctx, managerRole); err != nil {
		t.Fatalf("create manager Role: %v", err)
	}
	managerBinding := &rbacv1.RoleBinding{
		Name:      managerName,
		Namespace: run.Namespace,
		RoleRef:   rbacv1.RoleRef{APIGroup: rbacv1.GroupName, Kind: "Role", Name: managerRole.Name},
		Subjects:  []rbacv1.Subject{{APIGroup: rbacv1.GroupName, Kind: rbacv1.UserKind, Name: managerName}},
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
	if !roleAllowsRunFinalizerUpdate(createdRole.Rules, run.Name) {
		t.Fatalf("publisher Role does not allow updating finalizers of Run %q: %#v", run.Name, createdRole.Rules)
	}
}

func roleAllowsRunFinalizerUpdate(rules []rbacv1.PolicyRule, runName string) bool {
	for _, rule := range rules {
		if len(rule.APIGroups) == 1 && rule.APIGroups[0] == "nvidia.com" &&
			len(rule.Resources) == 1 && rule.Resources[0] == "dynamographdeploymentruns/finalizers" &&
			len(rule.ResourceNames) == 1 && rule.ResourceNames[0] == runName &&
			len(rule.Verbs) == 1 && rule.Verbs[0] == "update" {
			return true
		}
	}
	return false
}

func TestDGDRRunTraceClaimProtectionIsFencedByTheAPIServer(t *testing.T) {
	env := sharedEnv.ForTest(t)
	admin := env.Client()
	ctx := t.Context()
	reconciler := &DynamoGraphDeploymentRunV1beta2Reconciler{Client: admin}
	first := &nvidiacomv1beta2.DynamoGraphDeploymentRun{Name: "first", UID: "first-uid"}
	second := &nvidiacomv1beta2.DynamoGraphDeploymentRun{Name: "second", UID: "second-uid"}

	t.Log("Create a trace claim that another finalizer keeps while it is deleted")
	claim := &corev1.PersistentVolumeClaim{
		Name: "request-traces", Namespace: env.Namespace(), Finalizers: []string{"example.com/hold"},
		Spec: corev1.PersistentVolumeClaimSpec{
			AccessModes: []corev1.PersistentVolumeAccessMode{corev1.ReadWriteMany},
			Resources:   corev1.VolumeResourceRequirements{Requests: corev1.ResourceList{corev1.ResourceStorage: resource.MustParse("1Mi")}},
		},
	}
	if err := admin.Create(ctx, claim); err != nil {
		t.Fatalf("create claim: %v", err)
	}
	t.Cleanup(func() {
		stored := &corev1.PersistentVolumeClaim{}
		if err := admin.Get(context.Background(), client.ObjectKeyFromObject(claim), stored); err != nil {
			return
		}
		stored.Finalizers = nil
		if err := admin.Update(context.Background(), stored); client.IgnoreNotFound(err) != nil {
			t.Errorf("release claim: %v", err)
		}
	})
	stale := claim.DeepCopy()

	t.Log("The first Run protects the claim")
	if err := reconciler.protectTraceClaim(ctx, first, claim, nil); err != nil {
		t.Fatalf("protect claim: %v", err)
	}
	if !controllerutil.ContainsFinalizer(claim, dgdrRunTraceClaimFinalizer(first.UID)) {
		t.Fatalf("finalizers = %v, want the first Run's", claim.Finalizers)
	}
	protected := claim.DeepCopy()

	t.Log("A protection read from a stale claim conflicts and is retried")
	err := reconciler.protectTraceClaim(ctx, second, stale, nil)
	if !apierrors.IsConflict(err) || dgdrRunIsPermanentError(err) {
		t.Fatalf("stale protection error = %v, want a retryable conflict", err)
	}

	t.Log("Delete the claim")
	if err := admin.Delete(ctx, claim); err != nil {
		t.Fatalf("delete claim: %v", err)
	}

	t.Log("A protected claim whose deletion the cache has not shown fails the fence before a Job is created")
	err = reconciler.protectTraceClaim(ctx, first, protected, nil)
	if !apierrors.IsConflict(err) || dgdrRunIsPermanentError(err) {
		t.Fatalf("fence error for a stale protected claim = %v, want a retryable conflict", err)
	}

	t.Log("The API server refuses a new finalizer on the claim being deleted, and the refusal is retried")
	deleting := &corev1.PersistentVolumeClaim{}
	if err := admin.Get(ctx, client.ObjectKeyFromObject(claim), deleting); err != nil {
		t.Fatalf("get claim: %v", err)
	}
	err = reconciler.protectTraceClaim(ctx, second, deleting, nil)
	if !apierrors.IsInvalid(err) || dgdrRunIsPermanentError(err) {
		t.Fatalf("protection error for a claim being deleted = %v, want a retryable Invalid", err)
	}
}
