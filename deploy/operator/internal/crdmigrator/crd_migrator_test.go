/*
SPDX-FileCopyrightText: Copyright 2025 The Kubernetes Authors.
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Tests derived in part from kubernetes-sigs/cluster-api/controllers/crdmigrator
at v1.13.3, commit cf0f6c00fbf7d5c5dbf37bd09554c6389de93861.
*/

package crdmigrator

import (
	"context"
	"sort"
	"testing"
	"time"

	operatorv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

func TestSetupBuildsExpectedDynamoCRDNames(t *testing.T) {
	scheme := runtime.NewScheme()
	if err := operatorv1beta1.AddToScheme(scheme); err != nil {
		t.Fatal(err)
	}
	c := fake.NewClientBuilder().WithScheme(scheme).Build()
	migrator := &CRDMigrator{
		Client: c, APIReader: c,
		Config: map[client.Object]ByObjectConfig{
			&operatorv1beta1.DynamoComponentDeployment{}:           {},
			&operatorv1beta1.DynamoGraphDeployment{}:               {},
			&operatorv1beta1.DynamoGraphDeploymentRequest{}:        {},
			&operatorv1beta1.DynamoGraphDeploymentScalingAdapter{}: {},
		},
	}
	if err := migrator.setup(scheme); err != nil {
		t.Fatal(err)
	}
	want := []string{
		"dynamocomponentdeployments.nvidia.com",
		"dynamographdeploymentrequests.nvidia.com",
		"dynamographdeployments.nvidia.com",
		"dynamographdeploymentscalingadapters.nvidia.com",
	}
	got := make([]string, 0, len(migrator.configByCRDName))
	for name := range migrator.configByCRDName {
		got = append(got, name)
	}
	sort.Strings(got)
	if len(got) != len(want) {
		t.Fatalf("configured CRDs = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("configured CRDs = %v, want %v", got, want)
		}
	}
}

func TestSetupRejectsInvalidConfiguration(t *testing.T) {
	migrator := &CRDMigrator{}
	if err := migrator.setup(runtime.NewScheme()); err == nil {
		t.Fatal("setup succeeded without clients and config")
	}
}

func TestReconcileEmptyCRDConvergesStoredVersionsAndAnnotation(t *testing.T) {
	scheme := runtime.NewScheme()
	if err := apiextensionsv1.AddToScheme(scheme); err != nil {
		t.Fatal(err)
	}
	if err := operatorv1beta1.AddToScheme(scheme); err != nil {
		t.Fatal(err)
	}
	crd := &apiextensionsv1.CustomResourceDefinition{
		ObjectMeta: metav1.ObjectMeta{
			Name:       "dynamocomponentdeployments.nvidia.com",
			Generation: 7,
		},
		Spec: apiextensionsv1.CustomResourceDefinitionSpec{
			Group: "nvidia.com",
			Names: apiextensionsv1.CustomResourceDefinitionNames{
				Plural: "dynamocomponentdeployments", Kind: "DynamoComponentDeployment",
				ListKind: "DynamoComponentDeploymentList",
			},
			Scope: apiextensionsv1.NamespaceScoped,
			Versions: []apiextensionsv1.CustomResourceDefinitionVersion{
				{Name: "v1alpha1", Served: true},
				{Name: "v1beta1", Served: true, Storage: true},
			},
		},
		Status: apiextensionsv1.CustomResourceDefinitionStatus{StoredVersions: []string{"v1alpha1", "v1beta1"}},
	}
	c := fake.NewClientBuilder().WithScheme(scheme).WithStatusSubresource(crd).WithObjects(crd).Build()
	migrator := &CRDMigrator{
		Client: c, APIReader: c,
		Config: map[client.Object]ByObjectConfig{
			&operatorv1beta1.DynamoComponentDeployment{}: {UseStatusForStorageVersionMigration: true},
		},
	}
	if err := migrator.setup(scheme); err != nil {
		t.Fatal(err)
	}
	if _, err := migrator.Reconcile(context.Background(), ctrl.Request{NamespacedName: types.NamespacedName{Name: crd.Name}}); err != nil {
		t.Fatal(err)
	}
	got := &apiextensionsv1.CustomResourceDefinition{}
	if err := c.Get(context.Background(), client.ObjectKey{Name: crd.Name}, got); err != nil {
		t.Fatal(err)
	}
	if len(got.Status.StoredVersions) != 1 || got.Status.StoredVersions[0] != "v1beta1" {
		t.Fatalf("storedVersions = %v, want [v1beta1]", got.Status.StoredVersions)
	}
	if got.Annotations[ObservedGenerationAnnotation] != "7" {
		t.Fatalf("observed generation = %q, want 7", got.Annotations[ObservedGenerationAnnotation])
	}
}

func TestFilterManagedFields(t *testing.T) {
	obj := &operatorv1beta1.DynamoComponentDeployment{ObjectMeta: metav1.ObjectMeta{
		ManagedFields: []metav1.ManagedFieldsEntry{
			{APIVersion: "nvidia.com/v1alpha1"},
			{APIVersion: "nvidia.com/v1beta1"},
		},
	}}
	got, removed := filterManagedFields(obj, sets.New("nvidia.com/v1beta1"))
	if !removed || len(got) != 1 || got[0].APIVersion != "nvidia.com/v1beta1" {
		t.Fatalf("filterManagedFields() = (%v, %t)", got, removed)
	}
}

func TestTTLSetExpiresEntries(t *testing.T) {
	set := newTTLSet(time.Millisecond)
	set.Add("object")
	if !set.Has("object") {
		t.Fatal("entry missing before expiry")
	}
	time.Sleep(5 * time.Millisecond)
	if set.Has("object") {
		t.Fatal("entry present after expiry")
	}
}
