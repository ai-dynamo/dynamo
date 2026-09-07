/*
SPDX-FileCopyrightText: Copyright 2025 The Kubernetes Authors.
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Tests derived in part from kubernetes-sigs/cluster-api/controllers/crdmigrator and
kubernetes-sigs/cluster-api/util/cache at v1.13.4,
commit 27f464418c195d96ae2ef4b96f3b6a047ea89310.
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

func TestSetup(t *testing.T) {
	t.Log("Define valid and invalid migrator setup cases")
	tests := []struct {
		name      string
		valid     bool
		wantNames []string
	}{
		{
			name:  "builds expected Dynamo CRD names",
			valid: true,
			wantNames: []string{
				"dynamocomponentdeployments.nvidia.com",
				"dynamographdeploymentrequests.nvidia.com",
				"dynamographdeployments.nvidia.com",
				"dynamographdeploymentscalingadapters.nvidia.com",
			},
		},
		{name: "rejects missing clients and config"},
	}

	t.Log("Run each migrator setup case")
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Log("Arrange a scheme and the requested migrator configuration")
			scheme := runtime.NewScheme()
			migrator := &CRDMigrator{}
			if tt.valid {
				if err := operatorv1beta1.AddToScheme(scheme); err != nil {
					t.Fatal(err)
				}
				c := fake.NewClientBuilder().WithScheme(scheme).Build()
				migrator.Client = c
				migrator.APIReader = c
				migrator.Config = map[client.Object]ByObjectConfig{
					&operatorv1beta1.DynamoComponentDeployment{}:           {},
					&operatorv1beta1.DynamoGraphDeployment{}:               {},
					&operatorv1beta1.DynamoGraphDeploymentRequest{}:        {},
					&operatorv1beta1.DynamoGraphDeploymentScalingAdapter{}: {},
				}
			}

			t.Log("Run migrator setup")
			err := migrator.setup(scheme)

			t.Log("Verify setup validation and configured CRD names")
			if !tt.valid {
				if err == nil {
					t.Fatal("setup succeeded without clients and config")
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			gotNames := make([]string, 0, len(migrator.configByCRDName))
			for name := range migrator.configByCRDName {
				gotNames = append(gotNames, name)
			}
			sort.Strings(gotNames)
			if len(gotNames) != len(tt.wantNames) {
				t.Fatalf("configured CRDs = %v, want %v", gotNames, tt.wantNames)
			}
			for i := range tt.wantNames {
				if gotNames[i] != tt.wantNames[i] {
					t.Fatalf("configured CRDs = %v, want %v", gotNames, tt.wantNames)
				}
			}
		})
	}
}

func TestReconcileEmptyCRDConvergesStoredVersionsAndAnnotation(t *testing.T) {
	t.Log("Arrange the API schemes used by the CRD migrator")
	scheme := runtime.NewScheme()
	if err := apiextensionsv1.AddToScheme(scheme); err != nil {
		t.Fatal(err)
	}
	if err := operatorv1beta1.AddToScheme(scheme); err != nil {
		t.Fatal(err)
	}

	t.Log("Arrange an empty CRD that still records the old storage version")
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

	t.Log("Arrange the fake client and configured migrator")
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

	t.Log("Reconcile the CRD storage version")
	if _, err := migrator.Reconcile(context.Background(), ctrl.Request{NamespacedName: types.NamespacedName{Name: crd.Name}}); err != nil {
		t.Fatal(err)
	}

	t.Log("Read the reconciled CRD")
	got := &apiextensionsv1.CustomResourceDefinition{}
	if err := c.Get(context.Background(), client.ObjectKey{Name: crd.Name}, got); err != nil {
		t.Fatal(err)
	}

	t.Log("Verify the storage version and observed generation converged")
	if len(got.Status.StoredVersions) != 1 || got.Status.StoredVersions[0] != "v1beta1" {
		t.Fatalf("storedVersions = %v, want [v1beta1]", got.Status.StoredVersions)
	}
	if got.Annotations[ObservedGenerationAnnotation] != "7" {
		t.Fatalf("observed generation = %q, want 7", got.Annotations[ObservedGenerationAnnotation])
	}
}

func TestFilterManagedFields(t *testing.T) {
	t.Log("Define managed-fields filtering cases")
	tests := []struct {
		name            string
		managedVersions []string
		servedVersions  []string
		wantVersions    []string
		wantRemoved     bool
	}{
		{
			name:            "removes unserved versions",
			managedVersions: []string{"nvidia.com/v1alpha1", "nvidia.com/v1beta1"},
			servedVersions:  []string{"nvidia.com/v1beta1"},
			wantVersions:    []string{"nvidia.com/v1beta1"},
			wantRemoved:     true,
		},
		{
			name:            "keeps all served versions",
			managedVersions: []string{"nvidia.com/v1alpha1", "nvidia.com/v1beta1"},
			servedVersions:  []string{"nvidia.com/v1alpha1", "nvidia.com/v1beta1"},
			wantVersions:    []string{"nvidia.com/v1alpha1", "nvidia.com/v1beta1"},
		},
		{
			name:            "removes all unserved versions",
			managedVersions: []string{"nvidia.com/v1alpha1"},
			servedVersions:  []string{"nvidia.com/v1beta1"},
			wantRemoved:     true,
		},
	}

	t.Log("Run each managed-fields filtering case")
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Log("Arrange managed fields and the served API versions")
			managedFields := make([]metav1.ManagedFieldsEntry, 0, len(tt.managedVersions))
			for _, version := range tt.managedVersions {
				managedFields = append(managedFields, metav1.ManagedFieldsEntry{APIVersion: version})
			}
			obj := &operatorv1beta1.DynamoComponentDeployment{
				ObjectMeta: metav1.ObjectMeta{ManagedFields: managedFields},
			}

			t.Log("Filter managed fields for unserved API versions")
			got, removed := filterManagedFields(obj, sets.New(tt.servedVersions...))

			t.Log("Verify only served managed fields remain")
			if removed != tt.wantRemoved {
				t.Fatalf("removed = %t, want %t", removed, tt.wantRemoved)
			}
			if len(got) != len(tt.wantVersions) {
				t.Fatalf("managed fields = %v, want API versions %v", got, tt.wantVersions)
			}
			for i := range tt.wantVersions {
				if got[i].APIVersion != tt.wantVersions[i] {
					t.Fatalf("managed fields = %v, want API versions %v", got, tt.wantVersions)
				}
			}
		})
	}
}

func TestTTLCacheSweepsExpiredEntries(t *testing.T) {
	t.Log("Arrange a short-lived cache entry and confirm it is initially present")
	cache := newTTLCacheWithExpirationInterval[objectEntry](time.Millisecond, time.Millisecond)
	entry := objectEntry{Kind: "DynamoGraphDeployment", ObjectKey: client.ObjectKey{Name: "object"}, CRDGeneration: 1}
	cache.Add(entry)
	if _, ok := cache.Has(entry.Key()); !ok {
		t.Fatal("entry missing before expiry")
	}

	t.Log("Wait for the periodic sweep without reading the entry by key")
	deadline := time.Now().Add(time.Second)
	for cache.Len() != 0 && time.Now().Before(deadline) {
		time.Sleep(time.Millisecond)
	}

	t.Log("Verify the periodic sweep reclaimed the expired entry")
	if cache.Len() != 0 {
		t.Fatal("expired entry was not removed by the periodic sweep")
	}
}
