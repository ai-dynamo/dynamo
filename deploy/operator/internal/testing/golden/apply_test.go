/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package golden

import (
	"os"
	"path/filepath"
	"testing"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

func TestApplyManifestsCreatesEveryDocumentInNamespace(t *testing.T) {
	t.Log("Write a multi-kind input manifest without namespaces")
	path := filepath.Join(t.TempDir(), "input.yaml")
	contents := `apiVersion: v1
kind: ConfigMap
metadata:
  name: settings
data:
  key: value
---
apiVersion: v1
kind: Secret
metadata:
  name: credentials
---
apiVersion: v1
kind: Namespace
metadata:
  name: shared
`
	if err := os.WriteFile(path, []byte(contents), 0o600); err != nil {
		t.Fatalf("write input manifests: %v", err)
	}
	scheme := runtime.NewScheme()
	if err := corev1.AddToScheme(scheme); err != nil {
		t.Fatalf("add core types to scheme: %v", err)
	}
	mapper := meta.NewDefaultRESTMapper([]schema.GroupVersion{corev1.SchemeGroupVersion})
	mapper.Add(corev1.SchemeGroupVersion.WithKind("ConfigMap"), meta.RESTScopeNamespace)
	mapper.Add(corev1.SchemeGroupVersion.WithKind("Secret"), meta.RESTScopeNamespace)
	mapper.Add(corev1.SchemeGroupVersion.WithKind("Namespace"), meta.RESTScopeRoot)
	k8sClient := fake.NewClientBuilder().WithScheme(scheme).WithRESTMapper(mapper).Build()

	t.Log("Apply both documents as unstructured objects in the test namespace")
	objects := ApplyManifests(t, path, k8sClient, "test")
	if len(objects) != 3 {
		t.Fatalf("ApplyManifests() returned %d objects, want 3", len(objects))
	}
	for i := 0; i < 2; i++ {
		if objects[i].GetNamespace() != "test" {
			t.Fatalf("object %d namespace = %q, want test", i, objects[i].GetNamespace())
		}
	}
	if objects[2].GetNamespace() != "" {
		t.Fatalf("cluster-scoped object namespace = %q, want empty", objects[2].GetNamespace())
	}

	t.Log("Read both persisted objects through the typed client")
	var configMap corev1.ConfigMap
	if err := k8sClient.Get(t.Context(), client.ObjectKey{Namespace: "test", Name: "settings"}, &configMap); err != nil {
		t.Fatalf("get ConfigMap: %v", err)
	}
	if configMap.Data["key"] != "value" {
		t.Fatalf("ConfigMap data = %v, want key=value", configMap.Data)
	}
	var secret corev1.Secret
	if err := k8sClient.Get(t.Context(), client.ObjectKey{Namespace: "test", Name: "credentials"}, &secret); err != nil {
		t.Fatalf("get Secret: %v", err)
	}
	var namespace corev1.Namespace
	if err := k8sClient.Get(t.Context(), client.ObjectKey{Name: "shared"}, &namespace); err != nil {
		t.Fatalf("get Namespace: %v", err)
	}
}

func TestApplyManifestsAcceptsMatchingExplicitNamespace(t *testing.T) {
	t.Log("Write an input manifest already scoped to the test namespace")
	path := filepath.Join(t.TempDir(), "input.yaml")
	contents := `apiVersion: v1
kind: ConfigMap
metadata:
  name: settings
  namespace: test
`
	if err := os.WriteFile(path, []byte(contents), 0o600); err != nil {
		t.Fatalf("write input manifest: %v", err)
	}
	scheme := runtime.NewScheme()
	if err := corev1.AddToScheme(scheme); err != nil {
		t.Fatalf("add core types to scheme: %v", err)
	}
	mapper := meta.NewDefaultRESTMapper([]schema.GroupVersion{corev1.SchemeGroupVersion})
	mapper.Add(corev1.SchemeGroupVersion.WithKind("ConfigMap"), meta.RESTScopeNamespace)
	k8sClient := fake.NewClientBuilder().WithScheme(scheme).WithObjects(
		&corev1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: "test"}},
	).WithRESTMapper(mapper).Build()

	t.Log("Apply the manifest without changing its explicit namespace")
	objects := ApplyManifests(t, path, k8sClient, "test")
	if len(objects) != 1 || objects[0].GetNamespace() != "test" {
		t.Fatalf("ApplyManifests() objects = %#v, want one object in test", objects)
	}
}
