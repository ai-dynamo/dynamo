/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package golden

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

func TestCompareMatchesExactSetsAcrossKinds(t *testing.T) {
	t.Log("Create actual objects of two kinds in one namespace")
	scheme := runtime.NewScheme()
	if err := corev1.AddToScheme(scheme); err != nil {
		t.Fatalf("add core types to scheme: %v", err)
	}
	k8sClient := fake.NewClientBuilder().WithScheme(scheme).WithObjects(
		&corev1.ConfigMap{ObjectMeta: metav1.ObjectMeta{Name: "generated-a1", Namespace: "test"}, Data: map[string]string{"key": "value"}},
		&corev1.Secret{ObjectMeta: metav1.ObjectMeta{Name: "credentials", Namespace: "test"}, StringData: map[string]string{"token": "secret"}},
	).Build()
	expected := readTestDocuments(t, `
$strict: false
apiVersion: v1
kind: ConfigMap
metadata:
  $strict: false
  name: $glob:generated-*
data:
  key: value
---
$strict: false
apiVersion: v1
kind: Secret
metadata:
  $strict: false
  name: credentials
stringData:
  token: secret
`)

	t.Log("Match each expected document to exactly one object and each kind to an exact set")
	result, err := compare(context.Background(), k8sClient, "test", expected)
	if err != nil {
		t.Fatalf("compare(): %v", err)
	}
	if len(result.actual) != 2 {
		t.Fatalf("compare() returned %d actual kinds, want 2", len(result.actual))
	}
}

func TestCompareRejectsAmbiguousGeneratedName(t *testing.T) {
	t.Log("Create two objects accepted by one generated-name glob")
	scheme := runtime.NewScheme()
	if err := corev1.AddToScheme(scheme); err != nil {
		t.Fatalf("add core types to scheme: %v", err)
	}
	k8sClient := fake.NewClientBuilder().WithScheme(scheme).WithObjects(
		&corev1.ConfigMap{ObjectMeta: metav1.ObjectMeta{Name: "generated-a", Namespace: "test"}},
		&corev1.ConfigMap{ObjectMeta: metav1.ObjectMeta{Name: "generated-b", Namespace: "test"}},
	).Build()
	expected := readTestDocuments(t, `
$strict: false
apiVersion: v1
kind: ConfigMap
metadata:
  $strict: false
  name: $glob:generated-*
`)

	t.Log("Reject the contract because one expected document must identify exactly one object")
	_, err := compare(context.Background(), k8sClient, "test", expected)
	if err == nil || !strings.Contains(err.Error(), "matches 2 objects") {
		t.Fatalf("compare() error = %v, want ambiguous match", err)
	}
}

func TestActualDocumentCountMeasuresObjectsAcrossKinds(t *testing.T) {
	t.Log("Build a comparison containing multiple objects for one kind and one object for another")
	comparison := comparison{actual: map[schema.GroupVersionKind][]actualDocument{
		{Group: "apps", Version: "v1", Kind: "Deployment"}: {{}, {}},
		{Version: "v1", Kind: "Service"}:                   {{}},
	}}

	t.Log("Count observed objects rather than only the number of represented kinds")
	if count := actualDocumentCount(comparison); count != 3 {
		t.Fatalf("actualDocumentCount() = %d, want 3", count)
	}
}

func readTestDocuments(t *testing.T, manifests string) []document {
	t.Helper()
	path := filepath.Join(t.TempDir(), "expected.yaml")
	if err := os.WriteFile(path, []byte(strings.TrimSpace(manifests)+"\n"), 0o600); err != nil {
		t.Fatalf("write expected manifests: %v", err)
	}
	documents, err := readDocuments(path)
	if err != nil {
		t.Fatalf("read expected manifests: %v", err)
	}
	return documents
}
