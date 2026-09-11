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
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

type delayedListClient struct {
	client.Client
	calls int
}

func (c *delayedListClient) List(ctx context.Context, list client.ObjectList, options ...client.ListOption) error {
	c.calls++
	if c.calls == 1 {
		list.(*unstructured.UnstructuredList).Items = nil
		return nil
	}
	return c.Client.List(ctx, list, options...)
}

func TestEventuallyMatchManifestsWaitsForObject(t *testing.T) {
	t.Log("Create an expected manifest and an API client that hides the object from the first list")
	expectedPath := filepath.Join(t.TempDir(), "expected.yaml")
	if err := os.WriteFile(expectedPath, []byte(`
$strict: false
apiVersion: v1
kind: ConfigMap
metadata:
  $strict: false
  name: generated
data:
  key: value
`), 0o600); err != nil {
		t.Fatalf("write expected manifests: %v", err)
	}
	scheme := runtime.NewScheme()
	if err := corev1.AddToScheme(scheme); err != nil {
		t.Fatalf("add core types to scheme: %v", err)
	}
	k8sClient := &delayedListClient{Client: fake.NewClientBuilder().WithScheme(scheme).WithObjects(
		&corev1.ConfigMap{ObjectMeta: metav1.ObjectMeta{Name: "generated", Namespace: "test"}, Data: map[string]string{"key": "value"}},
	).Build()}

	t.Log("Eventually match after the object appears on a later list")
	EventuallyMatchManifests(t, k8sClient, "test", expectedPath)
	if k8sClient.calls < 2 {
		t.Fatalf("EventuallyMatchManifests listed %d times, want at least 2", k8sClient.calls)
	}
}

func TestEventuallyMatchManifestsResolvesVariables(t *testing.T) {
	t.Log("Create a manifest that binds its expected value through a golden variable")
	expectedPath := filepath.Join(t.TempDir(), "expected.yaml")
	if err := os.WriteFile(expectedPath, []byte(`
$strict: false
apiVersion: v1
kind: ConfigMap
metadata:
  $strict: false
  name: generated
data:
  worker-hash: $var:worker-hash
`), 0o600); err != nil {
		t.Fatalf("write expected manifests: %v", err)
	}
	scheme := runtime.NewScheme()
	if err := corev1.AddToScheme(scheme); err != nil {
		t.Fatalf("add core types to scheme: %v", err)
	}
	k8sClient := fake.NewClientBuilder().WithScheme(scheme).WithObjects(
		&corev1.ConfigMap{ObjectMeta: metav1.ObjectMeta{Name: "generated", Namespace: "test"}, Data: map[string]string{"worker-hash": "aabbccdd"}},
	).Build()

	t.Log("Match the value exactly after resolving the worker-hash variable")
	EventuallyMatchManifestsWithVariables(t, k8sClient, "test", expectedPath, Variables{"worker-hash": "aabbccdd"})
}

func TestResolveVariablesRejectsUnknownVariable(t *testing.T) {
	t.Log("Declare an expected manifest with a variable that has no bound value")
	expected := readTestDocuments(t, `
$strict: false
apiVersion: v1
kind: ConfigMap
metadata:
  $strict: false
  name: generated
data:
  worker-hash: $var:worker-hash
`)

	t.Log("Reject the unresolved variable before the matcher attempts an API comparison")
	err := resolveVariables(expected, Variables{})
	if err == nil || !strings.Contains(err.Error(), `unknown variable "worker-hash"`) {
		t.Fatalf("resolveVariables() error = %v, want unknown worker-hash variable", err)
	}
}

func TestResolveVariablesMatchExactStrings(t *testing.T) {
	t.Log("Resolve a variable whose value resembles a golden pattern directive")
	expected := readTestDocuments(t, `
$strict: false
apiVersion: v1
kind: ConfigMap
metadata:
  $strict: false
  name: generated
data:
  value: $var:value
`)
	if err := resolveVariables(expected, Variables{"value": "$pattern:.*"}); err != nil {
		t.Fatalf("resolveVariables() error = %v", err)
	}

	t.Log("Require the substituted value literally rather than treating it as a matcher directive")
	actual := readTestDocuments(t, `
$strict: false
apiVersion: v1
kind: ConfigMap
metadata:
  $strict: false
  name: generated
data:
  value: any-value
`)
	if err := matchNode(documentRoot(&expected[0].node), documentRoot(&actual[0].node), "$"); err == nil {
		t.Fatal("matchNode() succeeded, want exact variable-value mismatch")
	}
}

func TestDocumentDiffPreservesVariableDirectives(t *testing.T) {
	t.Log("Resolve a worker-hash variable in a contract with an unrelated mismatch")
	expected := readTestDocuments(t, `
$strict: false
apiVersion: v1
kind: ConfigMap
metadata:
  $strict: false
  name: generated
data:
  worker-hash: $var:worker-hash
  other: expected
`)
	if err := resolveVariables(expected, Variables{"worker-hash": "aabbccdd"}); err != nil {
		t.Fatalf("resolveVariables() error = %v", err)
	}
	actual := readTestDocuments(t, `
$strict: false
apiVersion: v1
kind: ConfigMap
metadata:
  $strict: false
  name: generated
data:
  worker-hash: aabbccdd
  other: actual
`)

	t.Log("Keep the stable variable directive in the human-readable mismatch diff")
	diff := documentDiff(&expected[0].node, &actual[0].node)
	if !strings.Contains(diff, "$var:worker-hash") || strings.Contains(diff, variableValueTagPrefix) {
		t.Fatalf("documentDiff() = %q, want stable worker-hash directive without internal tag", diff)
	}
}

func TestCompareDescribesMissingObject(t *testing.T) {
	t.Log("Create an expected named object with no actual object")
	scheme := runtime.NewScheme()
	if err := corev1.AddToScheme(scheme); err != nil {
		t.Fatalf("add core types to scheme: %v", err)
	}
	expected := readTestDocuments(t, `
$strict: false
apiVersion: v1
kind: ConfigMap
metadata:
  $strict: false
  name: wanted
`)

	t.Log("Describe the exact object whose appearance is awaited")
	_, err := compare(context.Background(), fake.NewClientBuilder().WithScheme(scheme).Build(), "test", expected)
	if err == nil || !strings.Contains(err.Error(), "ConfigMap test/wanted has not appeared") {
		t.Fatalf("compare() error = %v, want named missing-object reason", err)
	}
}

func TestCompareIncludesManifestDiff(t *testing.T) {
	t.Log("Create an actual object whose data differs from the expected contract")
	scheme := runtime.NewScheme()
	if err := corev1.AddToScheme(scheme); err != nil {
		t.Fatalf("add core types to scheme: %v", err)
	}
	k8sClient := fake.NewClientBuilder().WithScheme(scheme).WithObjects(
		&corev1.ConfigMap{ObjectMeta: metav1.ObjectMeta{Name: "wanted", Namespace: "test"}, Data: map[string]string{"key": "actual"}},
	).Build()
	expected := readTestDocuments(t, `
$strict: false
apiVersion: v1
kind: ConfigMap
metadata:
  $strict: false
  name: wanted
data:
  key: expected
`)

	t.Log("Include the object identity and minimal expected-to-actual manifest diff")
	_, err := compare(context.Background(), k8sClient, "test", expected)
	if err == nil {
		t.Fatal("compare() succeeded, want manifest mismatch")
	}
	for _, fragment := range []string{
		"ConfigMap test/wanted does not match:",
		"--- expected",
		"+++ actual",
		"-  key: expected",
		"+  key: actual",
	} {
		if !strings.Contains(err.Error(), fragment) {
			t.Fatalf("compare() error does not contain %q:\n%s", fragment, err)
		}
	}
}

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
