/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package golden

import (
	"testing"

	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

// ApplyManifests creates every YAML document in path as an unstructured
// object. Namespaced objects are scoped to namespace for test isolation.
func ApplyManifests(
	t testing.TB,
	path string,
	k8sClient client.Client,
	namespace string,
) []unstructured.Unstructured {
	t.Helper()
	documents, err := readDocuments(path)
	if err != nil {
		t.Fatalf("read input manifests %q: %v", path, err)
	}

	objects := make([]unstructured.Unstructured, 0, len(documents))
	for i := range documents {
		object := unstructured.Unstructured{}
		if err := documentRoot(&documents[i].node).Decode(&object.Object); err != nil {
			t.Fatalf("decode input manifest %q document %d: %v", path, i+1, err)
		}
		object.SetGroupVersionKind(documents[i].gvk)
		namespaced, err := k8sClient.IsObjectNamespaced(&object)
		if err != nil {
			t.Fatalf("determine scope of input manifest %q document %d: %v", path, i+1, err)
		}
		if namespaced {
			if object.GetNamespace() != "" && object.GetNamespace() != namespace {
				t.Fatalf(
					"input manifest %q document %d declares namespace %q, want %q",
					path,
					i+1,
					object.GetNamespace(),
					namespace,
				)
			}
			object.SetNamespace(namespace)
		}
		if err := k8sClient.Create(t.Context(), &object); err != nil {
			t.Fatalf("apply input manifest %q document %d: %v", path, i+1, err)
		}
		objects = append(objects, object)
	}
	return objects
}
