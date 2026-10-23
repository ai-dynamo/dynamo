// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package golden

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"testing"

	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/util/retry"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

// MatchCapturedManifests compares the entire observed ownership graph, including
// resource kinds absent from the golden. Mismatches emit a reviewable .new file.
func MatchCapturedManifests(t testing.TB, objects []unstructured.Unstructured, path string) {
	t.Helper()
	MatchCapturedManifestsWithVariables(t, objects, path, nil)
}

// MatchCapturedManifestsWithVariables binds exact values before comparing the complete graph.
func MatchCapturedManifestsWithVariables(t testing.TB, objects []unstructured.Unstructured, path string, variables Variables) {
	t.Helper()
	expected, err := readDocuments(path)
	if err != nil {
		t.Fatalf("read golden: %v", err)
	}
	if err := resolveVariables(expected, variables); err != nil {
		t.Fatalf("resolve golden variables: %v", err)
	}
	result := comparison{expected: expected, actual: map[schema.GroupVersionKind][]actualDocument{}}
	for _, object := range objects {
		object = *object.DeepCopy()
		// The API server recreates managedFields; exclude that bookkeeping
		// from comparisons just as the fixture writers do.
		unstructured.RemoveNestedField(object.Object, "metadata", "managedFields")
		node, err := objectNode(object.Object)
		if err != nil {
			t.Fatal(err)
		}
		gvk := object.GroupVersionKind()
		result.actual[gvk] = append(result.actual[gvk], actualDocument{node: node, object: object})
	}
	matched := map[string]bool{}
	var diffs []string
	for _, want := range expected {
		found := false
		for _, actual := range result.actual[want.gvk] {
			candidates := nameCandidates(&want.node, []actualDocument{actual})
			if len(candidates) == 0 {
				continue
			}
			key := actual.object.GetAPIVersion() + "/" + actual.object.GetKind() + "/" + actual.object.GetName()
			if matched[key] {
				diffs = append(diffs, "duplicate golden object "+key)
			}
			matched[key] = true
			found = true
			if err := matchNode(documentRoot(&want.node), documentRoot(&actual.node), "$"); err != nil {
				diffs = append(diffs, documentDiff(&want.node, &actual.node))
			}
		}
		if !found {
			diffs = append(diffs, "missing "+expectedDescription(want, ""))
		}
	}
	if len(matched) != len(objects) {
		diffs = append(diffs, "observed ownership graph has unexpected objects")
	}
	if len(diffs) > 0 {
		if err := writeAdapted(path+".new", result); err != nil {
			t.Errorf("write golden candidate: %v", err)
		}
		t.Errorf("complete resource golden differs; inspect %s.new:\n%s", path, strings.Join(diffs, "\n"))
	}
}

// ReadManifests preserves fields that are not represented in the current Go API.
func ReadManifests(path string) ([]unstructured.Unstructured, error) {
	documents, err := readDocuments(path)
	if err != nil {
		return nil, err
	}
	return manifestObjects(documents)
}

// ReadManifestsWithVariables resolves exact golden bindings while preserving unknown API fields.
func ReadManifestsWithVariables(path string, variables Variables) ([]unstructured.Unstructured, error) {
	documents, err := readDocuments(path)
	if err != nil {
		return nil, err
	}
	if err := resolveVariables(documents, variables); err != nil {
		return nil, err
	}
	return manifestObjects(documents)
}

func manifestObjects(documents []document) ([]unstructured.Unstructured, error) {
	objects := make([]unstructured.Unstructured, len(documents))
	for i := range documents {
		// Decode once with the golden YAML parser. Re-parsing YAML 1.2 strings
		// through a YAML 1.1 converter would turn env values such as "n" into false.
		var raw map[string]any
		if err := documentRoot(&documents[i].node).Decode(&raw); err != nil {
			return nil, err
		}
		jsonData, err := json.Marshal(raw)
		if err != nil {
			return nil, err
		}
		if err := objects[i].UnmarshalJSON(jsonData); err != nil {
			return nil, err
		}
		objects[i].SetGroupVersionKind(documents[i].gvk)
	}
	return objects, nil
}

// WriteManifests writes concrete API objects without managedFields bookkeeping.
func WriteManifests(path string, objects []unstructured.Unstructured) error {
	var out strings.Builder
	out.WriteString("# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.\n# SPDX-License-Identifier: Apache-2.0\n# Generated from observed API objects; do not edit historical inputs.\n")
	for i := range objects {
		if i > 0 {
			out.WriteString("---\n")
		}
		object := objects[i].DeepCopy()
		unstructured.RemoveNestedField(object.Object, "metadata", "managedFields")
		node, err := objectNode(object.Object)
		if err != nil {
			return err
		}
		data, err := encodeDocument(&node)
		if err != nil {
			return err
		}
		out.WriteString(data)
	}
	return os.WriteFile(path, []byte(out.String()), 0644)
}

// ApplySnapshot reconstructs an observed resource graph before candidate admission
// starts. The caller owns the namespace and cleans it up on partial failure.
func ApplySnapshot(ctx context.Context, c client.Client, namespace string, objects []unstructured.Unstructured) error {
	created := make([]unstructured.Unstructured, len(objects))
	byUID := map[types.UID]int{}
	for i := range objects {
		old := &objects[i]
		if old.GetNamespace() != namespace || old.GetUID() == "" || old.GetDeletionTimestamp() != nil {
			return fmt.Errorf("invalid snapshot identity for %s %s/%s", old.GetKind(), old.GetNamespace(), old.GetName())
		}
		if _, exists := byUID[old.GetUID()]; exists {
			return fmt.Errorf("duplicate captured UID %s", old.GetUID())
		}
		byUID[old.GetUID()] = i
	}
	// Validate ownership before creating anything; an omitted owner is not optional.
	for i := range objects {
		for _, owner := range objects[i].GetOwnerReferences() {
			index, found := byUID[owner.UID]
			if !found {
				return fmt.Errorf("unresolved owner %s %s", owner.Kind, owner.Name)
			}
			target := &objects[index]
			if target.GetName() != owner.Name || target.GetAPIVersion() != owner.APIVersion || target.GetKind() != owner.Kind {
				return fmt.Errorf("owner identity disagrees with UID %s", owner.UID)
			}
		}
	}
	for i := range objects {
		object := objects[i].DeepCopy()
		for _, key := range []string{"uid", "resourceVersion", "generation", "creationTimestamp", "managedFields"} {
			unstructured.RemoveNestedField(object.Object, "metadata", key)
		}
		object.SetOwnerReferences(nil)
		unstructured.RemoveNestedField(object.Object, "status")
		// Cluster-assigned service addresses cannot be moved between API servers.
		if object.GetAPIVersion() == "v1" && object.GetKind() == "Service" {
			ip, _, _ := unstructured.NestedString(object.Object, "spec", "clusterIP")
			if ip != "None" {
				unstructured.RemoveNestedField(object.Object, "spec", "clusterIP")
				unstructured.RemoveNestedField(object.Object, "spec", "clusterIPs")
			}
		}
		if err := c.Create(ctx, object); err != nil {
			return fmt.Errorf("create %s %s: %w", object.GetKind(), object.GetName(), err)
		}
		created[i] = *object
	}
	for i := range objects {
		object := &created[i]
		refs := objects[i].GetOwnerReferences()
		for j := range refs {
			refs[j].UID = created[byUID[refs[j].UID]].GetUID()
		}
		status, found, err := unstructured.NestedMap(objects[i].Object, "status")
		if err != nil {
			return err
		}
		// Native workload controllers can update status during snapshot reconstruction.
		// Retry against the latest RV without changing the historical owners or status.
		if err := retry.RetryOnConflict(retry.DefaultRetry, func() error {
			if err := c.Get(ctx, client.ObjectKeyFromObject(object), object); err != nil {
				return err
			}
			if len(refs) > 0 {
				object.SetOwnerReferences(refs)
				if err := c.Update(ctx, object); err != nil {
					return err
				}
			}
			if found {
				object.Object["status"] = status
				return c.Status().Update(ctx, object)
			}
			return nil
		}); err != nil {
			return fmt.Errorf("restore owners/status of %s %s: %w", object.GetKind(), object.GetName(), err)
		}
	}
	return nil
}
