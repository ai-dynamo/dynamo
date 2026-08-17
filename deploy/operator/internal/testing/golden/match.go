/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Package golden matches namespaced Kubernetes manifests against YAML contracts.
package golden

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"sort"
	"strings"
	"testing"
	"time"

	dynamotesting "github.com/ai-dynamo/dynamo/deploy/operator/internal/testing"
	"github.com/pmezard/go-difflib/difflib"
	"go.yaml.in/yaml/v3"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

const matchTimeout = 30 * time.Second

type document struct {
	node yaml.Node
	gvk  schema.GroupVersionKind
}

type actualDocument struct {
	node   yaml.Node
	object unstructured.Unstructured
}

type comparison struct {
	expected []document
	actual   map[schema.GroupVersionKind][]actualDocument
}

// EventuallyMatchManifests waits until the objects in namespace exactly match
// the YAML documents in expectedPath. A final mismatch writes expectedPath + ".new".
func EventuallyMatchManifests(t testing.TB, k8sClient client.Client, namespace, expectedPath string) {
	t.Helper()
	expected, err := readDocuments(expectedPath)
	if err != nil {
		t.Fatalf("read golden manifests %q: %v", expectedPath, err)
	}

	var last comparison
	var lastErr error
	matched := false
	newPath := expectedPath + ".new"
	t.Cleanup(func() {
		if matched {
			return
		}
		if err := writeAdapted(newPath, last); err != nil {
			t.Errorf("golden manifests do not match: %v; write %q: %v", lastErr, newPath, err)
			return
		}
		t.Logf("golden manifests do not match: %v; adapted manifests written to %q", lastErr, newPath)
	})

	dynamotesting.Eventually(t, func() (bool, string) {
		candidate, candidateErr := compare(t.Context(), k8sClient, namespace, expected)
		if candidateErr == nil {
			matched = true
			return true, "manifests match"
		}
		if actualDocumentCount(candidate) >= actualDocumentCount(last) {
			last = candidate
			lastErr = candidateErr
		}
		return false, candidateErr.Error()
	}, matchTimeout, 100*time.Millisecond, "golden manifests do not match; adapted manifests will be written to %q", newPath)
}

func actualDocumentCount(comparison comparison) int {
	count := 0
	for _, documents := range comparison.actual {
		count += len(documents)
	}
	return count
}

func readDocuments(path string) ([]document, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer func() { _ = file.Close() }()
	decoder := yaml.NewDecoder(file)
	var documents []document
	for {
		var node yaml.Node
		if err := decoder.Decode(&node); err != nil {
			if errors.Is(err, io.EOF) {
				break
			}
			return nil, err
		}
		if len(node.Content) == 0 {
			continue
		}
		gvk, err := documentGVK(&node)
		if err != nil {
			return nil, fmt.Errorf("document %d: %w", len(documents)+1, err)
		}
		documents = append(documents, document{node: node, gvk: gvk})
	}
	if len(documents) == 0 {
		return nil, errors.New("golden manifest file contains no objects")
	}
	return documents, nil
}

func documentGVK(document *yaml.Node) (schema.GroupVersionKind, error) {
	root := documentRoot(document)
	if root == nil || root.Kind != yaml.MappingNode {
		return schema.GroupVersionKind{}, errors.New("manifest root must be a mapping")
	}
	apiVersion, found := mappingScalar(root, "apiVersion")
	if !found {
		return schema.GroupVersionKind{}, errors.New("apiVersion is required")
	}
	kind, found := mappingScalar(root, "kind")
	if !found {
		return schema.GroupVersionKind{}, errors.New("kind is required")
	}
	groupVersion, err := schema.ParseGroupVersion(apiVersion)
	if err != nil {
		return schema.GroupVersionKind{}, fmt.Errorf("parse apiVersion %q: %w", apiVersion, err)
	}
	return groupVersion.WithKind(kind), nil
}

func compare(ctx context.Context, k8sClient client.Client, namespace string, expected []document) (comparison, error) {
	result := comparison{expected: expected, actual: map[schema.GroupVersionKind][]actualDocument{}}
	groups := map[schema.GroupVersionKind][]document{}
	for _, manifest := range expected {
		groups[manifest.gvk] = append(groups[manifest.gvk], manifest)
	}
	gvks := make([]schema.GroupVersionKind, 0, len(groups))
	for gvk := range groups {
		gvks = append(gvks, gvk)
	}
	sort.Slice(gvks, func(i, j int) bool { return gvks[i].String() < gvks[j].String() })

	var mismatches []string
	for _, gvk := range gvks {
		actual, err := listActual(ctx, k8sClient, namespace, gvk)
		if err != nil {
			return result, fmt.Errorf("list %s: %w", gvk, err)
		}
		result.actual[gvk] = actual
		want := groups[gvk]
		claimed := map[int]int{}
		allExpectedMatched := true
		for i, expectedDocument := range want {
			var matches []int
			for j := range actual {
				if err := matchNode(documentRoot(&expectedDocument.node), documentRoot(&actual[j].node), "$"); err == nil {
					matches = append(matches, j)
				}
			}
			description := expectedDescription(expectedDocument, namespace)
			if len(matches) == 0 {
				allExpectedMatched = false
				candidates := nameCandidates(&expectedDocument.node, actual)
				if len(candidates) == 0 {
					mismatches = append(mismatches, fmt.Sprintf("%s has not appeared", description))
					continue
				}
				candidate := actual[candidates[0]]
				diff := documentDiff(&expectedDocument.node, &candidate.node)
				if len(candidates) == 1 {
					mismatches = append(mismatches, fmt.Sprintf("%s does not match:\n%s", description, diff))
					continue
				}
				mismatches = append(mismatches, fmt.Sprintf(
					"%s has %d name-matching candidates; %s does not match:\n%s",
					description, len(candidates), actualDescription(candidate.object), diff,
				))
				continue
			}
			if len(matches) > 1 {
				allExpectedMatched = false
				names := make([]string, 0, len(matches))
				for _, match := range matches {
					names = append(names, actualDescription(actual[match].object))
				}
				mismatches = append(mismatches, fmt.Sprintf("%s matches %d objects, want exactly one: %s", description, len(matches), strings.Join(names, ", ")))
				continue
			}
			if previous, found := claimed[matches[0]]; found {
				allExpectedMatched = false
				mismatches = append(mismatches, fmt.Sprintf(
					"%s and document %d match the same object", description, previous+1,
				))
				continue
			}
			claimed[matches[0]] = i
		}
		if allExpectedMatched {
			for i := range actual {
				if _, found := claimed[i]; !found {
					mismatches = append(mismatches, fmt.Sprintf("unexpected %s is present", actualDescription(actual[i].object)))
				}
			}
		}
	}
	if len(mismatches) > 0 {
		return result, errors.New(strings.Join(mismatches, "; "))
	}
	return result, nil
}

func expectedDescription(expected document, namespace string) string {
	name := "<unknown>"
	if nameNode := objectNameNode(documentRoot(&expected.node)); nameNode != nil && nameNode.Kind == yaml.ScalarNode {
		name = nameNode.Value
	}
	return fmt.Sprintf("%s %s/%s", expected.gvk.Kind, namespace, name)
}

func actualDescription(actual unstructured.Unstructured) string {
	return fmt.Sprintf("%s %s/%s", actual.GetKind(), actual.GetNamespace(), actual.GetName())
}

func nameCandidates(expected *yaml.Node, actual []actualDocument) []int {
	expectedName := objectNameNode(documentRoot(expected))
	if expectedName == nil {
		candidates := make([]int, len(actual))
		for i := range actual {
			candidates[i] = i
		}
		return candidates
	}
	var candidates []int
	for i := range actual {
		actualName := objectNameNode(documentRoot(&actual[i].node))
		if actualName != nil && matchNode(expectedName, actualName, "$.metadata.name") == nil {
			candidates = append(candidates, i)
		}
	}
	return candidates
}

func documentDiff(expected, actual *yaml.Node) string {
	adapted := cloneNode(expected)
	adaptedRoot := adaptNode(documentRoot(adapted), documentRoot(actual))
	if adapted.Kind == yaml.DocumentNode {
		adapted.Content[0] = adaptedRoot
	} else {
		adapted = adaptedRoot
	}
	expectedYAML, expectedErr := encodeDocument(expected)
	actualYAML, actualErr := encodeDocument(adapted)
	if expectedErr != nil || actualErr != nil {
		if err := matchNode(documentRoot(expected), documentRoot(actual), "$"); err != nil {
			return err.Error()
		}
		return "manifest differs"
	}
	diff, err := difflib.GetUnifiedDiffString(difflib.UnifiedDiff{
		A:        difflib.SplitLines(expectedYAML),
		B:        difflib.SplitLines(actualYAML),
		FromFile: "expected",
		ToFile:   "actual",
		Context:  3,
	})
	if err != nil || diff == "" {
		if matchErr := matchNode(documentRoot(expected), documentRoot(actual), "$"); matchErr != nil {
			return matchErr.Error()
		}
		return "manifest differs"
	}
	return strings.TrimSpace(diff)
}

func encodeDocument(document *yaml.Node) (string, error) {
	var output bytes.Buffer
	encoder := yaml.NewEncoder(&output)
	encoder.SetIndent(2)
	if err := encoder.Encode(document); err != nil {
		return "", err
	}
	if err := encoder.Close(); err != nil {
		return "", err
	}
	return output.String(), nil
}

func listActual(ctx context.Context, k8sClient client.Client, namespace string, gvk schema.GroupVersionKind) ([]actualDocument, error) {
	list := &unstructured.UnstructuredList{}
	list.SetGroupVersionKind(gvk.GroupVersion().WithKind(gvk.Kind + "List"))
	if err := k8sClient.List(ctx, list, client.InNamespace(namespace)); err != nil {
		return nil, err
	}
	actual := make([]actualDocument, 0, len(list.Items))
	for i := range list.Items {
		object := list.Items[i]
		object.SetGroupVersionKind(gvk)
		node, err := objectNode(object.Object)
		if err != nil {
			return nil, fmt.Errorf("encode %s %q: %w", gvk.Kind, object.GetName(), err)
		}
		actual = append(actual, actualDocument{node: node, object: object})
	}
	sort.Slice(actual, func(i, j int) bool { return actual[i].object.GetName() < actual[j].object.GetName() })
	return actual, nil
}

func objectNode(object map[string]any) (yaml.Node, error) {
	data, err := yaml.Marshal(object)
	if err != nil {
		return yaml.Node{}, err
	}
	var node yaml.Node
	if err := yaml.Unmarshal(data, &node); err != nil {
		return yaml.Node{}, err
	}
	return node, nil
}
