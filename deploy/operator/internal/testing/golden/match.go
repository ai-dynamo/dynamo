/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Package golden matches namespaced Kubernetes manifests against YAML contracts.
package golden

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"sort"
	"strings"
	"testing"
	"time"

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

// MatchManifests waits until the objects in namespace exactly match the YAML
// documents in expectedPath. A final mismatch writes expectedPath + ".new".
func MatchManifests(t testing.TB, k8sClient client.Client, namespace, expectedPath string) {
	t.Helper()
	expected, err := readDocuments(expectedPath)
	if err != nil {
		t.Fatalf("read golden manifests %q: %v", expectedPath, err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), matchTimeout)
	defer cancel()
	var last comparison
	var lastErr error
	for {
		candidate, candidateErr := compare(ctx, k8sClient, namespace, expected)
		if candidateErr == nil {
			return
		}
		if actualDocumentCount(candidate) >= actualDocumentCount(last) {
			last = candidate
			lastErr = candidateErr
		}
		select {
		case <-ctx.Done():
			newPath := expectedPath + ".new"
			if err := writeAdapted(newPath, last); err != nil {
				t.Fatalf("golden manifests do not match: %v; write %q: %v", lastErr, newPath, err)
			}
			t.Fatalf("golden manifests do not match: %v; adapted manifests written to %q", lastErr, newPath)
		case <-time.After(100 * time.Millisecond):
		}
	}
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
		if len(want) != len(actual) {
			mismatches = append(mismatches, fmt.Sprintf("%s: found %d objects, want %d", gvk, len(actual), len(want)))
		}
		claimed := map[int]int{}
		for i, expectedDocument := range want {
			var matches []int
			var closest string
			for j := range actual {
				if err := matchNode(documentRoot(&expectedDocument.node), documentRoot(&actual[j].node), "$"); err == nil {
					matches = append(matches, j)
				} else if closest == "" {
					closest = err.Error()
				}
			}
			if len(matches) != 1 {
				detail := ""
				if len(matches) == 0 && closest != "" {
					detail = ": " + closest
				}
				mismatches = append(mismatches, fmt.Sprintf("%s document %d matches %d objects, want exactly one%s", gvk, i+1, len(matches), detail))
				continue
			}
			if previous, found := claimed[matches[0]]; found {
				mismatches = append(mismatches, fmt.Sprintf("%s documents %d and %d match the same object", gvk, previous+1, i+1))
				continue
			}
			claimed[matches[0]] = i
		}
	}
	if len(mismatches) > 0 {
		return result, errors.New(strings.Join(mismatches, "; "))
	}
	return result, nil
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
