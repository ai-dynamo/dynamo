/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package golden

import (
	"bytes"
	"os"
	"sort"

	"go.yaml.in/yaml/v3"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

func writeAdapted(path string, comparison comparison) error {
	adapted := adaptDocuments(comparison)
	if len(adapted) == 0 {
		return os.WriteFile(path, nil, 0o644)
	}
	var output bytes.Buffer
	encoder := yaml.NewEncoder(&output)
	encoder.SetIndent(2)
	for i := range adapted {
		if err := encoder.Encode(&adapted[i]); err != nil {
			return err
		}
	}
	if err := encoder.Close(); err != nil {
		return err
	}
	return os.WriteFile(path, output.Bytes(), 0o644)
}

func adaptDocuments(comparison comparison) []yaml.Node {
	used := map[schema.GroupVersionKind]map[int]bool{}
	adapted := make([]yaml.Node, 0, len(comparison.expected))
	for i := range comparison.expected {
		expected := comparison.expected[i]
		actuals := comparison.actual[expected.gvk]
		if used[expected.gvk] == nil {
			used[expected.gvk] = map[int]bool{}
		}
		actualIndex := closestActual(&expected.node, actuals, used[expected.gvk])
		if actualIndex < 0 {
			continue
		}
		used[expected.gvk][actualIndex] = true
		updated := cloneNode(&expected.node)
		updated.Content[0] = adaptNode(updated.Content[0], documentRoot(&actuals[actualIndex].node))
		adapted = append(adapted, *updated)
	}
	gvks := make([]schema.GroupVersionKind, 0, len(comparison.actual))
	for gvk := range comparison.actual {
		gvks = append(gvks, gvk)
	}
	sort.Slice(gvks, func(i, j int) bool { return gvks[i].String() < gvks[j].String() })
	for _, gvk := range gvks {
		actuals := comparison.actual[gvk]
		for i := range actuals {
			if !used[gvk][i] {
				adapted = append(adapted, *cloneNode(&actuals[i].node))
			}
		}
	}
	return adapted
}

func closestActual(expected *yaml.Node, actuals []actualDocument, used map[int]bool) int {
	expectedRoot := documentRoot(expected)
	expectedName := objectNameNode(expectedRoot)
	for i := range actuals {
		if used[i] {
			continue
		}
		actualName := objectNameNode(documentRoot(&actuals[i].node))
		if expectedName != nil && actualName != nil && matchNode(expectedName, actualName, "$.metadata.name") == nil {
			return i
		}
	}
	for i := range actuals {
		if !used[i] {
			return i
		}
	}
	return -1
}

func objectNameNode(root *yaml.Node) *yaml.Node {
	metadata := mappingValue(root, "metadata")
	if metadata == nil || metadata.Kind != yaml.MappingNode {
		return nil
	}
	return mappingValue(metadata, "name")
}

func adaptNode(expected, actual *yaml.Node) *yaml.Node {
	if matchNode(expected, actual, "$") == nil {
		return expected
	}
	if directive, found, _ := mappingDirective(expected); found {
		if directive == directiveIgnore || directive == directiveExists {
			return expected
		}
		return replacementNode(expected, actual)
	}
	if expected.Kind != actual.Kind {
		return replacementNode(expected, actual)
	}
	switch expected.Kind {
	case yaml.MappingNode:
		return adaptMapping(expected, actual)
	case yaml.SequenceNode:
		common := min(len(expected.Content), len(actual.Content))
		for i := 0; i < common; i++ {
			expected.Content[i] = adaptNode(expected.Content[i], actual.Content[i])
		}
		if len(expected.Content) > len(actual.Content) {
			expected.Content = expected.Content[:len(actual.Content)]
		}
		for i := common; i < len(actual.Content); i++ {
			expected.Content = append(expected.Content, cloneNode(actual.Content[i]))
		}
		return expected
	default:
		return replacementNode(expected, actual)
	}
}

func adaptMapping(expected, actual *yaml.Node) *yaml.Node {
	strict, err := mappingStrict(expected)
	if err != nil {
		return replacementNode(expected, actual)
	}
	actualPairs := mappingPairs(actual)
	seen := map[string]bool{}
	content := make([]*yaml.Node, 0, len(expected.Content))
	for i := 0; i < len(expected.Content); i += 2 {
		key, value := expected.Content[i], expected.Content[i+1]
		if key.Value == directiveStrict {
			content = append(content, key, value)
			continue
		}
		actualName := key.Value
		if actualName == "$$strict" {
			actualName = directiveStrict
		}
		actualPair, exists := actualPairs[actualName]
		if !exists {
			if strict {
				continue
			}
			content = append(content, key, scalarReplacement(value, directiveNotExists))
			continue
		}
		seen[actualName] = true
		content = append(content, key, adaptNode(value, actualPair.value))
	}
	if strict {
		for i := 0; i < len(actual.Content); i += 2 {
			name := actual.Content[i].Value
			if seen[name] {
				continue
			}
			content = append(content, cloneNode(actual.Content[i]), cloneNode(actual.Content[i+1]))
		}
	}
	expected.Content = content
	return expected
}

type mappingPair struct {
	key   *yaml.Node
	value *yaml.Node
}

func mappingPairs(node *yaml.Node) map[string]mappingPair {
	pairs := map[string]mappingPair{}
	for i := 0; i < len(node.Content); i += 2 {
		pairs[node.Content[i].Value] = mappingPair{key: node.Content[i], value: node.Content[i+1]}
	}
	return pairs
}

func mappingValue(node *yaml.Node, name string) *yaml.Node {
	if node == nil || node.Kind != yaml.MappingNode {
		return nil
	}
	for i := 0; i < len(node.Content); i += 2 {
		if node.Content[i].Value == name {
			return node.Content[i+1]
		}
	}
	return nil
}

func replacementNode(expected, actual *yaml.Node) *yaml.Node {
	replacement := cloneNode(actual)
	replacement.HeadComment = expected.HeadComment
	replacement.LineComment = expected.LineComment
	replacement.FootComment = expected.FootComment
	return replacement
}

func scalarReplacement(expected *yaml.Node, value string) *yaml.Node {
	replacement := &yaml.Node{Kind: yaml.ScalarNode, Tag: yamlStringTag, Value: value}
	replacement.HeadComment = expected.HeadComment
	replacement.LineComment = expected.LineComment
	replacement.FootComment = expected.FootComment
	return replacement
}

func cloneNode(node *yaml.Node) *yaml.Node {
	if node == nil {
		return nil
	}
	clone := *node
	clone.Content = make([]*yaml.Node, len(node.Content))
	for i := range node.Content {
		clone.Content[i] = cloneNode(node.Content[i])
	}
	if node.Alias != nil {
		clone.Alias = cloneNode(node.Alias)
	}
	return &clone
}

func min(left, right int) int {
	if left < right {
		return left
	}
	return right
}
