/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package golden

import (
	"fmt"
	"path"
	"reflect"
	"regexp"
	"strconv"
	"strings"

	"go.yaml.in/yaml/v3"
)

const (
	directiveIgnore    = "$ignore"
	directiveExists    = "$exists"
	directiveNotExists = "$notexists"
	directiveStrict    = "$strict"
	yamlStringTag      = "!!str"
)

func matchNode(expected, actual *yaml.Node, fieldPath string) error {
	expected = dereference(expected)
	actual = dereference(actual)
	if directive, found, err := mappingDirective(expected); err != nil {
		return fmt.Errorf("%s: %w", fieldPath, err)
	} else if found {
		switch directive {
		case directiveIgnore, directiveExists:
			if actual.Kind != yaml.MappingNode {
				return fmt.Errorf("%s: expected a mapping", fieldPath)
			}
			return nil
		case directiveNotExists:
			return fmt.Errorf("%s: field exists", fieldPath)
		}
	}
	if expected.Kind != actual.Kind {
		return fmt.Errorf("%s: node kind is %d, want %d", fieldPath, actual.Kind, expected.Kind)
	}
	switch expected.Kind {
	case yaml.MappingNode:
		return matchMapping(expected, actual, fieldPath)
	case yaml.SequenceNode:
		if len(expected.Content) != len(actual.Content) {
			return fmt.Errorf("%s: sequence has %d entries, want %d", fieldPath, len(actual.Content), len(expected.Content))
		}
		for i := range expected.Content {
			if err := matchNode(expected.Content[i], actual.Content[i], fmt.Sprintf("%s[%d]", fieldPath, i)); err != nil {
				return err
			}
		}
		return nil
	case yaml.ScalarNode:
		return matchScalar(expected, actual, fieldPath)
	default:
		return fmt.Errorf("%s: unsupported YAML node kind %d", fieldPath, expected.Kind)
	}
}

func matchMapping(expected, actual *yaml.Node, fieldPath string) error {
	strict, err := mappingStrict(expected)
	if err != nil {
		return fmt.Errorf("%s: %w", fieldPath, err)
	}
	expectedFields := mappingFields(expected, true)
	actualFields := mappingFields(actual, false)
	for name, expectedValue := range expectedFields {
		actualValue, exists := actualFields[name]
		if !exists {
			if isNotExistsDirective(expectedValue) {
				continue
			}
			return fmt.Errorf("%s.%s: field does not exist", fieldPath, name)
		}
		if isNotExistsDirective(expectedValue) {
			return fmt.Errorf("%s.%s: field exists", fieldPath, name)
		}
		if err := matchNode(expectedValue, actualValue, fieldPath+"."+name); err != nil {
			return err
		}
	}
	if strict {
		for name := range actualFields {
			value, exists := expectedFields[name]
			if !exists || isNotExistsDirective(value) {
				return fmt.Errorf("%s.%s: unexpected field", fieldPath, name)
			}
		}
	}
	return nil
}

func matchScalar(expected, actual *yaml.Node, fieldPath string) error {
	if expected.Tag == yamlStringTag {
		switch expected.Value {
		case directiveIgnore, directiveExists:
			if actual.Kind != yaml.ScalarNode {
				return fmt.Errorf("%s: expected a scalar", fieldPath)
			}
			return nil
		case directiveNotExists:
			return fmt.Errorf("%s: field exists", fieldPath)
		}
		if strings.HasPrefix(expected.Value, "$glob:") {
			if actual.Tag != yamlStringTag {
				return fmt.Errorf("%s: glob requires a string", fieldPath)
			}
			matched, err := path.Match(strings.TrimPrefix(expected.Value, "$glob:"), actual.Value)
			if err != nil {
				return fmt.Errorf("%s: invalid glob: %w", fieldPath, err)
			}
			if !matched {
				return fmt.Errorf("%s: %q does not match glob %q", fieldPath, actual.Value, expected.Value)
			}
			return nil
		}
		if strings.HasPrefix(expected.Value, "$pattern:") {
			if actual.Tag != yamlStringTag {
				return fmt.Errorf("%s: pattern requires a string", fieldPath)
			}
			pattern := strings.TrimPrefix(expected.Value, "$pattern:")
			matched, err := regexp.MatchString("^(?:"+pattern+")$", actual.Value)
			if err != nil {
				return fmt.Errorf("%s: invalid pattern: %w", fieldPath, err)
			}
			if !matched {
				return fmt.Errorf("%s: %q does not match pattern %q", fieldPath, actual.Value, pattern)
			}
			return nil
		}
	}
	var expectedValue, actualValue any
	if err := expected.Decode(&expectedValue); err != nil {
		return fmt.Errorf("%s: decode expected scalar: %w", fieldPath, err)
	}
	if err := actual.Decode(&actualValue); err != nil {
		return fmt.Errorf("%s: decode actual scalar: %w", fieldPath, err)
	}
	if !reflect.DeepEqual(expectedValue, actualValue) {
		return fmt.Errorf("%s: value is %v, want %v", fieldPath, actualValue, expectedValue)
	}
	return nil
}

func mappingStrict(node *yaml.Node) (bool, error) {
	for i := 0; i < len(node.Content); i += 2 {
		if node.Content[i].Value != directiveStrict {
			continue
		}
		value := node.Content[i+1]
		if value.Kind != yaml.ScalarNode || value.Tag != "!!bool" {
			return false, errorsForDirective(directiveStrict, "must be a boolean")
		}
		strict, err := strconv.ParseBool(value.Value)
		if err != nil {
			return false, errorsForDirective(directiveStrict, "must be a boolean")
		}
		return strict, nil
	}
	return true, nil
}

func mappingDirective(node *yaml.Node) (string, bool, error) {
	if node == nil || node.Kind != yaml.MappingNode || len(node.Content) != 2 {
		return "", false, nil
	}
	name := node.Content[0].Value
	if name != directiveIgnore && name != directiveExists && name != directiveNotExists {
		return "", false, nil
	}
	value := node.Content[1]
	if value.Kind != yaml.ScalarNode || value.Tag != "!!bool" || value.Value != "true" {
		return "", false, errorsForDirective(name, "must be true")
	}
	return name, true, nil
}

func errorsForDirective(name, message string) error {
	return fmt.Errorf("directive %s %s", name, message)
}

func mappingFields(node *yaml.Node, directives bool) map[string]*yaml.Node {
	fields := map[string]*yaml.Node{}
	for i := 0; i < len(node.Content); i += 2 {
		name := node.Content[i].Value
		if directives && name == directiveStrict {
			continue
		}
		if directives && name == "$$strict" {
			name = directiveStrict
		}
		fields[name] = node.Content[i+1]
	}
	return fields
}

func mappingScalar(node *yaml.Node, name string) (string, bool) {
	for i := 0; i < len(node.Content); i += 2 {
		if node.Content[i].Value == name && node.Content[i+1].Kind == yaml.ScalarNode {
			return node.Content[i+1].Value, true
		}
	}
	return "", false
}

func isNotExistsDirective(node *yaml.Node) bool {
	if node == nil {
		return false
	}
	if node.Kind == yaml.ScalarNode && node.Tag == yamlStringTag && node.Value == directiveNotExists {
		return true
	}
	directive, found, _ := mappingDirective(node)
	return found && directive == directiveNotExists
}

func documentRoot(node *yaml.Node) *yaml.Node {
	if node == nil {
		return nil
	}
	if node.Kind == yaml.DocumentNode && len(node.Content) > 0 {
		return node.Content[0]
	}
	return node
}

func dereference(node *yaml.Node) *yaml.Node {
	for node != nil && node.Kind == yaml.AliasNode {
		node = node.Alias
	}
	return node
}
