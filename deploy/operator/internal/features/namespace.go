/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package features

import (
	"encoding/json"
	"fmt"
	"strings"
)

// LeaseAnnotation stores the namespaced operator's effective feature gates.
const LeaseAnnotation = "nvidia.com/dynamo-operator-admission-feature-gates"

// Resolver returns the effective gates and compatibility warnings for a namespace.
type Resolver interface {
	ForNamespace(namespace string) (Gate, []string)
}

type namespaceResolver struct {
	base   Gates
	source func(namespace string) (string, bool)
}

// NewResolver creates a namespace-aware gate resolver.
func NewResolver(base Gates, source func(namespace string) (string, bool)) Resolver {
	return &namespaceResolver{base: base, source: source}
}

func (r *namespaceResolver) ForNamespace(namespace string) (Gate, []string) {
	snapshot, found := r.source(namespace)
	if !found {
		return r.base, nil
	}

	resolved, unknown, err := resolveSnapshot(r.base, snapshot)
	if err != nil {
		return r.base, []string{fmt.Sprintf("ignoring invalid admission feature gates published for namespace %q: %v", namespace, err)}
	}
	if unknown == "" {
		return resolved, nil
	}
	return resolved, []string{fmt.Sprintf("namespace operator requested feature gate unknown to the cluster-wide operator: %s", unknown)}
}

// resolveSnapshot overlays a JSON gate snapshot onto base. Missing fields inherit base,
// allowing an older namespaced operator to coexist with a newer global operator.
func resolveSnapshot(base Gates, snapshot string) (Gates, string, error) {
	if snapshot == "" {
		return base, "", nil
	}

	resolved := base
	if err := json.Unmarshal([]byte(snapshot), &resolved); err != nil {
		return base, "", fmt.Errorf("decoding admission feature gates: %w", err)
	}

	decoder := json.NewDecoder(strings.NewReader(snapshot))
	decoder.DisallowUnknownFields()
	strict := base
	if err := decoder.Decode(&strict); err != nil {
		const prefix = "json: unknown field \""
		if message := err.Error(); strings.HasPrefix(message, prefix) && strings.HasSuffix(message, "\"") {
			return resolved, strings.TrimSuffix(strings.TrimPrefix(message, prefix), "\""), nil
		}
		return base, "", fmt.Errorf("decoding admission feature gates: %w", err)
	}

	return resolved, "", nil
}
