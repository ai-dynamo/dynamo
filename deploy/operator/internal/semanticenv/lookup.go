// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package semanticenv

import corev1 "k8s.io/api/core/v1"

// Lookup resolves one binding from its effective, transitive input definitions.
// It does not validate unrelated bindings: later feature-owned definitions may
// replace them before final Compose. The bool reports whether name is declared;
// missing runtime inputs or a cycle within the selected inputs remain unknown.
// Nil layers mean no declarations. Inputs are not mutated.
func Lookup(name string, layers ...Layer) (Value, bool) {
	// Select source layers before following references; discarded defaults cannot reappear.
	winners := make(map[string]int)
	for i, layer := range layers {
		for _, variable := range layer.Env {
			winners[variable.Name] = i
		}
	}
	type definition struct {
		variable     corev1.EnvVar
		layer        int
		position     int
		dependencies []int
	}
	var definitions []definition
	last := make(map[string]int)
	for i, layer := range layers {
		for j, variable := range layer.Env {
			if winners[variable.Name] != i {
				continue
			}
			position := j
			if layer.positions != nil {
				position = layer.positions[j]
			}
			last[variable.Name] = len(definitions)
			definitions = append(definitions, definition{variable: variable, layer: i, position: position})
		}
	}
	target, exists := last[name]
	if !exists {
		return Value{}, false
	}

	// Same-list references read the preceding occurrence, not a later duplicate.
	previous := make(map[string]int)
	for i, current := range definitions {
		if current.variable.ValueFrom == nil {
			for _, reference := range References(current.variable.Value) {
				source, found := last[reference]
				if found && definitions[source].layer == current.layer {
					source, found = previous[reference]
				}
				if found {
					definitions[i].dependencies = append(definitions[i].dependencies, source)
				}
			}
		}
		previous[current.variable.Name] = i
	}

	// Retain only definitions that can affect the selected final occurrence.
	needed := make([]bool, len(definitions))
	pending := []int{target}
	for len(pending) != 0 {
		index := pending[len(pending)-1]
		pending = pending[:len(pending)-1]
		if !needed[index] {
			needed[index] = true
			pending = append(pending, definitions[index].dependencies...)
		}
	}
	selected := make([]Layer, len(layers))
	for i, layer := range layers {
		selected[i].Source = layer.Source
	}
	for i, current := range definitions {
		if needed[i] {
			layer := &selected[current.layer]
			layer.Env = append(layer.Env, current.variable)
			layer.positions = append(layer.positions, current.position)
		}
	}

	// The normal composer still enforces relevant cross-layer and local order constraints.
	environment, err := Compose(selected...)
	if err != nil {
		return Value{}, true
	}
	return Replay(environment)[name], true
}
