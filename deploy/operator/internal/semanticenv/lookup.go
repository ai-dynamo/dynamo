// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package semanticenv

// Lookup resolves one binding from its effective, transitive input definitions.
// It does not validate unrelated bindings: later feature-owned definitions may
// replace them before final Compose. The bool reports whether name is declared;
// missing runtime inputs or a cycle within the selected inputs remain unknown.
// Nil layers mean no declarations. Inputs are not mutated.
func Lookup(name string, layers ...Layer) (Value, bool) {
	// Select source layers before following references; discarded defaults cannot reappear.
	definitions, last := effectiveDefinitions(layers)
	target, exists := last[name]
	if !exists {
		return Value{}, false
	}

	// Same-list references read the preceding occurrence, not a later duplicate.
	dependencies := make([][]int, len(definitions))
	previous := make(map[string]int)
	for i, current := range definitions {
		if current.env.ValueFrom == nil {
			for _, reference := range References(current.env.Value) {
				source, found := last[reference]
				if found && definitions[source].layer == current.layer {
					source, found = previous[reference]
				}
				if found {
					dependencies[i] = append(dependencies[i], source)
				}
			}
		}
		previous[current.env.Name] = i
	}

	// Retain only definitions that can affect the selected final occurrence.
	needed := make([]bool, len(definitions))
	pending := []int{target}
	for len(pending) != 0 {
		index := pending[len(pending)-1]
		pending = pending[:len(pending)-1]
		if !needed[index] {
			needed[index] = true
			pending = append(pending, dependencies[index]...)
		}
	}
	selected := make([]Layer, len(layers))
	for i, layer := range layers {
		selected[i].Source = layer.Source
	}
	for i, current := range definitions {
		if needed[i] {
			layer := &selected[current.layer]
			layer.Env = append(layer.Env, current.env)
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
