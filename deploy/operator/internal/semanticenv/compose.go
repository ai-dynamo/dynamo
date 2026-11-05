/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Package semanticenv composes Kubernetes environments without expanding runtime
// values. List order is binding within a layer; dependencies order entries across
// layers. Callers supply layers from general to specific, retaining their sources
// until composition is complete.
package semanticenv

import (
	"container/heap"
	"fmt"
	"strings"

	corev1 "k8s.io/api/core/v1"
)

// Layer is an ordered Kubernetes env list. Source is its field path, used in
// diagnostics. Later layers override earlier definitions of the same name.
type Layer struct {
	Source    string
	Env       []corev1.EnvVar
	positions []int
}

// Without removes bindings intentionally omitted by a container transformation,
// retaining original source positions for diagnostics. It does not mutate l.
func (l Layer) Without(names ...string) Layer {
	removed := make(map[string]bool, len(names))
	for _, name := range names {
		removed[name] = true
	}
	result := Layer{Source: l.Source}
	for i, env := range l.Env {
		if !removed[env.Name] {
			result.Env = append(result.Env, env)
			position := i
			if l.positions != nil {
				position = l.positions[i]
			}
			result.positions = append(result.positions, position)
		}
	}
	return result
}

type entry struct {
	env      corev1.EnvVar
	layer    int
	position int
	next     []int
	indegree int
}

type readyEntries []int

func (r readyEntries) Len() int           { return len(r) }
func (r readyEntries) Less(i, j int) bool { return r[i] < r[j] }
func (r readyEntries) Swap(i, j int)      { r[i], r[j] = r[j], r[i] }
func (r *readyEntries) Push(x any)        { *r = append(*r, x.(int)) }
func (r *readyEntries) Pop() any {
	last := len(*r) - 1
	x := (*r)[last]
	*r = (*r)[:last]
	return x
}

// Compose selects cross-layer overrides and stably interleaves the surviving
// lists. It preserves Kubernetes' within-list forward-reference, escaping, and
// duplicate-name semantics. Missing references and self-references are runtime
// lookups, not dependency errors. Nil or empty layers produce a nil environment.
// Inputs are not mutated, and the returned environment owns its values.
func Compose(layers ...Layer) ([]corev1.EnvVar, error) {
	// Select the winning source without conflating repeated entries in one list.
	winners := make(map[string]int)
	for i, layer := range layers {
		for _, env := range layer.Env {
			winners[env.Name] = i
		}
	}

	// Preserve the order of surviving entries within each input list.
	var entries []entry
	definitions := make(map[string]int)
	for i, layer := range layers {
		previous := -1
		for j, env := range layer.Env {
			if winners[env.Name] != i {
				continue
			}
			current := len(entries)
			position := j
			if layer.positions != nil {
				position = layer.positions[j]
			}
			entries = append(entries, entry{env: env, layer: i, position: position})
			definitions[env.Name] = current
			if previous >= 0 {
				addEdge(entries, previous, current)
			}
			previous = current
		}
	}

	// Only cross-layer references add ordering beyond the authored list order.
	for i := range entries {
		for _, name := range references(entries[i].env.Value) {
			if source, exists := definitions[name]; exists && entries[source].layer != entries[i].layer {
				addEdge(entries, source, i)
			}
		}
	}

	// Pick the earliest available original entry, never its alphabetical name.
	ready := &readyEntries{}
	for i := range entries {
		if entries[i].indegree == 0 {
			heap.Push(ready, i)
		}
	}
	var result []corev1.EnvVar
	for ready.Len() > 0 {
		available := heap.Pop(ready).(int)
		result = append(result, *entries[available].env.DeepCopy())
		for _, next := range entries[available].next {
			entries[next].indegree--
			if entries[next].indegree == 0 {
				heap.Push(ready, next)
			}
		}
	}
	if len(result) < len(entries) {
		return nil, cycleError(entries, layers)
	}
	return result, nil
}

func addEdge(entries []entry, from, to int) {
	// Every occurrence is balanced when the source releases its outgoing edges.
	entries[from].next = append(entries[from].next, to)
	entries[to].indegree++
}

func references(value string) []string {
	// Kubernetes consumes escaped dollar pairs before recognizing $(NAME).
	var names []string
	for i := 0; i+1 < len(value); i++ {
		if value[i] != '$' {
			continue
		}
		switch value[i+1] {
		case '$':
			i++
		case '(':
			if end := strings.IndexByte(value[i+2:], ')'); end >= 0 {
				names = append(names, value[i+2:i+2+end])
				i += end + 2
			}
		}
	}
	return names
}

func cycleError(entries []entry, layers []Layer) error {
	// Find an actual cycle, rather than reporting downstream blocked entries.
	state := make([]uint8, len(entries))
	var path, cycle []int
	var visit func(int) bool
	visit = func(i int) bool {
		if state[i] == 1 {
			for start, node := range path {
				if node == i {
					cycle = append(append([]int(nil), path[start:]...), i)
					return true
				}
			}
		}
		if state[i] == 2 {
			return false
		}
		state[i] = 1
		path = append(path, i)
		for _, next := range entries[i].next {
			if visit(next) {
				return true
			}
		}
		path = path[:len(path)-1]
		state[i] = 2
		return false
	}
	for i := range entries {
		if visit(i) {
			break
		}
	}

	// Include source positions so a conflict can be fixed in its authored list.
	locations := make([]string, len(cycle))
	for i, node := range cycle {
		e := entries[node]
		locations[i] = fmt.Sprintf("%s (%s[%d])", e.env.Name, layers[e.layer].Source, e.position)
	}
	return fmt.Errorf("environment dependency cycle: %s", strings.Join(locations, " -> "))
}
