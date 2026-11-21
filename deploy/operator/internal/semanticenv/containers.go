/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package semanticenv

import (
	"fmt"
	"slices"

	corev1 "k8s.io/api/core/v1"
)

// ContainerLayers retains the original sources through Pod transformations.
// A nil map disables composition while mutation methods retain legacy behavior.
type ContainerLayers map[string][]Layer

// captureContainer retains the source before a physical mutation can change it.
// container must not be nil; Capture supplies precise Pod field paths when available.
func (c ContainerLayers) captureContainer(container *corev1.Container, source string) {
	if c != nil {
		if _, exists := c[container.Name]; !exists {
			c[container.Name] = []Layer{{Source: source, Env: container.DeepCopy().Env}}
		}
	}
}

// Capture retains authored environments for containers not registered by a
// caller. pod must not be nil; call before applying further bindings.
func (c ContainerLayers) Capture(pod *corev1.PodSpec) {
	if c == nil {
		return
	}
	for _, group := range []struct {
		name       string
		containers []corev1.Container
	}{{"containers", pod.Containers}, {"initContainers", pod.InitContainers}} {
		for i := range group.containers {
			container := &group.containers[i]
			c.captureContainer(container, "podTemplate.spec."+group.name+"[name="+container.Name+"].env")
		}
	}
}

// Bind appends authoritative values and records their source before finalization.
// container must not be nil. A nil collector performs only the legacy append.
func (c ContainerLayers) Bind(container *corev1.Container, source string, values []corev1.EnvVar) {
	c.captureContainer(container, "container[name="+container.Name+"].env")
	if c != nil && len(values) > 0 {
		c[container.Name] = append(c[container.Name], Layer{Source: source, Env: (&corev1.Container{Env: values}).DeepCopy().Env})
	}
	container.Env = append(container.Env, values...)
}

// Prepend replaces matching names and prepends authoritative values.
// container must not be nil. A nil collector performs only the legacy mutation.
func (c ContainerLayers) Prepend(container *corev1.Container, source string, values []corev1.EnvVar) {
	c.captureContainer(container, "container[name="+container.Name+"].env")
	if c != nil && len(values) > 0 {
		c[container.Name] = append(c[container.Name], Layer{Source: source, Env: (&corev1.Container{Env: values}).DeepCopy().Env})
	}

	// The physical view follows legacy replacement; the collector retains provenance.
	names := make([]string, len(values))
	for i, value := range values {
		names[i] = value.Name
	}
	result := slices.Clone(values)
	for _, value := range container.Env {
		if !slices.Contains(names, value.Name) {
			result = append(result, value)
		}
	}
	container.Env = result
}

// BindPod applies the same binding to every regular and init container.
// pod must not be nil. Existing sources are captured before bindings are added.
func (c ContainerLayers) BindPod(pod *corev1.PodSpec, source string, values []corev1.EnvVar) {
	c.Capture(pod)
	for _, group := range [][]corev1.Container{pod.Containers, pod.InitContainers} {
		for i := range group {
			c.Bind(&group[i], source, values)
		}
	}
}

// PrependPod replaces matching names and prepends values in every container.
// pod must not be nil. A nil collector retains the legacy physical mutation.
func (c ContainerLayers) PrependPod(pod *corev1.PodSpec, source string, values []corev1.EnvVar) {
	c.Capture(pod)
	for _, group := range [][]corev1.Container{pod.Containers, pod.InitContainers} {
		for i := range group {
			c.Prepend(&group[i], source, values)
		}
	}
}

// Omit removes names from the physical environment and every retained source.
// container must not be nil. Source diagnostic indexes remain unchanged.
func (c ContainerLayers) Omit(container *corev1.Container, names ...string) {
	c.captureContainer(container, "container[name="+container.Name+"].env")
	for i, layer := range c[container.Name] {
		c[container.Name][i] = layer.Without(names...)
	}

	// Filter in place only after retaining an independent copy of authored values.
	filtered := container.Env[:0]
	for _, value := range container.Env {
		if !slices.Contains(names, value.Name) {
			filtered = append(filtered, value)
		}
	}
	container.Env = filtered
}

// Clone copies the filtered physical environment and retains its source identity.
// from and to must not be nil; other container fields are not changed.
func (c ContainerLayers) Clone(from, to *corev1.Container, omitted []string) {
	c.captureContainer(from, "container[name="+from.Name+"].env")
	if c != nil {
		layers := c[from.Name]
		c[to.Name] = nil
		for _, layer := range layers {
			c[to.Name] = append(c[to.Name], layer.Without(omitted...))
		}
	}

	// Cloned engines do not share mutable physical Env entries with their source.
	var filtered []corev1.EnvVar
	for _, value := range from.Env {
		if !slices.Contains(omitted, value.Name) {
			filtered = append(filtered, *value.DeepCopy())
		}
	}
	to.Env = filtered
}

// Apply finalizes the remaining containers after all bindings and omissions.
// pod must not be nil. Sources are retained so independently cloned Pods can
// share read-only layers; returned environments do not alias those layers.
func (c ContainerLayers) Apply(pod *corev1.PodSpec) error {
	if c == nil {
		return nil
	}
	c.Capture(pod)
	for _, group := range [][]corev1.Container{pod.Containers, pod.InitContainers} {
		for i := range group {
			container := &group[i]
			env, err := Compose(c[container.Name]...)
			if err != nil {
				return fmt.Errorf("container %q: %w", container.Name, err)
			}
			container.Env = env
		}
	}
	return nil
}
