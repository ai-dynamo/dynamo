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

// ContainerLayers retains the authoritative sources through transformations of
// one logical Pod lineage. After capture or registration, all environment changes
// must use the collector: container.Env is only a staged legacy view, and Apply
// replaces it from retained sources. Container names identify registrations;
// register replacement containers and use Clone for renamed or derived containers.
// Independent Pod copies may share a collector only while its sources are read-only.
// Its zero value is ready to register containers. A nil collector retains legacy
// mutations without composing their sources.
type ContainerLayers struct {
	containers map[string][]Layer
}

// Registered reports whether a container already has retained sources.
// A nil collector has no registrations.
func (c *ContainerLayers) Registered(name string) bool {
	if c == nil {
		return false
	}
	_, exists := c.containers[name]
	return exists
}

// Register replaces a container's sources and stages their raw concatenation.
// container must not be nil. Inputs and the physical environment are independent
// snapshots; precedence and reference expansion are deferred until Apply.
// A nil collector leaves legacy rendering unchanged.
func (c *ContainerLayers) Register(container *corev1.Container, layers ...Layer) {
	if c == nil {
		return
	}
	if c.containers == nil {
		c.containers = make(map[string][]Layer)
	}

	// Keep physical mutations from changing either authored values or their provenance.
	sources := make([]Layer, len(layers))
	var staged []corev1.EnvVar
	for i, layer := range layers {
		sources[i] = Layer{Source: layer.Source, Env: (&corev1.Container{Env: layer.Env}).DeepCopy().Env, positions: slices.Clone(layer.positions)}
		for _, value := range layer.Env {
			staged = append(staged, *value.DeepCopy())
		}
	}
	c.containers[container.Name] = sources
	container.Env = staged
}

// AppendDefaults extends the first registered source while retaining the legacy
// physical append order. container must be registered with a defaults layer first.
// A nil collector performs only the legacy append.
func (c *ContainerLayers) AppendDefaults(container *corev1.Container, values []corev1.EnvVar) {
	if c != nil {
		defaults := &c.containers[container.Name][0]
		if defaults.positions != nil {
			// New generated entries follow the retained original diagnostic positions.
			position := 0
			if len(defaults.positions) > 0 {
				position = defaults.positions[len(defaults.positions)-1] + 1
			}
			for i := range values {
				defaults.positions = append(defaults.positions, position+i)
			}
		}
		defaults.Env = append(defaults.Env, (&corev1.Container{Env: values}).DeepCopy().Env...)
	}
	container.Env = append(container.Env, values...)
}

// captureContainer retains the source before a physical mutation can change it.
// container must not be nil; Capture supplies precise Pod field paths when available.
func (c *ContainerLayers) captureContainer(container *corev1.Container, source string) {
	if c != nil && !c.Registered(container.Name) {
		c.Register(container, Layer{Source: source, Env: container.Env})
	}
}

// Capture retains authored environments for containers not registered by a
// caller. pod must not be nil; call before applying further bindings.
func (c *ContainerLayers) Capture(pod *corev1.PodSpec) {
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
func (c *ContainerLayers) Bind(container *corev1.Container, source string, values []corev1.EnvVar) {
	c.captureContainer(container, "container[name="+container.Name+"].env")
	if c != nil && len(values) > 0 {
		c.containers[container.Name] = append(c.containers[container.Name], Layer{Source: source, Env: (&corev1.Container{Env: values}).DeepCopy().Env})
	}
	container.Env = append(container.Env, values...)
}

// Prepend replaces matching names and prepends authoritative values.
// container must not be nil. A nil collector performs only the legacy mutation.
func (c *ContainerLayers) Prepend(container *corev1.Container, source string, values []corev1.EnvVar) {
	c.captureContainer(container, "container[name="+container.Name+"].env")
	if c != nil && len(values) > 0 {
		c.containers[container.Name] = append(c.containers[container.Name], Layer{Source: source, Env: (&corev1.Container{Env: values}).DeepCopy().Env})
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

// PrependPod replaces matching names and prepends values in every container.
// pod must not be nil. A nil collector retains the legacy physical mutation.
func (c *ContainerLayers) PrependPod(pod *corev1.PodSpec, source string, values []corev1.EnvVar) {
	c.Capture(pod)
	for _, group := range [][]corev1.Container{pod.Containers, pod.InitContainers} {
		for i := range group {
			c.Prepend(&group[i], source, values)
		}
	}
}

// Omit removes names from the physical environment and every retained source.
// container must not be nil. Source diagnostic indexes remain unchanged.
func (c *ContainerLayers) Omit(container *corev1.Container, names ...string) {
	c.captureContainer(container, "container[name="+container.Name+"].env")
	if c != nil {
		for i, layer := range c.containers[container.Name] {
			c.containers[container.Name][i] = layer.Without(names...)
		}
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
func (c *ContainerLayers) Clone(from, to *corev1.Container, omitted []string) {
	c.captureContainer(from, "container[name="+from.Name+"].env")
	if c != nil {
		layers := c.containers[from.Name]
		c.containers[to.Name] = nil
		for _, layer := range layers {
			c.containers[to.Name] = append(c.containers[to.Name], layer.Without(omitted...))
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
func (c *ContainerLayers) Apply(pod *corev1.PodSpec) error {
	if c == nil {
		return nil
	}
	c.Capture(pod)
	for _, group := range [][]corev1.Container{pod.Containers, pod.InitContainers} {
		for i := range group {
			container := &group[i]
			env, err := Compose(c.containers[container.Name]...)
			if err != nil {
				return fmt.Errorf("container %q: %w", container.Name, err)
			}
			container.Env = env
		}
	}
	return nil
}
