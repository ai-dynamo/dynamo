/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package semanticenv

import (
	"fmt"

	corev1 "k8s.io/api/core/v1"
)

// ContainerLayers retains the original sources through Pod transformations.
// A nil map disables composition and leaves legacy rendering unchanged.
type ContainerLayers map[string][]Layer

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
		for _, container := range group.containers {
			if _, exists := c[container.Name]; !exists {
				c[container.Name] = []Layer{{Source: "podTemplate.spec." + group.name + "[name=" + container.Name + "].env", Env: container.DeepCopy().Env}}
			}
		}
	}
}

// Bind appends an authoritative source for an already captured container.
func (c ContainerLayers) Bind(name, source string, values []corev1.EnvVar) {
	if c != nil && len(values) > 0 {
		c[name] = append(c[name], Layer{Source: source, Env: values})
	}
}

// BindPod applies the same binding to every regular and init container.
// pod must not be nil. Existing sources are captured before bindings are added.
func (c ContainerLayers) BindPod(pod *corev1.PodSpec, source string, values []corev1.EnvVar) {
	if c == nil {
		return
	}
	c.Capture(pod)
	for _, group := range [][]corev1.Container{pod.Containers, pod.InitContainers} {
		for _, container := range group {
			c.Bind(container.Name, source, values)
		}
	}
}

// Omit removes intentional absences from every source without renumbering it.
func (c ContainerLayers) Omit(name string, names ...string) {
	for i, layer := range c[name] {
		c[name][i] = layer.Without(names...)
	}
}

// Clone retains source identity when a transformation clones a container.
func (c ContainerLayers) Clone(from, to string, omitted []string) {
	if c == nil {
		return
	}
	for _, layer := range c[from] {
		c[to] = append(c[to], layer.Without(omitted...))
	}
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
