/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package dynamo

import (
	"fmt"
	"slices"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features/compatibility"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/semanticenv"
	corev1 "k8s.io/api/core/v1"
)

// podEnvironment retains source layers through backend, GMS and failover
// transformations. A nil collector selects the unchanged legacy render path.
type podEnvironment struct {
	general    []corev1.EnvVar
	containers map[string][]semanticenv.Layer
	overrides  []map[string]semanticenv.Layer
}

func newPodEnvironment(annotations map[string]string, general []corev1.EnvVar) *podEnvironment {
	if !compatibility.OrderedEnvironmentVariables.Enabled(annotations) {
		return nil
	}
	return &podEnvironment{general: general, containers: make(map[string][]semanticenv.Layer)}
}

func (p *podEnvironment) register(name, source string, defaults, general, authored []corev1.EnvVar) {
	if p == nil {
		return
	}
	p.containers[name] = []semanticenv.Layer{
		{Source: "operator.defaults." + name, Env: defaults},
		{Source: "spec.env", Env: general},
		{Source: source, Env: authored},
	}
	for _, overrides := range p.overrides {
		if layer, exists := overrides[name]; exists {
			p.containers[name] = append(p.containers[name], layer)
		}
	}
}

func (p *podEnvironment) appendDefaults(name string, values []corev1.EnvVar) {
	if p != nil {
		p.containers[name][0].Env = slices.Concat(p.containers[name][0].Env, values)
	}
}

// staged exposes raw, last-definition-wins inputs to backend readers without
// prematurely sorting or expanding them. p must not be nil.
func (p *podEnvironment) staged(name string) []corev1.EnvVar {
	var result []corev1.EnvVar
	for _, layer := range p.containers[name] {
		result = append(result, layer.Env...)
	}
	return result
}

func (p *podEnvironment) bind(name, source string, values []corev1.EnvVar) {
	if p != nil && len(values) > 0 {
		p.containers[name] = append(p.containers[name], semanticenv.Layer{Source: source, Env: values})
	}
}

func (p *podEnvironment) clone(from, to string, omitted []string) {
	if p == nil {
		return
	}
	for _, layer := range p.containers[from] {
		p.containers[to] = append(p.containers[to], layer.Without(omitted...))
	}
}

func (p *podEnvironment) omit(name string, omitted ...string) {
	if p == nil {
		return
	}
	for i, layer := range p.containers[name] {
		p.containers[name][i] = layer.Without(omitted...)
	}
}

// apply finalizes all registered containers. pod must not be nil. A nil
// collector leaves the complete legacy rendering unchanged.
func (p *podEnvironment) apply(pod *corev1.PodSpec) error {
	if p == nil {
		return nil
	}
	for _, containers := range [][]corev1.Container{pod.Containers, pod.InitContainers} {
		for i := range containers {
			container := &containers[i]
			layers, registered := p.containers[container.Name]
			if !registered {
				p.register(container.Name, "podTemplate.containers[name="+container.Name+"].env", nil, nil, container.Env)
				layers = p.containers[container.Name]
			}
			env, err := semanticenv.Compose(layers...)
			if err != nil {
				return fmt.Errorf("container %q: %w", container.Name, err)
			}
			container.Env = env
		}
	}
	return nil
}
