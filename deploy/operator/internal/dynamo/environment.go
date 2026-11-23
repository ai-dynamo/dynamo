/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package dynamo

import (
	"fmt"
	"slices"
	"strings"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features/compatibility"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/semanticenv"
	corev1 "k8s.io/api/core/v1"
)

// podEnvironment retains source layers through backend, GMS and failover
// transformations. A nil collector selects the unchanged legacy render path.
type podEnvironment struct {
	general        []corev1.EnvVar
	containers     *semanticenv.ContainerLayers
	overrides      map[string]semanticenv.Layer
	policyRuntime  string
	policyBindings []corev1.EnvVar
}

// FinalizePodEnvironment composes retained sources and applies backend work that
// depends on the final environment. pod must not be nil. Nil layers retain the
// legacy path, whose backend processing already ran during rendering.
// engineNames selects the combined-runtime engines; nil selects none.
// containerGPUs is resolved lazily for SGLang exporter ports and may be nil when
// no SGLang engine is selected.
func FinalizePodEnvironment(pod *corev1.PodSpec, layers *semanticenv.ContainerLayers, backend BackendFramework, engineNames []string, containerGPUs ContainerGPUCount) error {
	if layers == nil {
		return nil
	}
	if err := layers.Apply(pod); err != nil {
		return err
	}
	if backend == BackendFrameworkSGLang {
		return finalizeSGLangExporterPorts(pod, engineNames, containerGPUs)
	}
	return nil
}

// DCDUsesOrderedEnvironment follows the origin selected by the workload renderer.
// dcd must not be nil. Any semantic role consumes the shared root env list.
func DCDUsesOrderedEnvironment(dcd *v1beta1.DynamoComponentDeployment) bool {
	if dcd.GetGraphDeploymentController() == nil {
		return compatibility.OrderedEnvironmentVariables.Enabled(dcd.Annotations)
	}
	for _, template := range ComponentPodTemplates(&dcd.Spec.DynamoComponentDeploymentSharedSpec) {
		if compatibility.OrderedEnvironmentVariables.Enabled(template.Annotations) {
			return true
		}
	}
	return false
}

func newPodEnvironment(annotations map[string]string, general []corev1.EnvVar) *podEnvironment {
	if !compatibility.OrderedEnvironmentVariables.Enabled(annotations) {
		return nil
	}
	return &podEnvironment{general: general, containers: &semanticenv.ContainerLayers{}}
}

// ResolveEnvironmentValue reads a staged general/container binding without
// assuming unavailable operator or runtime values. The bool reports whether a
// binding is declared, independently of whether its value is known. Inputs may
// be nil and are never mutated; callers select this lookup only for new origins.
func ResolveEnvironmentValue(general, authored []corev1.EnvVar, name string) (semanticenv.Value, bool) {
	return semanticenv.Lookup(name,
		semanticenv.Layer{Source: "spec.env", Env: general},
		semanticenv.Layer{Source: "podTemplate.env", Env: authored},
	)
}

// rewriteFrontendGeneralEnvironment rewrites an active graph-level config in
// its original source. component must not be nil; general may be nil. Neither
// the component nor its input environment is mutated.
func rewriteFrontendGeneralEnvironment(component *v1beta1.DynamoComponentDeploymentSharedSpec, componentName string, general []corev1.EnvVar) ([]corev1.EnvVar, error) {
	if component.ComponentType != v1beta1.ComponentTypeFrontend {
		return general, nil
	}

	// A fully shadowed graph config is not a frontend input, including invalid values.
	templates := ComponentPodTemplates(component)
	if len(templates) == 0 {
		updated, _, err := rewriteFrontendDeploymentConfig(general, nil, componentName)
		return updated, err
	}
	for _, template := range templates {
		main := GetMainContainer(&v1beta1.DynamoComponentDeploymentSharedSpec{PodTemplate: template})
		var authored []corev1.EnvVar
		if main != nil {
			authored = main.Env
		}
		if findEnvVar(authored, commonconsts.DynamoDeploymentConfigEnvVar) == nil {
			updated, _, err := rewriteFrontendDeploymentConfig(general, authored, componentName)
			return updated, err
		}
	}
	return general, nil
}

func rewriteFrontendDeploymentConfig(general, authored []corev1.EnvVar, componentName string) ([]corev1.EnvVar, []corev1.EnvVar, error) {
	// Unknown inputs remain the user's runtime binding, never an empty JSON value.
	value, exists := ResolveEnvironmentValue(general, authored, commonconsts.DynamoDeploymentConfigEnvVar)
	if !exists || !value.Known {
		return general, authored, nil
	}
	config, err := updateDynDeploymentConfigBytes([]byte(value.Value), componentName, commonconsts.DynamoServicePort)
	if err != nil {
		return nil, nil, fmt.Errorf("rewrite frontend deployment config: %w", err)
	}

	// Keep the winner in its source, escaping the resolved JSON against a second expansion.
	binding := corev1.EnvVar{Name: commonconsts.DynamoDeploymentConfigEnvVar, Value: strings.ReplaceAll(string(config), "$", "$$")}
	if findEnvVar(authored, commonconsts.DynamoDeploymentConfigEnvVar) != nil {
		authored = slices.Clone(authored)
		*findEnvVar(authored, commonconsts.DynamoDeploymentConfigEnvVar) = binding
	} else {
		general = slices.Clone(general)
		*findEnvVar(general, commonconsts.DynamoDeploymentConfigEnvVar) = binding
	}
	return general, authored, nil
}

func (p *podEnvironment) register(container *corev1.Container, source string, defaults, general, authored []corev1.EnvVar) {
	if p == nil {
		return
	}
	layers := []semanticenv.Layer{
		{Source: "operator.defaults." + container.Name, Env: defaults},
		{Source: "spec.env", Env: general},
		{Source: source, Env: authored},
	}

	// Only the compiled policy's runtime omits graph-level policy inputs.
	if container.Name == p.policyRuntime {
		layers[1] = layers[1].Without(workerKvTransferPolicyEnvNames...)
		original := layers[2].Without(workerKvTransferPolicyEnvNames...)
		layers = append(layers[:2],
			semanticenv.Layer{Source: "operator.kvTransferPolicy", Env: p.policyBindings}, original)
	}
	if layer, exists := p.overrides[container.Name]; exists {
		layers = append(layers, layer)
	}
	p.containers.Register(container, layers...)
}

// DCDHasKVTransferPolicy recognizes a graph-compiled policy from its existing
// topology-source snapshot, not from standalone DCD topology annotations.
// dcd must not be nil.
func DCDHasKVTransferPolicy(dcd *v1beta1.DynamoComponentDeployment) bool {
	return dcd.GetGraphDeploymentController() != nil &&
		(dcd.Annotations[commonconsts.KubeAnnotationTopologyLabelKey] != "" ||
			dcd.Annotations[commonconsts.KubeAnnotationTopologyClusterTopologyName] != "")
}

// ownKVTransferPolicy scopes graph-default removal to the container publishing
// the compiled policy. Compute-only roles have no such runtime container.
func (p *podEnvironment) ownKVTransferPolicy(component *v1beta1.DynamoComponentDeploymentSharedSpec, nativeSidecar bool, bindings []corev1.EnvVar) {
	if p == nil || len(bindings) == 0 {
		return
	}
	if GetDynamoSidecar(component) != nil {
		p.policyRuntime = commonconsts.RuntimeContainerName
	} else if !nativeSidecar {
		p.policyRuntime = commonconsts.MainContainerName
	} else {
		return
	}
	p.policyBindings = bindings
}

// DCDKVTransferPolicyEnvironment reads the policy snapshot stored in a child
// DCD's selected runtime. Standalone DCDs do not own graph policy bindings.
// dcd and component must not be nil.
func DCDKVTransferPolicyEnvironment(dcd *v1beta1.DynamoComponentDeployment, component *v1beta1.DynamoComponentDeploymentSharedSpec) []corev1.EnvVar {
	if !DCDHasKVTransferPolicy(dcd) {
		return nil
	}
	runtime := GetDynamoSidecar(component)
	if runtime == nil {
		runtime = GetMainContainer(component)
	}
	if runtime == nil {
		return nil
	}
	var bindings []corev1.EnvVar
	for _, value := range runtime.Env {
		if slices.Contains(workerKvTransferPolicyEnvNames, value.Name) {
			bindings = append(bindings, value)
		}
	}
	return bindings
}

func (p *podEnvironment) appendDefaults(container *corev1.Container, values []corev1.EnvVar) {
	var layers *semanticenv.ContainerLayers
	if p != nil {
		layers = p.containers
	}
	layers.AppendDefaults(container, values)
}

func (p *podEnvironment) bind(container *corev1.Container, source string, values []corev1.EnvVar) {
	var layers *semanticenv.ContainerLayers
	if p != nil {
		layers = p.containers
	}
	layers.Bind(container, source, values)
}

func (p *podEnvironment) clone(from, to *corev1.Container, omitted []string) {
	var layers *semanticenv.ContainerLayers
	if p != nil {
		layers = p.containers
	}
	layers.Clone(from, to, omitted)
}

func (p *podEnvironment) omit(container *corev1.Container, omitted ...string) {
	var layers *semanticenv.ContainerLayers
	if p != nil {
		layers = p.containers
	}
	layers.Omit(container, omitted...)
}

// capture registers remaining authored containers before provider mutations.
// pod must not be nil. A nil collector selects legacy rendering.
func (p *podEnvironment) capture(pod *corev1.PodSpec) *semanticenv.ContainerLayers {
	if p == nil {
		return nil
	}
	for _, group := range []struct {
		name       string
		containers []corev1.Container
	}{{"containers", pod.Containers}, {"initContainers", pod.InitContainers}} {
		for i := range group.containers {
			container := &group.containers[i]
			if !p.containers.Registered(container.Name) {
				p.register(container, "podTemplate.spec."+group.name+"[name="+container.Name+"].env", nil, nil, container.Env)
			}
		}
	}
	return p.containers
}

// combinedRuntimeEngineContainerNames identifies containers created by the
// component's runtime transformation, never by position in a selected capture Pod.
// component must not be nil. Native engines and non-workers receive no policy.
func combinedRuntimeEngineContainerNames(component *v1beta1.DynamoComponentDeploymentSharedSpec, nativeSidecar bool) []string {
	if nativeSidecar || !IsWorkerComponent(string(component.ComponentType)) {
		return nil
	}
	if IsIntraPodFailoverEnabled(component) {
		return IntraPodFailoverEngineContainerNames()
	}
	return []string{commonconsts.MainContainerName}
}
