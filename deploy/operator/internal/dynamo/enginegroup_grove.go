/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package dynamo

import (
	"context"
	"fmt"

	config "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	api "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup"
	groveconstants "github.com/ai-dynamo/grove/operator/api/common/constants"
	grove "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

// addGroveEngineGroupTopologyEnvVars binds portable aliases for one elastic member clique per world.
// main must not be nil and must not declare the operator-owned aliases. Its launch command is unchanged.
func addGroveEngineGroupTopologyEnvVars(main *corev1.Container) {
	// Grove prepends its own variables; these aliases must precede user values that reference them.
	topology := []corev1.EnvVar{
		{Name: consts.DynamoRankEnvVar, Value: fmt.Sprintf("$(%s)", groveconstants.EnvVarPodIndex)},
		{Name: consts.DynamoLeaderAddressEnvVar, Value: fmt.Sprintf("$(%s)-0.$(%s)", groveconstants.EnvVarPodCliqueName, groveconstants.EnvVarHeadlessService)},
	}
	main.Env = append(topology, main.Env...)
}

// configureGroveEngineGroupContainer preserves allocation-specific probes and supplies portable topology.
// main is the shared renderer's main container; declaredMain is the validated user template. Neither may be nil.
func configureGroveEngineGroupContainer(main, declaredMain *corev1.Container) {
	// Generated inference probes cannot describe every waiting joiner; retain only caller-owned probes.
	main.LivenessProbe = declaredMain.LivenessProbe.DeepCopy()
	main.ReadinessProbe = declaredMain.ReadinessProbe.DeepCopy()
	main.StartupProbe = declaredMain.StartupProbe.DeepCopy()
	addGroveEngineGroupTopologyEnvVars(main)
}

// groveEngineGroupRenderer owns backend selection and launch layout, not PCS reconciliation.
type groveEngineGroupRenderer struct {
	operatorConfig   *config.OperatorConfiguration
	runtimeConfig    *controller_common.RuntimeConfig
	secretsRetriever SecretsRetriever
	reader           client.Reader
}

// Render selects one implemented backend and returns its complete world layout.
// dgd, component, component.EngineGroup and the renderer's configurations must not be nil.
// Nil restart state and annotations mean no pending restart. Inputs are not mutated.
func (r groveEngineGroupRenderer) Render(
	ctx context.Context,
	dgd *api.DynamoGraphDeployment,
	component *api.DynamoComponentDeploymentSharedSpec,
	queueName string,
	topologyDomains []api.TopologyDomain,
	restart *RestartState,
	restartAnnotations map[string]string,
) (*grove.PodCliqueTemplateSpec, grove.PodCliqueScalingGroupConfig, error) {
	// Unsupported engines must not fall back to ordinary, static workload rendering.
	backend, err := getBackendFrameworkFromComponent(component, dgd)
	if err != nil {
		return nil, grove.PodCliqueScalingGroupConfig{}, err
	}
	switch backend {
	case BackendFrameworkSGLang:
		return r.renderSGLang(ctx, dgd, component, queueName, topologyDomains, restart, restartAnnotations)
	default:
		return nil, grove.PodCliqueScalingGroupConfig{}, fmt.Errorf("Engine Group Grove rendering is not implemented for backend %q", backend)
	}
}

// renderWorld constructs the common single-clique world from validated backend geometry.
// dgd, component, component.EngineGroup and renderer dependencies must not be nil.
// Nil restart state and annotations mean no pending restart. Inputs are not mutated.
func (r groveEngineGroupRenderer) renderWorld(
	ctx context.Context,
	dgd *api.DynamoGraphDeployment,
	component *api.DynamoComponentDeploymentSharedSpec,
	geometry enginegroup.ResolvedProfileGeometry,
	initialReplicas int32,
	queueName string,
	topologyDomains []api.TopologyDomain,
	restart *RestartState,
	restartAnnotations map[string]string,
) (*grove.PodCliqueTemplateSpec, grove.PodCliqueScalingGroupConfig, error) {
	// Propagate metadata on a private component copy before standard Pod construction.
	component = component.DeepCopy()
	propagateDGDAnnotations(dgd.Annotations, component)
	propagateDGDSpecMetadata(dgd, component)
	podTemplate := ensurePodTemplate(component)
	podTemplate.Labels[consts.KubeLabelDynamoNamespace] = GetDynamoNamespace(dgd, component)
	discovery := controller_common.GetDiscoveryBackend(r.operatorConfig.Discovery.Backend, dgd.Annotations)
	if discovery != "" {
		podTemplate.Annotations[consts.KubeAnnotationDynamoDiscoveryBackend] = string(discovery)
	}

	// Keep backend launch configuration explicit while reusing infrastructure defaults.
	role := ServiceRole{Name: component.ComponentName, Role: RoleMain, Replicas: initialReplicas}
	clique, err := buildCliqueForRole(cliqueParams{
		ctx: ctx, reader: r.reader,
		r: role, component: component, dynamoDeployment: dgd,
		backendFramework: BackendFramework(geometry.Backend), secretsRetriever: r.secretsRetriever, numberOfNodes: 1,
		operatorConfig: r.operatorConfig, runtimeConfig: r.runtimeConfig, componentName: component.ComponentName,
		usesPCSG: true, discoveryBackend: discovery,
		discoveryContext: NewDiscoveryContext(r.operatorConfig.Discovery.Backend, dgd.Annotations),
		restartState:     restart, existingRestartAnnotations: restartAnnotations,
		validatedQueueName: queueName, groveClusterTopologyDomains: topologyDomains,
		containerGPUs: func() (int64, error) { return geometry.GPUsPerReplica, nil },
	})
	if err != nil {
		return nil, grove.PodCliqueScalingGroupConfig{}, err
	}
	configureGroveEngineGroupContainer(&clique.Spec.PodSpec.Containers[0], GetMainContainer(component))
	clique.Spec.MinAvailable = ptr.To(initialReplicas)
	clique.Labels[consts.KubeLabelDynamoEngineGroup] = EngineGroupNameForComponent(dgd.Name, component.ComponentName, 0)
	clique.Labels[consts.KubeLabelDynamoScaleRepresentative] = consts.KubeLabelDynamoScaleRepresentativeYes

	// PCSG replicas count worlds; the member clique counts initial allocations.
	world := buildGroveScalingGroupConfig(component.ComponentName, component, dgd.Spec.TopologyConstraint,
		[]ServiceRole{role}, []string{clique.Name}, nil, false)
	return clique, world, nil
}
