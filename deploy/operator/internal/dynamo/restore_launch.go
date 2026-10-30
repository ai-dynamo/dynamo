/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package dynamo

import (
	"context"
	"fmt"
	"sync"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
)

// RenderedLaunch is the command and args a rendered role gives the main container.
type RenderedLaunch struct {
	Role    Role
	Command []string
	Args    []string
}

// RenderMainContainerLaunches returns the launch the backend renders for the main
// container of every role that serves the component, by running the same backend
// rewrite as Pod rendering on a copy of the authored container. A single-node
// component renders RoleMain; a multinode component renders RoleLeader and RoleWorker.
// Complete role PodTemplates own their launch, so each role is rendered from its own
// template. dgdAnnotations are the graph annotations that rendering propagates to
// components. component must not be nil; it is not modified. An error reports that the
// launch cannot be derived without the cluster, for example the multinode inter-pod GMS
// layout, which renders rank-aware roles.
func RenderMainContainerLaunches(
	ctx context.Context,
	component *v1beta1.DynamoComponentDeploymentSharedSpec,
	explicitBackendFramework string,
	dgdAnnotations map[string]string,
	multinodeDeploymentType commonconsts.MultinodeDeploymentType,
) ([]RenderedLaunch, error) {
	// A multinode component is served by a leader and its workers.
	numberOfNodes := component.GetNumberOfNodes()
	roles := []Role{RoleMain}
	if numberOfNodes > 1 {
		if component.IsInterPodGMSEnabled() {
			return nil, fmt.Errorf("multinode inter-pod GMS renders rank-aware roles")
		}
		roles = []Role{RoleLeader, RoleWorker}
	}

	backendFramework, err := determineBackendFrameworkForComponent(component, explicitBackendFramework)
	if err != nil {
		return nil, err
	}

	launches := make([]RenderedLaunch, 0, len(roles))
	for _, role := range roles {
		launch, err := renderMainContainerLaunch(ctx, component, backendFramework, dgdAnnotations, multinodeDeploymentType, role, numberOfNodes)
		if err != nil {
			return nil, fmt.Errorf("failed to render the %s launch: %w", role, err)
		}
		launches = append(launches, launch)
	}
	return launches, nil
}

// renderMainContainerLaunch applies the backend rewrite for one role to a copy of the
// authored main container. component must not be nil; it is not modified.
func renderMainContainerLaunch(
	ctx context.Context,
	component *v1beta1.DynamoComponentDeploymentSharedSpec,
	backendFramework BackendFramework,
	dgdAnnotations map[string]string,
	multinodeDeploymentType commonconsts.MultinodeDeploymentType,
	role Role,
	numberOfNodes int32,
) (RenderedLaunch, error) {
	// Complete role sources own the launch of their role, as in GenerateBasePodSpec.
	ownership := roleLaunchOwnedByOperator
	effective := component.DeepCopy()
	if HasRolePodTemplates(component) {
		ownership = roleLaunchOwnedByPodTemplate
		var err error
		effective, err = EffectiveComponentForRole(component, role)
		if err != nil {
			return RenderedLaunch{}, err
		}
	}
	propagateDGDAnnotations(dgdAnnotations, effective)

	main := GetMainContainer(effective)
	if main == nil {
		return RenderedLaunch{}, fmt.Errorf("component has no %s container", commonconsts.MainContainerName)
	}
	container := main.DeepCopy()

	// An authored command replaces the operator's /bin/sh -c default, so a missing one keeps the shell.
	if len(container.Command) == 0 {
		container.Command = []string{"/bin/sh", "-c"}
	}

	deployer := MultinodeDeployerFactory(multinodeDeploymentType)
	if deployer == nil {
		return RenderedLaunch{}, fmt.Errorf("unsupported multinode deployment type: %s", multinodeDeploymentType)
	}
	backend := BackendFactory(backendFramework, &configv1alpha1.OperatorConfiguration{}, "", ownership)
	if backend == nil {
		return RenderedLaunch{}, fmt.Errorf("unsupported backend framework: %s", backendFramework)
	}
	containerGPUs := sync.OnceValues(func() (int64, error) {
		return ResolveContainerGPUs(ctx, nil, "", effective)
	})
	if err := backend.UpdateContainer(container, numberOfNodes, role, effective, effective.ComponentName, deployer, containerGPUs); err != nil {
		return RenderedLaunch{}, err
	}
	return RenderedLaunch{Role: role, Command: container.Command, Args: container.Args}, nil
}
