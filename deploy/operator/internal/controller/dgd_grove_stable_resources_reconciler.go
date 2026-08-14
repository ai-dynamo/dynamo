/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package controller

import (
	"context"
	"fmt"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	commoncontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	networkingv1beta1 "istio.io/client-go/pkg/apis/networking/v1beta1"
	corev1 "k8s.io/api/core/v1"
	networkingv1 "k8s.io/api/networking/v1"
	"k8s.io/client-go/tools/events"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

// groveStableResourcesReconciler owns the services and ingress resources that
// accompany the Grove workload but do not participate in Grove readiness.
type groveStableResourcesReconciler struct {
	dgdResourceSyncer
	config *configv1alpha1.OperatorConfiguration
}

func newGroveStableResourcesReconciler(
	kubeClient client.Client,
	recorder events.EventRecorder,
	config *configv1alpha1.OperatorConfiguration,
) *groveStableResourcesReconciler {
	return &groveStableResourcesReconciler{
		dgdResourceSyncer: newDGDResourceSyncer(kubeClient, recorder),
		config:            config,
	}
}

func (r *groveStableResourcesReconciler) Reconcile(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	renderDeployment *nvidiacomv1beta1.DynamoGraphDeployment,
) ([]Resource, error) {
	logger := log.FromContext(ctx)
	if err := dynamo.ReconcileModelServicesForComponents(
		ctx,
		r,
		dgd,
		dynamo.ComponentsByName(dgd),
		dgd.Namespace,
	); err != nil {
		logger.Error(err, "failed to reconcile model services")
		return nil, fmt.Errorf("failed to reconcile model services: %w", err)
	}

	resources := []Resource{}
	isK8sDiscoveryEnabled := commoncontroller.IsK8sDiscoveryEnabled(
		r.config.Discovery.Backend,
		dgd.Annotations,
	)
	for i := range renderDeployment.Spec.Components {
		component := &renderDeployment.Spec.Components[i]
		if isK8sDiscoveryEnabled || string(component.ComponentType) == commonconsts.ComponentTypeFrontend {
			serviceResource, err := r.reconcileComponentService(
				ctx,
				dgd,
				renderDeployment,
				component,
				isK8sDiscoveryEnabled,
			)
			if err != nil {
				return nil, err
			}
			if serviceResource != nil {
				resources = append(resources, serviceResource)
			}
		}

		// Give a single-pod elastic-EP leader a stable address for its followers to join.
		// Sync every component so one that stops qualifying has its Service deleted.
		epService, err := r.reconcileElasticEPLeaderService(
			ctx,
			dgd,
			renderDeployment,
			component,
			!isSinglePodElasticEPLeader(component),
		)
		if err != nil {
			return nil, err
		}
		if epService != nil {
			resources = append(resources, epService)
		}

		if string(component.ComponentType) != commonconsts.ComponentTypeFrontend {
			continue
		}
		ingressResources, err := r.reconcileFrontendIngress(
			ctx,
			dgd,
			component,
		)
		if err != nil {
			return nil, err
		}
		resources = append(resources, ingressResources...)
	}

	return resources, nil
}

func (r *groveStableResourcesReconciler) reconcileComponentService(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	renderDeployment *nvidiacomv1beta1.DynamoGraphDeployment,
	component *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec,
	isK8sDiscoveryEnabled bool,
) (Resource, error) {
	logger := log.FromContext(ctx)
	componentName := component.ComponentName
	service, err := dynamo.GenerateComponentService(dynamo.ComponentServiceParams{
		ServiceName:     dynamo.GetDCDResourceName(dgd, componentName, ""),
		Namespace:       dgd.Namespace,
		ComponentType:   string(component.ComponentType),
		DynamoNamespace: renderDeployment.GetDynamoNamespaceForComponent(component),
		ComponentName:   componentName,
		Labels:          dynamo.GetDGDComponentResourceLabels(renderDeployment, componentName, component),
		Annotations:     dynamo.GetDGDComponentResourceAnnotations(renderDeployment, componentName, component),
		IsK8sDiscovery:  isK8sDiscoveryEnabled,
	})
	if err != nil {
		logger.Error(err, "failed to generate the main component service")
		return nil, fmt.Errorf("failed to generate the main component service: %w", err)
	}

	_, syncedService, err := commoncontroller.SyncResource(
		ctx,
		r,
		dgd,
		func(context.Context) (*corev1.Service, bool, error) {
			return service, false, nil
		},
	)
	if err != nil {
		logger.Error(err, "failed to sync the main component service")
		return nil, fmt.Errorf("failed to sync the main component service: %w", err)
	}
	if syncedService == nil {
		return nil, nil
	}

	if syncedService.Annotations == nil {
		syncedService.Annotations = make(map[string]string)
	}
	desiredAnnotations := dynamo.GetDGDComponentResourceAnnotations(
		renderDeployment,
		componentName,
		component,
	)
	updateAnnotations := false
	for key, value := range desiredAnnotations {
		if current, ok := syncedService.Annotations[key]; !ok || current != value {
			syncedService.Annotations[key] = value
			updateAnnotations = true
		}
	}
	if updateAnnotations {
		if err := r.Update(ctx, syncedService); err != nil {
			logger.Error(err, "Failed to update main component service", "component", componentName)
			if r.GetRecorder() != nil {
				r.GetRecorder().Eventf(
					dgd,
					syncedService,
					corev1.EventTypeWarning,
					"UpdateService",
					"Update",
					"Failed to update Service %s: %s",
					componentName,
					err,
				)
			}
			return nil, fmt.Errorf("failed to update main component service %s: %w", componentName, err)
		}
	}

	resource, err := commoncontroller.NewResource(syncedService, func() (bool, string) {
		return true, ""
	})
	if err != nil {
		return nil, fmt.Errorf("failed to sync the main component service: %w", err)
	}
	return resource, nil
}

// isSinglePodElasticEPLeader reports whether the component is the single-pod elastic-EP
// leader the headless Service is meant to address.
//
// The Service selector matches every pod carrying the component labels, so it resolves
// to the Ray head alone only while the component renders as one pod. Two shapes break
// that:
//
//   - replicas > 1: every replica runs its own Ray head, so one DNS name would
//     round-robin across unrelated clusters.
//   - numberOfNodes > 1: worker pods share the component labels, so the Service would
//     publish them as heads. That topology reaches its leader through the Grove leader
//     hostname and does not need this Service.
func isSinglePodElasticEPLeader(component *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) bool {
	container := dynamo.GetMainContainer(component)
	if container == nil || !dynamo.IsElasticEPRayLaunch(container) {
		return false
	}
	if component.GetNumberOfNodes() > 1 {
		return false
	}
	return component.Replicas == nil || *component.Replicas == 1
}

// reconcileElasticEPLeaderService creates the headless Service a single-pod elastic-EP
// leader is reachable at, or deletes it when toDelete says the component no longer
// qualifies. See dynamo.GenerateElasticEPHeadlessService and isSinglePodElasticEPLeader.
func (r *groveStableResourcesReconciler) reconcileElasticEPLeaderService(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	renderDeployment *nvidiacomv1beta1.DynamoGraphDeployment,
	component *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec,
	toDelete bool,
) (Resource, error) {
	logger := log.FromContext(ctx)
	componentName := component.ComponentName
	service := dynamo.GenerateElasticEPHeadlessService(dynamo.ComponentServiceParams{
		ServiceName:     dynamo.GetDCDResourceName(dgd, componentName, ""),
		Namespace:       dgd.Namespace,
		ComponentType:   string(component.ComponentType),
		DynamoNamespace: renderDeployment.GetDynamoNamespaceForComponent(component),
		ComponentName:   componentName,
		Labels:          dynamo.GetDGDComponentResourceLabels(renderDeployment, componentName, component),
		Annotations:     dynamo.GetDGDComponentResourceAnnotations(renderDeployment, componentName, component),
	})

	_, syncedService, err := commoncontroller.SyncResource(
		ctx,
		r,
		dgd,
		func(context.Context) (*corev1.Service, bool, error) {
			return service, toDelete, nil
		},
	)
	if err != nil {
		logger.Error(err, "failed to sync the elastic-EP leader service")
		return nil, fmt.Errorf("failed to sync the elastic-EP leader service: %w", err)
	}
	if syncedService == nil {
		return nil, nil
	}

	resource, err := commoncontroller.NewResource(syncedService, func() (bool, string) {
		return true, ""
	})
	if err != nil {
		return nil, fmt.Errorf("failed to wrap the elastic-EP leader service: %w", err)
	}
	return resource, nil
}

func (r *groveStableResourcesReconciler) reconcileFrontendIngress(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	component *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec,
) ([]Resource, error) {
	logger := log.FromContext(ctx)
	componentName := component.ComponentName
	ingressSpec := dynamo.GenerateDefaultIngressSpec(dgd, r.config.Ingress)
	if preservedIngressSpec, ok := dynamo.GetDGDComponentPreservedIngressSpec(dgd, componentName); ok {
		ingressSpec = preservedIngressSpec
	}

	ingress := dynamo.GenerateComponentIngress(
		ctx,
		dynamo.GetDCDResourceName(dgd, componentName, ""),
		dgd.Namespace,
		ingressSpec,
	)
	_, syncedIngress, err := commoncontroller.SyncResource(
		ctx,
		r,
		dgd,
		func(context.Context) (*networkingv1.Ingress, bool, error) {
			if !ingressSpec.Enabled || ingressSpec.IngressControllerClassName == nil {
				logger.Info("Ingress is not enabled")
				return ingress, true, nil
			}
			return ingress, false, nil
		},
	)
	if err != nil {
		logger.Error(err, "failed to sync the main component ingress")
		return nil, fmt.Errorf("failed to sync the main component ingress: %w", err)
	}

	resources := []Resource{}
	if syncedIngress != nil {
		resource, err := commoncontroller.NewResource(syncedIngress, func() (bool, string) {
			return true, ""
		})
		if err != nil {
			return nil, fmt.Errorf("failed to create the main component ingress resource: %w", err)
		}
		resources = append(resources, resource)
	}

	if !r.config.Ingress.UseVirtualService() {
		return resources, nil
	}
	virtualService := dynamo.GenerateComponentVirtualService(
		ctx,
		dynamo.GetDCDResourceName(dgd, componentName, ""),
		dgd.Namespace,
		ingressSpec,
	)
	_, syncedVirtualService, err := commoncontroller.SyncResource(
		ctx,
		r,
		dgd,
		func(context.Context) (*networkingv1beta1.VirtualService, bool, error) {
			if !ingressSpec.IsVirtualServiceEnabled() {
				logger.Info("VirtualService is not enabled")
				return virtualService, true, nil
			}
			return virtualService, false, nil
		},
	)
	if err != nil {
		logger.Error(err, "failed to sync the main component virtual service")
		return nil, fmt.Errorf("failed to sync the main component virtual service: %w", err)
	}
	if syncedVirtualService != nil {
		resource, err := commoncontroller.NewResource(syncedVirtualService, func() (bool, string) {
			return true, ""
		})
		if err != nil {
			return nil, fmt.Errorf("failed to create the main component virtual service resource: %w", err)
		}
		resources = append(resources, resource)
	}
	return resources, nil
}
