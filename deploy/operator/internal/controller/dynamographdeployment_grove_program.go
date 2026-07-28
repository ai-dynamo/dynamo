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

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	commoncontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	networkingv1beta1 "istio.io/client-go/pkg/apis/networking/v1beta1"
	corev1 "k8s.io/api/core/v1"
	networkingv1 "k8s.io/api/networking/v1"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

type groveProgram struct {
	// The DGD reconciler supplies provider-neutral resource synchronization and
	// shared stable-resource operations. Grove-specific rendering, scaling, and
	// status interpretation are explicit collaborators owned by this program.
	reconciler     *DynamoGraphDeploymentReconciler
	renderer       *groveWorkloadRenderer
	scaler         *groveScaler
	statusResolver *groveStatusResolver
}

func newGroveProgram(reconciler *DynamoGraphDeploymentReconciler) *groveProgram {
	return &groveProgram{
		reconciler: reconciler,
		renderer: newGroveWorkloadRenderer(
			reconciler.Client,
			reconciler.Config,
			reconciler.RuntimeConfig,
			reconciler.DockerSecretRetriever,
		),
		scaler:         newGroveScaler(reconciler.ScaleClient),
		statusResolver: newGroveStatusResolver(reconciler.Client),
	}
}

// Reconcile composes the complete Grove pathway. Each earlier operation
// returns a typed value consumed by later operations. Non-status DGD changes
// are persisted through req.DGD; status accumulates in the returned result.
func (p *groveProgram) Reconcile(
	ctx context.Context,
	req workloadProgramRequest,
) (programResult workloadProgramResult, retErr error) {
	programResult = newWorkloadProgramResult(req.DGD)
	defer func() {
		if retErr != nil {
			reason := reasonFailedToReconcileResources
			if classified, ok := workloadProgramFailureReason(retErr); ok {
				reason = classified
			}
			programResult.Fail(req.DGD.Generation, reason, retErr)
		}
		p.statusResolver.projectTopologyCondition(ctx, req.DGD, &programResult)
	}()
	log.FromContext(ctx).Info(
		"Reconciling Grove resources",
		"hasMultinode", req.DGD.HasAnyMultinodeComponent(),
	)

	if err := p.reconciler.migrateCurrentWorkerHashIfNeeded(ctx, req.DGD); err != nil {
		log.FromContext(ctx).Error(err, "Failed to migrate worker hash")
		return programResult, failWorkloadProgram(reasonFailedToMigrateWorkerHash, err)
	}
	if err := reconcileUnsupportedWorkerRollout(ctx, p.reconciler, req.DGD, true); err != nil {
		return programResult, err
	}
	inputs, err := p.reconciler.reconcileProgramInputs(ctx, req.DGD)
	if inputs.CheckpointStatuses != nil {
		programResult.Status.Checkpoints = inputs.CheckpointStatuses
	}
	if err != nil {
		return programResult, err
	}
	previousRestart := programResult.Status.Restart
	restart := p.reconciler.resolveProgramRestartState(
		ctx,
		req.DGD,
		&programResult.Status,
		p.statusResolver.getUpdatedInProgress,
	)
	recordRestartTransition(previousRestart, restart.Status, &programResult)
	programResult.Status.Restart = restart.Status

	result, err := p.reconcileWorkloads(ctx, workloadReconcileRequest{
		DGD:             req.DGD,
		RestartState:    restart.State,
		CheckpointInfos: inputs.CheckpointInfos,
	})
	if err != nil {
		return programResult, fmt.Errorf("failed to reconcile Grove workloads: %w", err)
	}
	result, err = p.reconciler.reconcileProgramResult(ctx, req.DGD, inputs, restart, result)
	if err != nil {
		return programResult, err
	}

	programResult.applyReconcileResult(req.DGD.Generation, result)
	return programResult, nil
}

// reconcileWorkloads owns the Grove pathway's complete provider workload
// sequence. Provider-specific rendering, scaling, and readiness operations are
// owned by Grove collaborators; shared stable resources remain common.
func (p *groveProgram) reconcileWorkloads(
	ctx context.Context,
	req workloadReconcileRequest,
) (ReconcileResult, error) {
	dynamoDeployment := req.DGD
	logger := log.FromContext(ctx)

	desiredPodCliqueSet, err := p.renderer.Render(
		ctx,
		dynamoDeployment,
		req.RestartState,
		req.CheckpointInfos,
	)
	if err != nil {
		logger.Error(err, "failed to generate the Grove GangSet")
		return ReconcileResult{}, fmt.Errorf("failed to generate the Grove GangSet: %w", err)
	}
	renderDeployment := groveRenderDeployment(dynamoDeployment, desiredPodCliqueSet)

	syncedPodCliqueSet, err := p.reconcilePodCliqueSet(
		ctx,
		dynamoDeployment,
		desiredPodCliqueSet,
	)
	if err != nil {
		logger.Error(err, "failed to reconcile the Grove PodCliqueSet")
		return ReconcileResult{}, fmt.Errorf("failed to reconcile the Grove PodCliqueSet: %w", err)
	}

	// Handle Grove scaling operations after structural changes.
	if err := p.scaler.reconcile(ctx, dynamoDeployment, req.CheckpointInfos); err != nil {
		logger.Error(err, "failed to reconcile Grove scaling")
		return ReconcileResult{}, fmt.Errorf("failed to reconcile Grove scaling: %w", err)
	}

	stableResources, err := p.reconcileStableResources(
		ctx,
		dynamoDeployment,
		renderDeployment,
	)
	if err != nil {
		return ReconcileResult{}, err
	}

	grovePodCliqueSetAsResource, groveReadiness, err := p.observePodCliqueSetReadiness(
		ctx,
		dynamoDeployment,
		syncedPodCliqueSet,
	)
	if err != nil {
		return ReconcileResult{}, err
	}

	resources := append(stableResources, grovePodCliqueSetAsResource)
	return p.checkResourcesReadiness(resources, groveReadiness.Classification), nil
}

// reconcileStableResources owns the services and ingress resources that
// accompany the Grove workload but do not participate in Grove readiness.
func (p *groveProgram) reconcileStableResources(
	ctx context.Context,
	dynamoDeployment *nvidiacomv1beta1.DynamoGraphDeployment,
	renderDeployment *nvidiacomv1beta1.DynamoGraphDeployment,
) ([]Resource, error) {
	r := p.reconciler
	logger := log.FromContext(ctx)

	// Reconcile headless services for model endpoint discovery.
	if err := dynamo.ReconcileModelServicesForComponents(
		ctx,
		r,
		dynamoDeployment,
		dynamo.ComponentsByName(dynamoDeployment),
		dynamoDeployment.Namespace,
	); err != nil {
		logger.Error(err, "failed to reconcile model services")
		return nil, fmt.Errorf("failed to reconcile model services: %w", err)
	}

	resources := []Resource{}
	for i := range renderDeployment.Spec.Components {
		component := &renderDeployment.Spec.Components[i]
		componentName := component.ComponentName

		// If Kubernetes discovery is enabled, create a Service for each
		// component; otherwise create one only for the frontend.
		isK8sDiscoveryEnabled := commoncontroller.IsK8sDiscoveryEnabled(
			r.Config.Discovery.Backend,
			dynamoDeployment.Annotations,
		)
		if isK8sDiscoveryEnabled || string(component.ComponentType) == commonconsts.ComponentTypeFrontend {
			dynamoNamespace := renderDeployment.GetDynamoNamespaceForComponent(component)
			mainComponentService, err := dynamo.GenerateComponentService(dynamo.ComponentServiceParams{
				ServiceName:     dynamo.GetDCDResourceName(dynamoDeployment, componentName, ""),
				Namespace:       dynamoDeployment.Namespace,
				ComponentType:   string(component.ComponentType),
				DynamoNamespace: dynamoNamespace,
				ComponentName:   componentName,
				Labels:          dynamo.GetDGDComponentResourceLabels(renderDeployment, componentName, component),
				Annotations:     dynamo.GetDGDComponentResourceAnnotations(renderDeployment, componentName, component),
				IsK8sDiscovery:  isK8sDiscoveryEnabled,
			})
			if err != nil {
				logger.Error(err, "failed to generate the main component service")
				return nil, fmt.Errorf("failed to generate the main component service: %w", err)
			}
			_, syncedMainComponentService, err := commoncontroller.SyncResource(
				ctx,
				r,
				dynamoDeployment,
				func(context.Context) (*corev1.Service, bool, error) {
					return mainComponentService, false, nil
				},
			)
			if err != nil {
				logger.Error(err, "failed to sync the main component service")
				return nil, fmt.Errorf("failed to sync the main component service: %w", err)
			}
			if syncedMainComponentService != nil {
				if syncedMainComponentService.Annotations == nil {
					syncedMainComponentService.Annotations = make(map[string]string)
				}
				desiredAnnotations := dynamo.GetDGDComponentResourceAnnotations(
					renderDeployment,
					componentName,
					component,
				)
				var updateAnnotations bool
				for key, value := range desiredAnnotations {
					if val, ok := syncedMainComponentService.Annotations[key]; !ok || val != value {
						syncedMainComponentService.Annotations[key] = value
						updateAnnotations = true
					}
				}
				if updateAnnotations {
					if err := r.Update(ctx, syncedMainComponentService); err != nil {
						logger.Error(err, fmt.Sprintf("Failed to update main component service %s.", componentName))
						r.GetRecorder().Eventf(
							dynamoDeployment,
							corev1.EventTypeWarning,
							"UpdateService",
							"Failed to update Service %s: %s",
							componentName,
							err,
						)
						return nil, fmt.Errorf("failed to update main component service %s: %w", componentName, err)
					}
				}
				mainComponentServiceAsResource, err := commoncontroller.NewResource(
					syncedMainComponentService,
					func() (bool, string) {
						return true, ""
					},
				)
				if err != nil {
					return nil, fmt.Errorf("failed to sync the main component service: %w", err)
				}
				resources = append(resources, mainComponentServiceAsResource)
			}
		}

		if string(component.ComponentType) == commonconsts.ComponentTypeFrontend {
			ingressSpec := dynamo.GenerateDefaultIngressSpec(dynamoDeployment, r.Config.Ingress)
			if preservedIngressSpec, ok := dynamo.GetDGDComponentPreservedIngressSpec(dynamoDeployment, componentName); ok {
				ingressSpec = preservedIngressSpec
			}
			mainComponentIngress := dynamo.GenerateComponentIngress(
				ctx,
				dynamo.GetDCDResourceName(dynamoDeployment, componentName, ""),
				dynamoDeployment.Namespace,
				ingressSpec,
			)
			_, syncedMainComponentIngress, err := commoncontroller.SyncResource(
				ctx,
				r,
				dynamoDeployment,
				func(context.Context) (*networkingv1.Ingress, bool, error) {
					if !ingressSpec.Enabled || ingressSpec.IngressControllerClassName == nil {
						logger.Info("Ingress is not enabled")
						return mainComponentIngress, true, nil
					}
					return mainComponentIngress, false, nil
				},
			)
			if err != nil {
				logger.Error(err, "failed to sync the main component ingress")
				return nil, fmt.Errorf("failed to sync the main component ingress: %w", err)
			}
			if syncedMainComponentIngress != nil {
				mainComponentIngressAsResource, err := commoncontroller.NewResource(
					syncedMainComponentIngress,
					func() (bool, string) {
						return true, ""
					},
				)
				if err != nil {
					return nil, fmt.Errorf("failed to create the main component ingress resource: %w", err)
				}
				resources = append(resources, mainComponentIngressAsResource)
			}

			if r.Config.Ingress.UseVirtualService() {
				mainComponentVirtualService := dynamo.GenerateComponentVirtualService(
					ctx,
					dynamo.GetDCDResourceName(dynamoDeployment, componentName, ""),
					dynamoDeployment.Namespace,
					ingressSpec,
				)
				_, syncedMainComponentVirtualService, err := commoncontroller.SyncResource(
					ctx,
					r,
					dynamoDeployment,
					func(context.Context) (*networkingv1beta1.VirtualService, bool, error) {
						if !ingressSpec.IsVirtualServiceEnabled() {
							logger.Info("VirtualService is not enabled")
							return mainComponentVirtualService, true, nil
						}
						return mainComponentVirtualService, false, nil
					},
				)
				if err != nil {
					logger.Error(err, "failed to sync the main component virtual service")
					return nil, fmt.Errorf("failed to sync the main component virtual service: %w", err)
				}
				if syncedMainComponentVirtualService != nil {
					mainComponentVirtualServiceAsResource, err := commoncontroller.NewResource(
						syncedMainComponentVirtualService,
						func() (bool, string) {
							return true, ""
						},
					)
					if err != nil {
						return nil, fmt.Errorf("failed to create the main component virtual service resource: %w", err)
					}
					resources = append(resources, mainComponentVirtualServiceAsResource)
				}
			}
		}
	}

	return resources, nil
}

func (p *groveProgram) reconcilePodCliqueSet(
	ctx context.Context,
	dynamoDeployment *nvidiacomv1beta1.DynamoGraphDeployment,
	desired *grovev1alpha1.PodCliqueSet,
) (*grovev1alpha1.PodCliqueSet, error) {
	r := p.reconciler
	logger := log.FromContext(ctx)

	_, synced, err := commoncontroller.SyncResource(
		ctx,
		r,
		dynamoDeployment,
		func(context.Context) (*grovev1alpha1.PodCliqueSet, bool, error) {
			return desired, false, nil
		},
	)
	if err != nil {
		logger.Error(err, "failed to sync the Grove PodCliqueSet")
		return nil, fmt.Errorf("failed to sync the Grove PodCliqueSet: %w", err)
	}
	return synced, nil
}

// observePodCliqueSetReadiness takes the authoritative Grove snapshot after
// structural and scale reconciliation, then adapts it to the common Resource
// readiness interface without further Kubernetes reads.
func (p *groveProgram) observePodCliqueSetReadiness(
	ctx context.Context,
	dynamoDeployment *nvidiacomv1beta1.DynamoGraphDeployment,
	podCliqueSet *grovev1alpha1.PodCliqueSet,
) (*commoncontroller.Resource, dynamo.GroveReadiness, error) {
	readiness, err := dynamo.EvaluateGroveReadiness(ctx, p.reconciler.Client, dynamoDeployment)
	if err != nil {
		return nil, dynamo.GroveReadiness{}, err
	}
	resource, err := commoncontroller.NewResourceWithComponentStatuses(
		podCliqueSet,
		func() (bool, string, map[string]nvidiacomv1beta1.ComponentReplicaStatus) {
			return readiness.Ready, readiness.Message, readiness.ComponentStatuses
		},
	)
	if err != nil {
		return nil, dynamo.GroveReadiness{}, fmt.Errorf("failed to create the Grove PodCliqueSet resource: %w", err)
	}
	return resource, readiness, nil
}

// checkResourcesReadiness computes the readiness result for the synced Grove
// resources and overlays the Grove-specific Ready reason classification on a
// not-ready result.
func (p *groveProgram) checkResourcesReadiness(
	resources []Resource,
	classification string,
) ReconcileResult {
	result := p.reconciler.checkResourcesReadiness(resources)
	if result.State == nvidiacomv1beta1.DGDStateSuccessful {
		return result
	}
	if classification != "" {
		result.Reason = Reason(classification)
	}
	return result
}
