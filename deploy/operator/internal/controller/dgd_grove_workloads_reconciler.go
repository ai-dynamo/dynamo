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
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/checkpoint"
	commoncontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/provideroverride"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/tools/events"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

// groveWorkloadsReconciler owns the complete provider workload sequence while
// exposing no dependency on the top-level DGD controller.
type groveWorkloadsReconciler struct {
	syncer          dgdResourceSyncer
	rollout         *dgdWorkerRolloutReconciler
	reader          client.Reader
	renderer        *groveWorkloadRenderer
	scaler          *groveScaler
	stableResources *groveStableResourcesReconciler
}

func newGroveWorkloadsReconciler(
	kubeClient client.Client,
	recorder events.EventRecorder,
	rollout *dgdWorkerRolloutReconciler,
	config *configv1alpha1.OperatorConfiguration,
	runtimeConfig *commoncontroller.RuntimeConfig,
	dockerSecretRetriever DockerSecretRetriever,
) *groveWorkloadsReconciler {
	return &groveWorkloadsReconciler{
		syncer:  newDGDResourceSyncer(kubeClient, recorder),
		rollout: rollout,
		reader:  kubeClient,
		renderer: newGroveWorkloadRenderer(
			kubeClient,
			config,
			runtimeConfig,
			dockerSecretRetriever,
		),
		scaler:          newGroveScaler(kubeClient),
		stableResources: newGroveStableResourcesReconciler(kubeClient, recorder, config),
	}
}

func (r *groveWorkloadsReconciler) Reconcile(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	restartState *dynamo.RestartState,
	checkpointInfos map[string]*checkpoint.CheckpointInfo,
) (ReconcileResult, error) {
	logger := log.FromContext(ctx)

	workerHashTransition, err := r.rollout.planUnsupportedWorkerHashTransition(dgd)
	if err != nil {
		return ReconcileResult{}, failWorkloadProgram(reasonRollingUpdateFailed, err)
	}
	renderedPodCliqueSet, err := r.renderer.Render(
		ctx,
		dgd,
		restartState,
		checkpointInfos,
		workerHashTransition.hashChanged,
	)
	if err != nil {
		logger.Error(err, "failed to generate the Grove GangSet")
		return ReconcileResult{}, fmt.Errorf("failed to generate the Grove GangSet: %w", err)
	}
	syncedPodCliqueSet, pcsWasWritten, err := r.reconcilePodCliqueSet(ctx, dgd, renderedPodCliqueSet)
	if err != nil {
		logger.Error(err, "failed to reconcile the Grove PodCliqueSet")
		return ReconcileResult{}, fmt.Errorf("failed to reconcile the Grove PodCliqueSet: %w", err)
	}
	// Project the hash only once the informer cache reflects the target suffix on every worker clique.
	// If we wrote the PCS this reconcile, the informer has not yet caught up with that write; defer
	// the commit to the next watch-driven reconcile, where pcsWasWritten will be false.
	if workerHashTransition.needsCommit() && !pcsWasWritten {
		// allUnstamped is valid only for a pre-existing legacy PCS that was never suffixed.
		// A newly created PCS must carry the canonical hash label, so require allCanonical.
		isLegacyUnsuffixed := workerHashTransition.isFirstGeneration &&
			renderedPodCliqueSet.existing != nil &&
			!podCliqueSetUsesGroveWorkerHashSuffix(dgd, renderedPodCliqueSet.existing)
		observed, err := podCliqueSetObservesWorkerHash(dgd, renderedPodCliqueSet.existing, isLegacyUnsuffixed)
		if err != nil {
			return ReconcileResult{}, failWorkloadProgram(
				reasonRollingUpdateFailed,
				fmt.Errorf("observe Grove worker hash before projection: %w", err),
			)
		}
		if observed {
			if err := r.rollout.commitUnsupportedWorkerHashTransition(ctx, dgd, workerHashTransition, true); err != nil {
				return ReconcileResult{}, failWorkloadProgram(
					reasonRollingUpdateFailed,
					fmt.Errorf("project observed Grove worker hash: %w", err),
				)
			}
		}
	}

	if err := r.scaler.Reconcile(ctx, dgd, checkpointInfos); err != nil {
		logger.Error(err, "failed to reconcile Grove scaling")
		return ReconcileResult{}, fmt.Errorf("failed to reconcile Grove scaling: %w", err)
	}

	stableResources, err := r.stableResources.Reconcile(ctx, dgd, renderedPodCliqueSet.renderDeployment)
	if err != nil {
		return ReconcileResult{}, err
	}

	podCliqueSetResource, readiness, err := r.observePodCliqueSetReadiness(
		ctx,
		dgd,
		syncedPodCliqueSet,
	)
	if err != nil {
		return ReconcileResult{}, err
	}

	resources := append(stableResources, podCliqueSetResource)
	result := checkGroveResourcesReadiness(resources, readiness.Classification)
	applyComponentGPUShapes(result.ComponentStatus, renderedPodCliqueSet.gpuShapes)
	return result, nil
}

func (r *groveWorkloadsReconciler) reconcilePodCliqueSet(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	rendered *grovePodCliqueSetRender,
) (*grovev1alpha1.PodCliqueSet, bool, error) {
	// Compose opaque provider fields only at the serialization boundary.
	desiredProviderObject, err := provideroverride.ComposeGroveOverrides(dgd, rendered.desired)
	if err != nil {
		return nil, false, fmt.Errorf("compose Grove provider overrides: %w", err)
	}

	// Serialize the exact typed observation selected during rendering.
	var observedProviderObject *unstructured.Unstructured
	if rendered.existing != nil {
		object, conversionErr := runtime.DefaultUnstructuredConverter.ToUnstructured(rendered.existing)
		if conversionErr != nil {
			return nil, false, fmt.Errorf("convert observed Grove PodCliqueSet to unstructured: %w", conversionErr)
		}
		observedProviderObject = &unstructured.Unstructured{Object: object}
		observedProviderObject.SetGroupVersionKind(desiredProviderObject.GroupVersionKind())
	}

	// Synchronize without rereading or mutating the render observation.
	wasWritten, synced, err := commoncontroller.SyncObservedResource(
		ctx,
		&r.syncer,
		dgd,
		observedProviderObject,
		desiredProviderObject,
	)
	if err != nil {
		return nil, false, fmt.Errorf("sync Grove PodCliqueSet: %w", err)
	}

	// Adapt the synchronized wire object to the typed Grove readiness view.
	podCliqueSet := &grovev1alpha1.PodCliqueSet{}
	if err := runtime.DefaultUnstructuredConverter.FromUnstructured(synced.Object, podCliqueSet); err != nil {
		return nil, false, fmt.Errorf("convert synchronized Grove PodCliqueSet: %w", err)
	}
	return podCliqueSet, wasWritten, nil
}

// observePodCliqueSetReadiness takes the authoritative Grove snapshot after
// structural and scale reconciliation, then adapts it to the common Resource
// readiness interface without further Kubernetes reads.
func (r *groveWorkloadsReconciler) observePodCliqueSetReadiness(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	podCliqueSet *grovev1alpha1.PodCliqueSet,
) (*commoncontroller.Resource, dynamo.GroveReadiness, error) {
	readiness, err := dynamo.EvaluateGroveReadiness(ctx, r.reader, dgd, podCliqueSet)
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

func checkGroveResourcesReadiness(
	resources []Resource,
	classification string,
) ReconcileResult {
	result := checkResourcesReadiness(resources)
	if result.State == nvidiacomv1beta1.DGDStateSuccessful {
		return result
	}
	if classification != "" {
		result.Reason = Reason(classification)
	}
	return result
}
