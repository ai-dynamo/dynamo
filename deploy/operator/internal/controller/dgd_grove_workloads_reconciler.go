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
	"strconv"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/checkpoint"
	commoncontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	"k8s.io/client-go/scale"
	"k8s.io/client-go/tools/events"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

// groveWorkloadsReconciler owns the complete provider workload sequence while
// exposing no dependency on the top-level DGD controller.
type groveWorkloadsReconciler struct {
	syncer          dgdResourceSyncer
	reader          client.Reader
	renderer        *groveWorkloadRenderer
	scaler          *groveScaler
	stableResources *groveStableResourcesReconciler
}

func newGroveWorkloadsReconciler(
	kubeClient client.Client,
	recorder events.EventRecorder,
	config *configv1alpha1.OperatorConfiguration,
	runtimeConfig *commoncontroller.RuntimeConfig,
	dockerSecretRetriever DockerSecretRetriever,
	scaleClient scale.ScalesGetter,
) *groveWorkloadsReconciler {
	return &groveWorkloadsReconciler{
		syncer: newDGDResourceSyncer(kubeClient, recorder),
		reader: kubeClient,
		renderer: newGroveWorkloadRenderer(
			kubeClient,
			config,
			runtimeConfig,
			dockerSecretRetriever,
		),
		scaler:          newGroveScaler(scaleClient),
		stableResources: newGroveStableResourcesReconciler(kubeClient, recorder, config),
	}
}

func (r *groveWorkloadsReconciler) Reconcile(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	restartState *dynamo.RestartState,
	checkpointInfos map[string]*checkpoint.CheckpointInfo,
	workerGenerationChanged bool,
	afterPodCliqueSetSync func() error,
) (ReconcileResult, error) {
	logger := log.FromContext(ctx)

	renderedPodCliqueSet, err := r.renderer.Render(
		ctx,
		dgd,
		restartState,
		checkpointInfos,
		workerGenerationChanged,
	)
	if err != nil {
		logger.Error(err, "failed to generate the Grove GangSet")
		return ReconcileResult{}, fmt.Errorf("failed to generate the Grove GangSet: %w", err)
	}
	renderDeployment, err := groveRenderDeployment(
		dgd,
		renderedPodCliqueSet.desired,
		renderedPodCliqueSet.workerHashSuffixNeeded,
	)
	if err != nil {
		return ReconcileResult{}, fmt.Errorf("failed to prepare Grove deployment: %w", err)
	}

	syncedPodCliqueSet, err := r.reconcilePodCliqueSet(ctx, dgd, renderedPodCliqueSet)
	if err != nil {
		logger.Error(err, "failed to reconcile the Grove PodCliqueSet")
		return ReconcileResult{}, fmt.Errorf("failed to reconcile the Grove PodCliqueSet: %w", err)
	}
	// The PCS write is the transition's write barrier. A failed callback
	// leaves the DGD hash unchanged, so the next reconcile replans from live state.
	if afterPodCliqueSetSync != nil {
		if err := afterPodCliqueSetSync(); err != nil {
			return ReconcileResult{}, fmt.Errorf("commit worker hash after Grove PodCliqueSet sync: %w", err)
		}
	}

	if err := r.scaler.Reconcile(ctx, dgd, checkpointInfos); err != nil {
		logger.Error(err, "failed to reconcile Grove scaling")
		return ReconcileResult{}, fmt.Errorf("failed to reconcile Grove scaling: %w", err)
	}

	stableResources, err := r.stableResources.Reconcile(ctx, dgd, renderDeployment)
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
	return checkGroveResourcesReadiness(resources, readiness.Classification), nil
}

func (r *groveWorkloadsReconciler) reconcilePodCliqueSet(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	rendered *grovePodCliqueSetRender,
) (*grovev1alpha1.PodCliqueSet, error) {
	if rendered.existing == nil {
		if err := ctrl.SetControllerReference(dgd, rendered.desired, r.syncer.Scheme()); err != nil {
			return nil, fmt.Errorf("set Grove PodCliqueSet owner reference: %w", err)
		}
		hash, err := commoncontroller.GetSpecHash(rendered.desired)
		if err != nil {
			return nil, fmt.Errorf("hash Grove PodCliqueSet spec: %w", err)
		}
		annotations := rendered.desired.GetAnnotations()
		if annotations == nil {
			annotations = make(map[string]string)
		}
		annotations[commoncontroller.NvidiaAnnotationHashKey] = hash
		annotations[commoncontroller.NvidiaAnnotationGenerationKey] = "1"
		rendered.desired.SetAnnotations(annotations)
		if err := r.syncer.Create(ctx, rendered.desired); err != nil {
			return nil, fmt.Errorf("create Grove PodCliqueSet: %w", err)
		}
		return rendered.desired, nil
	}

	change, err := commoncontroller.GetSpecChangeResult(rendered.existing, rendered.desired)
	if err != nil {
		return nil, fmt.Errorf("compare Grove PodCliqueSet spec: %w", err)
	}
	if !change.NeedsUpdate {
		return rendered.existing, nil
	}
	if change.NewHash == nil {
		return nil, fmt.Errorf("Grove PodCliqueSet update has no desired spec hash")
	}

	synced := rendered.existing.DeepCopy()
	if change.SpecNeedsUpdate {
		if err := commoncontroller.CopySpec(rendered.desired, synced); err != nil {
			return nil, fmt.Errorf("copy Grove PodCliqueSet spec: %w", err)
		}
	}
	annotations := synced.GetAnnotations()
	if annotations == nil {
		annotations = make(map[string]string)
	}
	annotations[commoncontroller.NvidiaAnnotationHashKey] = *change.NewHash
	annotations[commoncontroller.NvidiaAnnotationGenerationKey] = strconv.FormatInt(change.NewGeneration, 10)
	synced.SetAnnotations(annotations)
	if err := r.syncer.Update(ctx, synced); err != nil {
		return nil, fmt.Errorf("update Grove PodCliqueSet: %w", err)
	}
	return synced, nil
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
