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

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/checkpoint"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	commoncontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

type groveScaler struct {
	client client.Client
}

func newGroveScaler(kubeClient client.Client) *groveScaler {
	return &groveScaler{client: kubeClient}
}

// Reconcile applies component replica changes to the Grove resources created
// asynchronously from the PodCliqueSet. scalingBlocked is resolved after PCS
// synchronization. While blocked, observe capacity without writing replicas.
// The result reports whether an explicit replica change is waiting.
func (s *groveScaler) Reconcile(
	ctx context.Context,
	req groveReconcileRequest,
	checkpointInfos map[string]*checkpoint.CheckpointInfo,
	scalingBlocked bool,
) (deferred bool, err error) {
	logger := log.FromContext(ctx)
	logger.V(1).Info("Reconciling Grove scaling operations")
	managedComponents := req.ManagedComponents()
	pcsName := dynamo.PCSNameForDGD(req.DGD, req.IsDelegated)

	for i := range managedComponents {
		component := &managedComponents[i]
		componentName := component.ComponentName
		info := checkpointInfos[componentName]
		gated := info != nil &&
			info.Enabled &&
			info.StartupPolicy == nvidiacomv1alpha1.CheckpointStartupPolicyWaitForCheckpoint &&
			!info.Ready
		if component.Replicas == nil && !gated {
			continue
		}
		replicas := int32(1)
		if component.Replicas != nil {
			replicas = *component.Replicas
		}
		if gated {
			replicas = 0
		}

		usesPCSG := component.UsesPCSG()
		resourceName := dynamo.GroveComponentResourceName(pcsName, componentName)
		resourceKind := "PodClique"
		gvr := consts.PodCliqueGVR
		if usesPCSG {
			resourceKind = "PodCliqueScalingGroup"
			gvr = consts.PodCliqueScalingGroupGVR
		}
		// Observe capacity while the workload configuration or coherent rollout is pending.
		if scalingBlocked {
			var child client.Object
			if usesPCSG {
				child = &grovev1alpha1.PodCliqueScalingGroup{}
			} else {
				child = &grovev1alpha1.PodClique{}
			}
			if err := s.client.Get(ctx, client.ObjectKey{Name: resourceName, Namespace: req.DGD.Namespace}, child); err != nil {
				if apierrors.IsNotFound(err) {
					continue
				}
				return deferred, err
			}
			switch resource := child.(type) {
			case *grovev1alpha1.PodClique:
				deferred = deferred || resource.Spec.Replicas != replicas
			case *grovev1alpha1.PodCliqueScalingGroup:
				deferred = deferred || resource.Spec.Replicas != replicas
			}
			continue
		}

		// Unexpected admission or API failures retain the normal controller retry path.
		if err := s.scaleResource(
			ctx,
			gvr,
			resourceName,
			req.DGD.Namespace,
			replicas,
		); err != nil {
			logger.Error(
				err,
				"Failed to scale Grove resource",
				"resourceKind", resourceKind,
				"componentName", componentName,
				"resourceName", resourceName,
				"replicas", replicas,
			)
			return deferred, fmt.Errorf("failed to scale %s %s: %w", resourceKind, resourceName, err)
		}
	}

	logger.V(1).Info("Successfully reconciled Grove scaling operations", "deferred", deferred)
	return deferred, nil
}

func (s *groveScaler) scaleResource(
	ctx context.Context,
	gvr schema.GroupVersionResource,
	resourceName string,
	namespace string,
	newReplicas int32,
) error {
	err := commoncontroller.ScaleResource(ctx, s.client, gvr, namespace, resourceName, newReplicas)
	if apierrors.IsNotFound(err) {
		// Grove creates these resources asynchronously after the PodCliqueSet.
		// A later reconciliation retries the scale once the child exists.
		log.FromContext(ctx).V(1).Info(
			"Grove resource not found yet, skipping scaling for now",
			"gvr", gvr,
			"name", resourceName,
			"namespace", namespace,
		)
		return nil
	}
	return err
}
