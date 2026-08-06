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
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/checkpoint"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	commoncontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/scale"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

type groveScaler struct {
	scaleClient scale.ScalesGetter
}

func newGroveScaler(scaleClient scale.ScalesGetter) *groveScaler {
	return &groveScaler{scaleClient: scaleClient}
}

// Reconcile applies component replica changes to the Grove resources created
// asynchronously from the PodCliqueSet.
func (s *groveScaler) Reconcile(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	checkpointInfos map[string]*checkpoint.CheckpointInfo,
) error {
	logger := log.FromContext(ctx)
	logger.V(1).Info("Reconciling Grove scaling operations")

	for i := range dgd.Spec.Components {
		component := &dgd.Spec.Components[i]
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
		resourceName := dynamo.GroveComponentResourceName(dgd, componentName)
		resourceKind := "PodClique"
		gvr := consts.PodCliqueGVR
		if usesPCSG {
			resourceKind = "PodCliqueScalingGroup"
			gvr = consts.PodCliqueScalingGroupGVR
		}
		if err := s.scaleResource(
			ctx,
			gvr,
			resourceName,
			dgd.Namespace,
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
			return fmt.Errorf("failed to scale %s %s: %w", resourceKind, resourceName, err)
		}
	}

	logger.V(1).Info("Successfully reconciled Grove scaling operations")
	return nil
}

func (s *groveScaler) scaleResource(
	ctx context.Context,
	gvr schema.GroupVersionResource,
	resourceName string,
	namespace string,
	newReplicas int32,
) error {
	err := commoncontroller.ScaleResource(ctx, s.scaleClient, gvr, namespace, resourceName, newReplicas)
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
