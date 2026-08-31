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
	"strings"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/tools/events"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

type dgdScalingAdaptersReconciler struct {
	dgdResourceSyncer
}

func newDGDScalingAdaptersReconciler(
	kubeClient client.Client,
	recorder events.EventRecorder,
) *dgdScalingAdaptersReconciler {
	return &dgdScalingAdaptersReconciler{
		dgdResourceSyncer: newDGDResourceSyncer(kubeClient, recorder),
	}
}

// Reconcile ensures a DynamoGraphDeploymentScalingAdapter exists for each DGD
// component that has scaling explicitly enabled.
func (r *dgdScalingAdaptersReconciler) Reconcile(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
) error {
	logger := log.FromContext(ctx)

	// Reconcile adapters for current components while preserving adapter-owned replicas.
	for i := range dgd.Spec.Components {
		component := &dgd.Spec.Components[i]
		componentName := component.ComponentName
		adapterName := generateAdapterName(dgd.Name, componentName)
		adapter := &nvidiacomv1alpha1.DynamoGraphDeploymentScalingAdapter{
			ObjectMeta: metav1.ObjectMeta{
				Name:      adapterName,
				Namespace: dgd.Namespace,
			},
		}

		// Remove the adapter when scaling is no longer enabled for the component.
		if component.ScalingAdapter == nil {
			if err := r.Delete(ctx, adapter); err != nil {
				if apierrors.IsNotFound(err) {
					continue
				}
				logger.Error(err, "Failed to delete DynamoGraphDeploymentScalingAdapter", "component", componentName)
				return err
			}

			logger.Info("Deleted DynamoGraphDeploymentScalingAdapter", "adapter", adapterName, "component", componentName)
			if r.recorder != nil {
				r.recorder.Eventf(
					dgd,
					adapter,
					corev1.EventTypeNormal,
					"AdapterDeleted",
					"Delete",
					"Deleted scaling adapter %s for component %s",
					adapterName,
					componentName,
				)
			}
			continue
		}

		initialReplicas := ptr.Deref(component.Replicas, int32(1))
		operation, err := controllerutil.CreateOrPatch(ctx, r.Client, adapter, func() error {
			if adapter.Labels == nil {
				adapter.Labels = map[string]string{}
			}
			adapter.Labels[consts.KubeLabelDynamoGraphDeploymentName] = dgd.Name
			adapter.Labels[consts.KubeLabelDynamoComponent] = componentName
			adapter.Spec.DGDRef = nvidiacomv1alpha1.DynamoGraphDeploymentServiceRef{
				Name:        dgd.Name,
				ServiceName: componentName,
			}

			// Seed replicas only when creating the adapter; it owns subsequent changes.
			if adapter.GetResourceVersion() == "" {
				adapter.Spec.Replicas = initialReplicas
			}

			return controllerutil.SetControllerReference(dgd, adapter, r.Scheme())
		})
		if err != nil {
			logger.Error(err, "Failed to reconcile DynamoGraphDeploymentScalingAdapter", "component", componentName)
			return err
		}

		// Emit resource events only after the corresponding mutation succeeds.
		switch operation {
		case controllerutil.OperationResultCreated:
			logger.Info("Created DynamoGraphDeploymentScalingAdapter", "adapter", adapterName, "component", componentName)
			if r.recorder != nil {
				r.recorder.Eventf(
					dgd,
					adapter,
					corev1.EventTypeNormal,
					"AdapterCreated",
					"Create",
					"Created scaling adapter %s for component %s",
					adapterName,
					componentName,
				)
			}
		case controllerutil.OperationResultUpdated:
			logger.Info("Updated DynamoGraphDeploymentScalingAdapter", "adapter", adapterName, "component", componentName)
			if r.recorder != nil {
				r.recorder.Eventf(
					dgd,
					adapter,
					corev1.EventTypeNormal,
					"AdapterUpdated",
					"Update",
					"Updated scaling adapter %s for component %s",
					adapterName,
					componentName,
				)
			}
		}
	}

	// Delete adapters whose components have been removed from the DGD.
	adapterList := &nvidiacomv1alpha1.DynamoGraphDeploymentScalingAdapterList{}
	if err := r.List(
		ctx,
		adapterList,
		client.InNamespace(dgd.Namespace),
		client.MatchingLabels{consts.KubeLabelDynamoGraphDeploymentName: dgd.Name},
	); err != nil {
		logger.Error(err, "Failed to list DynamoGraphDeploymentScalingAdapters")
		return err
	}

	for i := range adapterList.Items {
		adapter := &adapterList.Items[i]
		componentName := adapter.Spec.DGDRef.ServiceName
		if dgd.GetComponentByName(componentName) != nil {
			continue
		}

		logger.Info("Deleting orphaned DynamoGraphDeploymentScalingAdapter", "adapter", adapter.Name, "component", componentName)
		if err := r.Delete(ctx, adapter); err != nil {
			if apierrors.IsNotFound(err) {
				continue
			}
			logger.Error(err, "Failed to delete orphaned adapter", "adapter", adapter.Name)
			return err
		}
		if r.recorder != nil {
			r.recorder.Eventf(
				dgd,
				adapter,
				corev1.EventTypeNormal,
				"AdapterDeleted",
				"Delete",
				"Deleted orphaned scaling adapter %s for removed component %s",
				adapter.Name,
				componentName,
			)
		}
	}

	return nil
}

func generateAdapterName(dgdName, componentName string) string {
	return fmt.Sprintf("%s-%s", dgdName, strings.ToLower(componentName))
}
