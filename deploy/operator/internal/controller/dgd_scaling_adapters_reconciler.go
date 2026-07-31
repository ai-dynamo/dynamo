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
	commoncontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/tools/record"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

type dgdScalingAdaptersReconciler struct {
	dgdResourceSyncer
}

func newDGDScalingAdaptersReconciler(
	kubeClient client.Client,
	recorder record.EventRecorder,
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
	for i := range dgd.Spec.Components {
		component := &dgd.Spec.Components[i]
		componentName := component.ComponentName
		scalingAdapterEnabled := component.ScalingAdapter != nil

		currentReplicas := int32(1)
		if component.Replicas != nil {
			currentReplicas = *component.Replicas
		}

		_, _, err := commoncontroller.SyncResource(
			ctx,
			r,
			dgd,
			func(context.Context) (*nvidiacomv1alpha1.DynamoGraphDeploymentScalingAdapter, bool, error) {
				adapter := &nvidiacomv1alpha1.DynamoGraphDeploymentScalingAdapter{
					ObjectMeta: metav1.ObjectMeta{
						Name:      generateAdapterName(dgd.Name, componentName),
						Namespace: dgd.Namespace,
						Labels: map[string]string{
							consts.KubeLabelDynamoGraphDeploymentName: dgd.Name,
							consts.KubeLabelDynamoComponent:           componentName,
						},
					},
					Spec: nvidiacomv1alpha1.DynamoGraphDeploymentScalingAdapterSpec{
						Replicas: currentReplicas,
						DGDRef: nvidiacomv1alpha1.DynamoGraphDeploymentServiceRef{
							Name:        dgd.Name,
							ServiceName: componentName,
						},
					},
				}
				return adapter, !scalingAdapterEnabled, nil
			},
		)
		if err != nil {
			logger.Error(err, "Failed to sync DynamoGraphDeploymentScalingAdapter", "component", componentName)
			return err
		}
	}

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
				corev1.EventTypeNormal,
				"AdapterDeleted",
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
