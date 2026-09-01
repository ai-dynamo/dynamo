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
	"crypto/sha256"
	"fmt"
	"strings"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/componentgroups"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	k8svalidation "k8s.io/apimachinery/pkg/util/validation"
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
// component or component group that has scaling explicitly enabled.
//
//nolint:gocyclo // Component and component-group adapters share one reconciliation pass.
func (r *dgdScalingAdaptersReconciler) Reconcile(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
) error {
	logger := log.FromContext(ctx)
	componentGroups := componentgroups.New(dgd.Spec.Experimental)

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

		// Grouped components scale through their group adapter, never an individual adapter.
		if component.ScalingAdapter == nil || componentGroups.IsGrouped(componentName) {
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

	// Reconcile one adapter per component group while preserving adapter-owned replicas.
	for _, group := range componentGroups.Groups() {
		groupName := group.Name
		adapterName := generateAdapterName(dgd.Name, groupName)
		adapter := &nvidiacomv1alpha1.DynamoGraphDeploymentScalingAdapter{
			ObjectMeta: metav1.ObjectMeta{
				Name:      adapterName,
				Namespace: dgd.Namespace,
			},
		}

		// Remove the group adapter when group-level scaling is no longer enabled.
		if group.ScalingAdapter == nil {
			if err := r.Delete(ctx, adapter); err != nil {
				if apierrors.IsNotFound(err) {
					continue
				}
				logger.Error(err, "Failed to delete DynamoGraphDeploymentScalingAdapter", "componentGroup", groupName)
				return err
			}

			logger.Info("Deleted DynamoGraphDeploymentScalingAdapter", "adapter", adapterName, "componentGroup", groupName)
			if r.recorder != nil {
				r.recorder.Eventf(
					dgd,
					adapter,
					corev1.EventTypeNormal,
					"AdapterDeleted",
					"Delete",
					"Deleted scaling adapter %s for component group %s",
					adapterName,
					groupName,
				)
			}
			continue
		}

		operation, err := controllerutil.CreateOrPatch(ctx, r.Client, adapter, func() error {
			if adapter.Labels == nil {
				adapter.Labels = map[string]string{}
			}
			adapter.Labels[consts.KubeLabelDynamoGraphDeploymentName] = dgd.Name
			adapter.Labels[consts.KubeLabelDynamoComponentGroup] = groupName
			adapter.Spec.DGDRef = nvidiacomv1alpha1.DynamoGraphDeploymentServiceRef{
				Name:               dgd.Name,
				ComponentGroupName: groupName,
			}

			// Seed replicas only when creating the adapter; it owns subsequent changes.
			if adapter.GetResourceVersion() == "" {
				adapter.Spec.Replicas = group.Replicas
			}

			return controllerutil.SetControllerReference(dgd, adapter, r.Scheme())
		})
		if err != nil {
			logger.Error(err, "Failed to reconcile DynamoGraphDeploymentScalingAdapter", "componentGroup", groupName)
			return err
		}

		// Emit resource events only after the corresponding mutation succeeds.
		switch operation {
		case controllerutil.OperationResultCreated:
			logger.Info("Created DynamoGraphDeploymentScalingAdapter", "adapter", adapterName, "componentGroup", groupName)
			if r.recorder != nil {
				r.recorder.Eventf(
					dgd,
					adapter,
					corev1.EventTypeNormal,
					"AdapterCreated",
					"Create",
					"Created scaling adapter %s for component group %s",
					adapterName,
					groupName,
				)
			}
		case controllerutil.OperationResultUpdated:
			logger.Info("Updated DynamoGraphDeploymentScalingAdapter", "adapter", adapterName, "componentGroup", groupName)
			if r.recorder != nil {
				r.recorder.Eventf(
					dgd,
					adapter,
					corev1.EventTypeNormal,
					"AdapterUpdated",
					"Update",
					"Updated scaling adapter %s for component group %s",
					adapterName,
					groupName,
				)
			}
		}
	}

	// Delete adapters whose component or component-group target was removed.
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
		componentGroupName := adapter.Spec.DGDRef.ComponentGroupName

		// Retain only adapters whose exact component or group target still exists.
		targetName := componentName
		targetKind := "component"
		targetExists := dgd.GetComponentByName(componentName) != nil && !componentGroups.IsGrouped(componentName)
		if componentGroupName != "" {
			targetName = componentGroupName
			targetKind = "component group"
			targetExists = componentGroups.HasGroup(componentGroupName)
		}
		if targetExists {
			continue
		}

		logger.Info(
			"Deleting orphaned DynamoGraphDeploymentScalingAdapter",
			"adapter", adapter.Name,
			"targetKind", targetKind,
			"target", targetName,
		)
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
				"Deleted orphaned scaling adapter %s for removed %s %s",
				adapter.Name,
				targetKind,
				targetName,
			)
		}
	}

	return nil
}

func generateAdapterName(dgdName, componentName string) string {
	name := fmt.Sprintf("%s-%s", dgdName, strings.ToLower(componentName))
	if len(k8svalidation.IsDNS1123Subdomain(name)) == 0 {
		return name
	}

	// Preserve uniqueness after constraining an invalid combination to a DNS label.
	hash := sha256.Sum256([]byte(name))
	hashSuffix := fmt.Sprintf("-%x", hash[:8])
	prefix := strings.ReplaceAll(name, ".", "-")
	maxPrefixLength := k8svalidation.DNS1123LabelMaxLength - len(hashSuffix)
	if len(prefix) > maxPrefixLength {
		prefix = prefix[:maxPrefixLength]
	}
	prefix = strings.TrimRight(prefix, "-")
	return prefix + hashSuffix
}
