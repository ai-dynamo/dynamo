/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

package controller_common

import (
	"context"
	"fmt"

	autoscalingv1 "k8s.io/api/autoscaling/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

// ScaleResource scales any Kubernetes resource using the Scale subresource
func ScaleResource(ctx context.Context, kubeClient client.Client, gvr schema.GroupVersionResource, namespace, name string, replicas int32) error {
	logger := log.FromContext(ctx)
	logger.Info("Scaling resource", "gvr", gvr, "name", name, "namespace", namespace, "replicas", replicas)

	// Build the parent resource used to address its scale subresource.
	gvk, err := kubeClient.RESTMapper().KindFor(gvr)
	if err != nil {
		return fmt.Errorf("failed to resolve kind for %s: %w", gvr.String(), err)
	}
	resourceObject, err := kubeClient.Scheme().New(gvk)
	if err != nil {
		return fmt.Errorf("failed to create resource object for %s: %w", gvk.String(), err)
	}
	resource, ok := resourceObject.(client.Object)
	if !ok {
		return fmt.Errorf("resource object for %s does not implement client.Object", gvk.String())
	}
	resource.SetNamespace(namespace)
	resource.SetName(name)

	currentScale := &autoscalingv1.Scale{}
	if err := kubeClient.SubResource("scale").Get(ctx, resource, currentScale); err != nil {
		logger.Error(err, "Failed to get current scale - resource may not support scale subresource", "gvr", gvr, "name", name, "namespace", namespace, "groupResource", gvr.GroupResource())
		return fmt.Errorf("failed to get current scale for %s %s (resource may not support scale subresource): %w", gvr.Resource, name, err)
	}

	if replicas < 0 {
		return fmt.Errorf("replicas must be >= 0, got %d", replicas)
	}

	if currentScale.Spec.Replicas == replicas {
		logger.V(1).Info("Resource already at desired replica count", "gvr", gvr, "name", name, "replicas", replicas)
		return nil
	}

	scaleObj := &autoscalingv1.Scale{
		ObjectMeta: metav1.ObjectMeta{
			Name:            name,
			Namespace:       namespace,
			ResourceVersion: currentScale.ObjectMeta.ResourceVersion,
		},
		Spec: autoscalingv1.ScaleSpec{
			Replicas: replicas,
		},
	}

	logger.V(1).Info("Updating scale", "gvr", gvr, "name", name, "newReplicas", replicas)
	if err := kubeClient.SubResource("scale").Update(ctx, resource, client.WithSubResourceBody(scaleObj)); err != nil {
		logger.Error(err, "Failed to update scale", "gvr", gvr, "name", name, "replicas", replicas)
		return fmt.Errorf("failed to update scale for %s %s: %w", gvr.Resource, name, err)
	}

	logger.Info("Successfully scaled resource", "gvr", gvr, "name", name, "oldReplicas", currentScale.Spec.Replicas, "newReplicas", replicas)
	return nil
}
