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
	commoncontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dra"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	resourcev1 "k8s.io/api/resource/v1"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

// dgdGMSResourceClaimsReconciler owns the ResourceClaimTemplates required by
// GPU Memory Service and inter-pod GMS failover.
type dgdGMSResourceClaimsReconciler struct {
	dgdResourceSyncer
	gates features.Gates
}

func newDGDGMSResourceClaimsReconciler(
	syncer dgdResourceSyncer,
	gates features.Gates,
) *dgdGMSResourceClaimsReconciler {
	return &dgdGMSResourceClaimsReconciler{dgdResourceSyncer: syncer, gates: gates}
}

func (r *dgdGMSResourceClaimsReconciler) Reconcile(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
) error {
	if !r.gates.Enabled(features.DRA) {
		for i := range dgd.Spec.Components {
			component := &dgd.Spec.Components[i]
			if dynamo.GetGPUMemoryService(component) != nil || component.IsInterPodFailoverEnabled() {
				return fmt.Errorf(
					"gpuMemoryService / inter-pod GMS failover requires DRA (Dynamic Resource Allocation), " +
						"but DRA is not available (either the resource.k8s.io/v1 API is not registered on this cluster, " +
						"which requires Kubernetes 1.34+, or DRA has been explicitly disabled in the operator configuration)")
			}
		}
		return nil
	}

	logger := log.FromContext(ctx)
	for i := range dgd.Spec.Components {
		component := &dgd.Spec.Components[i]
		gmsSpec := dynamo.GetGPUMemoryService(component)
		componentName := component.ComponentName
		gpuCount := 0
		deviceClassName := ""
		if gmsSpec != nil {
			var err error
			gpuCount, err = dra.ExtractGPUCountFromResourceRequirements(dynamo.GetMainContainerResources(component))
			if err != nil {
				return fmt.Errorf("invalid GPU resource requirements for GMS ResourceClaimTemplate for %s: %w", componentName, err)
			}
			deviceClassName = gmsSpec.DeviceClassName
			if deviceClassName == "" {
				deviceClassName = dra.DefaultDeviceClassName
			}
		}

		claimTemplateName := dra.ResourceClaimTemplateName(dgd.Name, componentName)
		if _, _, err := commoncontroller.SyncResource(ctx, r, dgd, func(ctx context.Context) (*resourcev1.ResourceClaimTemplate, bool, error) {
			return dra.GenerateResourceClaimTemplate(ctx, r.Client, claimTemplateName, dgd.Namespace, gpuCount, deviceClassName)
		}); err != nil {
			logger.Error(err, "failed to sync GMS ResourceClaimTemplate", "component", componentName)
			return fmt.Errorf("failed to sync GMS ResourceClaimTemplate for %s: %w", componentName, err)
		}
	}
	return nil
}
