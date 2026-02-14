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

package v1alpha1

import (
	"fmt"

	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"sigs.k8s.io/controller-runtime/pkg/conversion"
)

// ConvertTo converts this DynamoGraphDeployment (v1alpha1) to the Hub version (v1beta1).
func (src *DynamoGraphDeployment) ConvertTo(dstRaw conversion.Hub) error {
	dst, ok := dstRaw.(*v1beta1.DynamoGraphDeployment)
	if !ok {
		return fmt.Errorf("expected *v1beta1.DynamoGraphDeployment but got %T", dstRaw)
	}

	dst.ObjectMeta = src.ObjectMeta

	// Spec
	dst.Spec.BackendFramework = src.Spec.BackendFramework
	dst.Spec.Envs = src.Spec.Envs

	// PVCs
	if src.Spec.PVCs != nil {
		dst.Spec.PVCs = make([]v1beta1.PVC, len(src.Spec.PVCs))
		for i, pvc := range src.Spec.PVCs {
			dst.Spec.PVCs[i] = v1beta1.PVC{
				Create:           pvc.Create,
				Name:             pvc.Name,
				StorageClass:     pvc.StorageClass,
				Size:             pvc.Size,
				VolumeAccessMode: pvc.VolumeAccessMode,
			}
		}
	}

	// Services
	if src.Spec.Services != nil {
		dst.Spec.Services = make(map[string]*v1beta1.DynamoComponentDeploymentSharedSpec, len(src.Spec.Services))
		for name, svc := range src.Spec.Services {
			if svc == nil {
				dst.Spec.Services[name] = nil
				continue
			}
			dstSvc := &v1beta1.DynamoComponentDeploymentSharedSpec{}
			convertSharedSpecTo(svc, dstSvc)
			dst.Spec.Services[name] = dstSvc
		}
	}

	// Restart
	if src.Spec.Restart != nil {
		dst.Spec.Restart = &v1beta1.Restart{
			ID: src.Spec.Restart.ID,
		}
		if src.Spec.Restart.Strategy != nil {
			dst.Spec.Restart.Strategy = &v1beta1.RestartStrategy{
				Type:  v1beta1.RestartStrategyType(src.Spec.Restart.Strategy.Type),
				Order: src.Spec.Restart.Strategy.Order,
			}
		}
	}

	// Status
	dst.Status.State = src.Status.State
	dst.Status.Conditions = src.Status.Conditions

	if src.Status.Services != nil {
		dst.Status.Services = make(map[string]v1beta1.ServiceReplicaStatus, len(src.Status.Services))
		for name, svc := range src.Status.Services {
			dst.Status.Services[name] = convertServiceReplicaStatusTo(svc)
		}
	}

	if src.Status.Restart != nil {
		dst.Status.Restart = &v1beta1.RestartStatus{
			ObservedID: src.Status.Restart.ObservedID,
			Phase:      v1beta1.RestartPhase(src.Status.Restart.Phase),
			InProgress: src.Status.Restart.InProgress,
		}
	}

	if src.Status.Checkpoints != nil {
		dst.Status.Checkpoints = make(map[string]v1beta1.ServiceCheckpointStatus, len(src.Status.Checkpoints))
		for name, cp := range src.Status.Checkpoints {
			dst.Status.Checkpoints[name] = v1beta1.ServiceCheckpointStatus{
				CheckpointName: cp.CheckpointName,
				IdentityHash:   cp.IdentityHash,
				Ready:          cp.Ready,
			}
		}
	}

	return nil
}

// ConvertFrom converts from the Hub version (v1beta1) to this DynamoGraphDeployment (v1alpha1).
func (dst *DynamoGraphDeployment) ConvertFrom(srcRaw conversion.Hub) error {
	src, ok := srcRaw.(*v1beta1.DynamoGraphDeployment)
	if !ok {
		return fmt.Errorf("expected *v1beta1.DynamoGraphDeployment but got %T", srcRaw)
	}

	dst.ObjectMeta = src.ObjectMeta

	// Spec
	dst.Spec.BackendFramework = src.Spec.BackendFramework
	dst.Spec.Envs = src.Spec.Envs

	// PVCs
	if src.Spec.PVCs != nil {
		dst.Spec.PVCs = make([]PVC, len(src.Spec.PVCs))
		for i, pvc := range src.Spec.PVCs {
			dst.Spec.PVCs[i] = PVC{
				Create:           pvc.Create,
				Name:             pvc.Name,
				StorageClass:     pvc.StorageClass,
				Size:             pvc.Size,
				VolumeAccessMode: pvc.VolumeAccessMode,
			}
		}
	}

	// Services
	if src.Spec.Services != nil {
		dst.Spec.Services = make(map[string]*DynamoComponentDeploymentSharedSpec, len(src.Spec.Services))
		for name, svc := range src.Spec.Services {
			if svc == nil {
				dst.Spec.Services[name] = nil
				continue
			}
			dstSvc := &DynamoComponentDeploymentSharedSpec{}
			convertSharedSpecFrom(svc, dstSvc)
			dst.Spec.Services[name] = dstSvc
		}
	}

	// Restart
	if src.Spec.Restart != nil {
		dst.Spec.Restart = &Restart{
			ID: src.Spec.Restart.ID,
		}
		if src.Spec.Restart.Strategy != nil {
			dst.Spec.Restart.Strategy = &RestartStrategy{
				Type:  RestartStrategyType(src.Spec.Restart.Strategy.Type),
				Order: src.Spec.Restart.Strategy.Order,
			}
		}
	}

	// Status
	dst.Status.State = src.Status.State
	dst.Status.Conditions = src.Status.Conditions

	if src.Status.Services != nil {
		dst.Status.Services = make(map[string]ServiceReplicaStatus, len(src.Status.Services))
		for name, svc := range src.Status.Services {
			dst.Status.Services[name] = convertServiceReplicaStatusFrom(svc)
		}
	}

	if src.Status.Restart != nil {
		dst.Status.Restart = &RestartStatus{
			ObservedID: src.Status.Restart.ObservedID,
			Phase:      RestartPhase(src.Status.Restart.Phase),
			InProgress: src.Status.Restart.InProgress,
		}
	}

	if src.Status.Checkpoints != nil {
		dst.Status.Checkpoints = make(map[string]ServiceCheckpointStatus, len(src.Status.Checkpoints))
		for name, cp := range src.Status.Checkpoints {
			dst.Status.Checkpoints[name] = ServiceCheckpointStatus{
				CheckpointName: cp.CheckpointName,
				IdentityHash:   cp.IdentityHash,
				Ready:          cp.Ready,
			}
		}
	}

	return nil
}
