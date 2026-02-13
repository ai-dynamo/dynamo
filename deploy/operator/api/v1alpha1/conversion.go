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

// ---------- DynamoCheckpoint ----------

// ConvertTo converts this DynamoCheckpoint (v1alpha1) to the Hub version (v1beta1).
func (src *DynamoCheckpoint) ConvertTo(dstRaw conversion.Hub) error {
	dst, ok := dstRaw.(*v1beta1.DynamoCheckpoint)
	if !ok {
		return fmt.Errorf("expected *v1beta1.DynamoCheckpoint but got %T", dstRaw)
	}

	// ObjectMeta
	dst.ObjectMeta = src.ObjectMeta

	// Spec
	dst.Spec.Identity = convertCheckpointIdentityTo(src.Spec.Identity)
	dst.Spec.Job = v1beta1.DynamoCheckpointJobConfig{
		PodTemplateSpec:         src.Spec.Job.PodTemplateSpec,
		ActiveDeadlineSeconds:   src.Spec.Job.ActiveDeadlineSeconds,
		BackoffLimit:            src.Spec.Job.BackoffLimit,
		TTLSecondsAfterFinished: src.Spec.Job.TTLSecondsAfterFinished,
	}

	// Status
	dst.Status.Phase = v1beta1.DynamoCheckpointPhase(src.Status.Phase)
	dst.Status.IdentityHash = src.Status.IdentityHash
	dst.Status.Location = src.Status.Location
	dst.Status.StorageType = v1beta1.DynamoCheckpointStorageType(src.Status.StorageType)
	dst.Status.JobName = src.Status.JobName
	dst.Status.CreatedAt = src.Status.CreatedAt
	dst.Status.Message = src.Status.Message
	dst.Status.Conditions = src.Status.Conditions

	return nil
}

// ConvertFrom converts from the Hub version (v1beta1) to this DynamoCheckpoint (v1alpha1).
func (dst *DynamoCheckpoint) ConvertFrom(srcRaw conversion.Hub) error {
	src, ok := srcRaw.(*v1beta1.DynamoCheckpoint)
	if !ok {
		return fmt.Errorf("expected *v1beta1.DynamoCheckpoint but got %T", srcRaw)
	}

	// ObjectMeta
	dst.ObjectMeta = src.ObjectMeta

	// Spec
	dst.Spec.Identity = convertCheckpointIdentityFrom(src.Spec.Identity)
	dst.Spec.Job = DynamoCheckpointJobConfig{
		PodTemplateSpec:         src.Spec.Job.PodTemplateSpec,
		ActiveDeadlineSeconds:   src.Spec.Job.ActiveDeadlineSeconds,
		BackoffLimit:            src.Spec.Job.BackoffLimit,
		TTLSecondsAfterFinished: src.Spec.Job.TTLSecondsAfterFinished,
	}

	// Status
	dst.Status.Phase = DynamoCheckpointPhase(src.Status.Phase)
	dst.Status.IdentityHash = src.Status.IdentityHash
	dst.Status.Location = src.Status.Location
	dst.Status.StorageType = DynamoCheckpointStorageType(src.Status.StorageType)
	dst.Status.JobName = src.Status.JobName
	dst.Status.CreatedAt = src.Status.CreatedAt
	dst.Status.Message = src.Status.Message
	dst.Status.Conditions = src.Status.Conditions

	return nil
}

func convertCheckpointIdentityTo(src DynamoCheckpointIdentity) v1beta1.DynamoCheckpointIdentity {
	return v1beta1.DynamoCheckpointIdentity{
		Model:                src.Model,
		BackendFramework:     src.BackendFramework,
		DynamoVersion:        src.DynamoVersion,
		TensorParallelSize:   src.TensorParallelSize,
		PipelineParallelSize: src.PipelineParallelSize,
		Dtype:                src.Dtype,
		MaxModelLen:          src.MaxModelLen,
		ExtraParameters:      src.ExtraParameters,
	}
}

func convertCheckpointIdentityFrom(src v1beta1.DynamoCheckpointIdentity) DynamoCheckpointIdentity {
	return DynamoCheckpointIdentity{
		Model:                src.Model,
		BackendFramework:     src.BackendFramework,
		DynamoVersion:        src.DynamoVersion,
		TensorParallelSize:   src.TensorParallelSize,
		PipelineParallelSize: src.PipelineParallelSize,
		Dtype:                src.Dtype,
		MaxModelLen:          src.MaxModelLen,
		ExtraParameters:      src.ExtraParameters,
	}
}

// ---------- DynamoComponentDeployment ----------

// ConvertTo converts this DynamoComponentDeployment (v1alpha1) to the Hub version (v1beta1).
func (src *DynamoComponentDeployment) ConvertTo(dstRaw conversion.Hub) error {
	dst, ok := dstRaw.(*v1beta1.DynamoComponentDeployment)
	if !ok {
		return fmt.Errorf("expected *v1beta1.DynamoComponentDeployment but got %T", dstRaw)
	}

	dst.ObjectMeta = src.ObjectMeta

	// Spec
	dst.Spec.BackendFramework = src.Spec.BackendFramework
	convertSharedSpecTo(&src.Spec.DynamoComponentDeploymentSharedSpec, &dst.Spec.DynamoComponentDeploymentSharedSpec)

	// Status
	dst.Status.ObservedGeneration = src.Status.ObservedGeneration
	dst.Status.Conditions = src.Status.Conditions
	dst.Status.PodSelector = src.Status.PodSelector
	if src.Status.Service != nil {
		svc := convertServiceReplicaStatusTo(*src.Status.Service)
		dst.Status.Service = &svc
	}

	return nil
}

// ConvertFrom converts from the Hub version (v1beta1) to this DynamoComponentDeployment (v1alpha1).
func (dst *DynamoComponentDeployment) ConvertFrom(srcRaw conversion.Hub) error {
	src, ok := srcRaw.(*v1beta1.DynamoComponentDeployment)
	if !ok {
		return fmt.Errorf("expected *v1beta1.DynamoComponentDeployment but got %T", srcRaw)
	}

	dst.ObjectMeta = src.ObjectMeta

	// Spec
	dst.Spec.BackendFramework = src.Spec.BackendFramework
	convertSharedSpecFrom(&src.Spec.DynamoComponentDeploymentSharedSpec, &dst.Spec.DynamoComponentDeploymentSharedSpec)

	// Status
	dst.Status.ObservedGeneration = src.Status.ObservedGeneration
	dst.Status.Conditions = src.Status.Conditions
	dst.Status.PodSelector = src.Status.PodSelector
	if src.Status.Service != nil {
		svc := convertServiceReplicaStatusFrom(*src.Status.Service)
		dst.Status.Service = &svc
	}

	return nil
}

// ---------- DynamoGraphDeployment ----------

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

// ---------- DynamoGraphDeploymentScalingAdapter ----------

// ConvertTo converts this DynamoGraphDeploymentScalingAdapter (v1alpha1) to the Hub version (v1beta1).
func (src *DynamoGraphDeploymentScalingAdapter) ConvertTo(dstRaw conversion.Hub) error {
	dst, ok := dstRaw.(*v1beta1.DynamoGraphDeploymentScalingAdapter)
	if !ok {
		return fmt.Errorf("expected *v1beta1.DynamoGraphDeploymentScalingAdapter but got %T", dstRaw)
	}

	dst.ObjectMeta = src.ObjectMeta

	// Spec
	dst.Spec.Replicas = src.Spec.Replicas
	dst.Spec.DGDRef = v1beta1.DynamoGraphDeploymentServiceRef{
		Name:        src.Spec.DGDRef.Name,
		ServiceName: src.Spec.DGDRef.ServiceName,
	}

	// Status
	dst.Status.Replicas = src.Status.Replicas
	dst.Status.Selector = src.Status.Selector
	dst.Status.LastScaleTime = src.Status.LastScaleTime

	return nil
}

// ConvertFrom converts from the Hub version (v1beta1) to this DynamoGraphDeploymentScalingAdapter (v1alpha1).
func (dst *DynamoGraphDeploymentScalingAdapter) ConvertFrom(srcRaw conversion.Hub) error {
	src, ok := srcRaw.(*v1beta1.DynamoGraphDeploymentScalingAdapter)
	if !ok {
		return fmt.Errorf("expected *v1beta1.DynamoGraphDeploymentScalingAdapter but got %T", srcRaw)
	}

	dst.ObjectMeta = src.ObjectMeta

	// Spec
	dst.Spec.Replicas = src.Spec.Replicas
	dst.Spec.DGDRef = DynamoGraphDeploymentServiceRef{
		Name:        src.Spec.DGDRef.Name,
		ServiceName: src.Spec.DGDRef.ServiceName,
	}

	// Status
	dst.Status.Replicas = src.Status.Replicas
	dst.Status.Selector = src.Status.Selector
	dst.Status.LastScaleTime = src.Status.LastScaleTime

	return nil
}

// ---------- DynamoModel ----------

// ConvertTo converts this DynamoModel (v1alpha1) to the Hub version (v1beta1).
func (src *DynamoModel) ConvertTo(dstRaw conversion.Hub) error {
	dst, ok := dstRaw.(*v1beta1.DynamoModel)
	if !ok {
		return fmt.Errorf("expected *v1beta1.DynamoModel but got %T", dstRaw)
	}

	dst.ObjectMeta = src.ObjectMeta

	// Spec
	dst.Spec.ModelName = src.Spec.ModelName
	dst.Spec.BaseModelName = src.Spec.BaseModelName
	dst.Spec.ModelType = src.Spec.ModelType
	if src.Spec.Source != nil {
		dst.Spec.Source = &v1beta1.ModelSource{
			URI: src.Spec.Source.URI,
		}
	}

	// Status
	dst.Status.ReadyEndpoints = src.Status.ReadyEndpoints
	dst.Status.TotalEndpoints = src.Status.TotalEndpoints
	dst.Status.Conditions = src.Status.Conditions
	if src.Status.Endpoints != nil {
		dst.Status.Endpoints = make([]v1beta1.EndpointInfo, len(src.Status.Endpoints))
		for i, ep := range src.Status.Endpoints {
			dst.Status.Endpoints[i] = v1beta1.EndpointInfo{
				Address: ep.Address,
				PodName: ep.PodName,
				Ready:   ep.Ready,
			}
		}
	}

	return nil
}

// ConvertFrom converts from the Hub version (v1beta1) to this DynamoModel (v1alpha1).
func (dst *DynamoModel) ConvertFrom(srcRaw conversion.Hub) error {
	src, ok := srcRaw.(*v1beta1.DynamoModel)
	if !ok {
		return fmt.Errorf("expected *v1beta1.DynamoModel but got %T", srcRaw)
	}

	dst.ObjectMeta = src.ObjectMeta

	// Spec
	dst.Spec.ModelName = src.Spec.ModelName
	dst.Spec.BaseModelName = src.Spec.BaseModelName
	dst.Spec.ModelType = src.Spec.ModelType
	if src.Spec.Source != nil {
		dst.Spec.Source = &ModelSource{
			URI: src.Spec.Source.URI,
		}
	}

	// Status
	dst.Status.ReadyEndpoints = src.Status.ReadyEndpoints
	dst.Status.TotalEndpoints = src.Status.TotalEndpoints
	dst.Status.Conditions = src.Status.Conditions
	if src.Status.Endpoints != nil {
		dst.Status.Endpoints = make([]EndpointInfo, len(src.Status.Endpoints))
		for i, ep := range src.Status.Endpoints {
			dst.Status.Endpoints[i] = EndpointInfo{
				Address: ep.Address,
				PodName: ep.PodName,
				Ready:   ep.Ready,
			}
		}
	}

	return nil
}

// ---------- Shared helpers ----------

func convertSharedSpecTo(src *DynamoComponentDeploymentSharedSpec, dst *v1beta1.DynamoComponentDeploymentSharedSpec) {
	dst.Annotations = src.Annotations
	dst.Labels = src.Labels
	dst.ServiceName = src.ServiceName
	dst.ComponentType = src.ComponentType
	dst.SubComponentType = src.SubComponentType
	dst.DynamoNamespace = src.DynamoNamespace
	dst.GlobalDynamoNamespace = src.GlobalDynamoNamespace
	dst.Envs = src.Envs
	dst.EnvFromSecret = src.EnvFromSecret
	dst.LivenessProbe = src.LivenessProbe
	dst.ReadinessProbe = src.ReadinessProbe
	dst.Replicas = src.Replicas

	if src.Resources != nil {
		dst.Resources = &v1beta1.Resources{
			Claims: src.Resources.Claims,
		}
		if src.Resources.Requests != nil {
			dst.Resources.Requests = &v1beta1.ResourceItem{
				CPU:     src.Resources.Requests.CPU,
				Memory:  src.Resources.Requests.Memory,
				GPU:     src.Resources.Requests.GPU,
				GPUType: src.Resources.Requests.GPUType,
				Custom:  src.Resources.Requests.Custom,
			}
		}
		if src.Resources.Limits != nil {
			dst.Resources.Limits = &v1beta1.ResourceItem{
				CPU:     src.Resources.Limits.CPU,
				Memory:  src.Resources.Limits.Memory,
				GPU:     src.Resources.Limits.GPU,
				GPUType: src.Resources.Limits.GPUType,
				Custom:  src.Resources.Limits.Custom,
			}
		}
	}

	if src.Autoscaling != nil {
		dst.Autoscaling = &v1beta1.Autoscaling{
			Enabled:     src.Autoscaling.Enabled,
			MinReplicas: src.Autoscaling.MinReplicas,
			MaxReplicas: src.Autoscaling.MaxReplicas,
			Behavior:    src.Autoscaling.Behavior,
			Metrics:     src.Autoscaling.Metrics,
		}
	}

	if src.VolumeMounts != nil {
		dst.VolumeMounts = make([]v1beta1.VolumeMount, len(src.VolumeMounts))
		for i, vm := range src.VolumeMounts {
			dst.VolumeMounts[i] = v1beta1.VolumeMount{
				Name:                  vm.Name,
				MountPoint:            vm.MountPoint,
				UseAsCompilationCache: vm.UseAsCompilationCache,
			}
		}
	}

	if src.Ingress != nil {
		dst.Ingress = &v1beta1.IngressSpec{
			Enabled:                    src.Ingress.Enabled,
			Host:                       src.Ingress.Host,
			UseVirtualService:          src.Ingress.UseVirtualService,
			VirtualServiceGateway:      src.Ingress.VirtualServiceGateway,
			HostPrefix:                 src.Ingress.HostPrefix,
			Annotations:                src.Ingress.Annotations,
			Labels:                     src.Ingress.Labels,
			HostSuffix:                 src.Ingress.HostSuffix,
			IngressControllerClassName: src.Ingress.IngressControllerClassName,
		}
		if src.Ingress.TLS != nil {
			dst.Ingress.TLS = &v1beta1.IngressTLSSpec{
				SecretName: src.Ingress.TLS.SecretName,
			}
		}
	}

	if src.ModelRef != nil {
		dst.ModelRef = &v1beta1.ModelReference{
			Name:     src.ModelRef.Name,
			Revision: src.ModelRef.Revision,
		}
	}

	if src.SharedMemory != nil {
		dst.SharedMemory = &v1beta1.SharedMemorySpec{
			Disabled: src.SharedMemory.Disabled,
			Size:     src.SharedMemory.Size,
		}
	}

	if src.ExtraPodMetadata != nil {
		dst.ExtraPodMetadata = &v1beta1.ExtraPodMetadata{
			Annotations: src.ExtraPodMetadata.Annotations,
			Labels:      src.ExtraPodMetadata.Labels,
		}
	}

	if src.ExtraPodSpec != nil {
		dst.ExtraPodSpec = &v1beta1.ExtraPodSpec{
			PodSpec:       src.ExtraPodSpec.PodSpec,
			MainContainer: src.ExtraPodSpec.MainContainer,
		}
	}

	if src.Multinode != nil {
		dst.Multinode = &v1beta1.MultinodeSpec{
			NodeCount: src.Multinode.NodeCount,
		}
	}

	if src.ScalingAdapter != nil {
		dst.ScalingAdapter = &v1beta1.ScalingAdapter{
			Enabled: src.ScalingAdapter.Enabled,
		}
	}

	if src.EPPConfig != nil {
		dst.EPPConfig = &v1beta1.EPPConfig{
			ConfigMapRef: src.EPPConfig.ConfigMapRef,
			Config:       src.EPPConfig.Config,
		}
	}

	if src.Checkpoint != nil {
		dst.Checkpoint = &v1beta1.ServiceCheckpointConfig{
			Enabled:       src.Checkpoint.Enabled,
			Mode:          v1beta1.CheckpointMode(src.Checkpoint.Mode),
			CheckpointRef: src.Checkpoint.CheckpointRef,
		}
		if src.Checkpoint.Identity != nil {
			identity := convertCheckpointIdentityTo(*src.Checkpoint.Identity)
			dst.Checkpoint.Identity = &identity
		}
	}
}

func convertSharedSpecFrom(src *v1beta1.DynamoComponentDeploymentSharedSpec, dst *DynamoComponentDeploymentSharedSpec) {
	dst.Annotations = src.Annotations
	dst.Labels = src.Labels
	dst.ServiceName = src.ServiceName
	dst.ComponentType = src.ComponentType
	dst.SubComponentType = src.SubComponentType
	dst.DynamoNamespace = src.DynamoNamespace
	dst.GlobalDynamoNamespace = src.GlobalDynamoNamespace
	dst.Envs = src.Envs
	dst.EnvFromSecret = src.EnvFromSecret
	dst.LivenessProbe = src.LivenessProbe
	dst.ReadinessProbe = src.ReadinessProbe
	dst.Replicas = src.Replicas

	if src.Resources != nil {
		dst.Resources = &Resources{
			Claims: src.Resources.Claims,
		}
		if src.Resources.Requests != nil {
			dst.Resources.Requests = &ResourceItem{
				CPU:     src.Resources.Requests.CPU,
				Memory:  src.Resources.Requests.Memory,
				GPU:     src.Resources.Requests.GPU,
				GPUType: src.Resources.Requests.GPUType,
				Custom:  src.Resources.Requests.Custom,
			}
		}
		if src.Resources.Limits != nil {
			dst.Resources.Limits = &ResourceItem{
				CPU:     src.Resources.Limits.CPU,
				Memory:  src.Resources.Limits.Memory,
				GPU:     src.Resources.Limits.GPU,
				GPUType: src.Resources.Limits.GPUType,
				Custom:  src.Resources.Limits.Custom,
			}
		}
	}

	if src.Autoscaling != nil {
		dst.Autoscaling = &Autoscaling{
			Enabled:     src.Autoscaling.Enabled,
			MinReplicas: src.Autoscaling.MinReplicas,
			MaxReplicas: src.Autoscaling.MaxReplicas,
			Behavior:    src.Autoscaling.Behavior,
			Metrics:     src.Autoscaling.Metrics,
		}
	}

	if src.VolumeMounts != nil {
		dst.VolumeMounts = make([]VolumeMount, len(src.VolumeMounts))
		for i, vm := range src.VolumeMounts {
			dst.VolumeMounts[i] = VolumeMount{
				Name:                  vm.Name,
				MountPoint:            vm.MountPoint,
				UseAsCompilationCache: vm.UseAsCompilationCache,
			}
		}
	}

	if src.Ingress != nil {
		dst.Ingress = &IngressSpec{
			Enabled:                    src.Ingress.Enabled,
			Host:                       src.Ingress.Host,
			UseVirtualService:          src.Ingress.UseVirtualService,
			VirtualServiceGateway:      src.Ingress.VirtualServiceGateway,
			HostPrefix:                 src.Ingress.HostPrefix,
			Annotations:                src.Ingress.Annotations,
			Labels:                     src.Ingress.Labels,
			HostSuffix:                 src.Ingress.HostSuffix,
			IngressControllerClassName: src.Ingress.IngressControllerClassName,
		}
		if src.Ingress.TLS != nil {
			dst.Ingress.TLS = &IngressTLSSpec{
				SecretName: src.Ingress.TLS.SecretName,
			}
		}
	}

	if src.ModelRef != nil {
		dst.ModelRef = &ModelReference{
			Name:     src.ModelRef.Name,
			Revision: src.ModelRef.Revision,
		}
	}

	if src.SharedMemory != nil {
		dst.SharedMemory = &SharedMemorySpec{
			Disabled: src.SharedMemory.Disabled,
			Size:     src.SharedMemory.Size,
		}
	}

	if src.ExtraPodMetadata != nil {
		dst.ExtraPodMetadata = &ExtraPodMetadata{
			Annotations: src.ExtraPodMetadata.Annotations,
			Labels:      src.ExtraPodMetadata.Labels,
		}
	}

	if src.ExtraPodSpec != nil {
		dst.ExtraPodSpec = &ExtraPodSpec{
			PodSpec:       src.ExtraPodSpec.PodSpec,
			MainContainer: src.ExtraPodSpec.MainContainer,
		}
	}

	if src.Multinode != nil {
		dst.Multinode = &MultinodeSpec{
			NodeCount: src.Multinode.NodeCount,
		}
	}

	if src.ScalingAdapter != nil {
		dst.ScalingAdapter = &ScalingAdapter{
			Enabled: src.ScalingAdapter.Enabled,
		}
	}

	if src.EPPConfig != nil {
		dst.EPPConfig = &EPPConfig{
			ConfigMapRef: src.EPPConfig.ConfigMapRef,
			Config:       src.EPPConfig.Config,
		}
	}

	if src.Checkpoint != nil {
		dst.Checkpoint = &ServiceCheckpointConfig{
			Enabled:       src.Checkpoint.Enabled,
			Mode:          CheckpointMode(src.Checkpoint.Mode),
			CheckpointRef: src.Checkpoint.CheckpointRef,
		}
		if src.Checkpoint.Identity != nil {
			identity := convertCheckpointIdentityFrom(*src.Checkpoint.Identity)
			dst.Checkpoint.Identity = &identity
		}
	}
}

func convertServiceReplicaStatusTo(src ServiceReplicaStatus) v1beta1.ServiceReplicaStatus {
	return v1beta1.ServiceReplicaStatus{
		ComponentKind:     v1beta1.ComponentKind(src.ComponentKind),
		ComponentName:     src.ComponentName,
		Replicas:          src.Replicas,
		UpdatedReplicas:   src.UpdatedReplicas,
		ReadyReplicas:     src.ReadyReplicas,
		AvailableReplicas: src.AvailableReplicas,
	}
}

func convertServiceReplicaStatusFrom(src v1beta1.ServiceReplicaStatus) ServiceReplicaStatus {
	return ServiceReplicaStatus{
		ComponentKind:     ComponentKind(src.ComponentKind),
		ComponentName:     src.ComponentName,
		Replicas:          src.Replicas,
		UpdatedReplicas:   src.UpdatedReplicas,
		ReadyReplicas:     src.ReadyReplicas,
		AvailableReplicas: src.AvailableReplicas,
	}
}
