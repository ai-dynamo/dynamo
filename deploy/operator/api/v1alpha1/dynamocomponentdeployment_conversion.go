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
