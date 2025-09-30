/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
	autoscalingv2 "k8s.io/api/autoscaling/v2"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
)

// +kubebuilder:validation:XValidation:rule="!has(self.create) || self.create == false || (has(self.size) && has(self.storageClass) && has(self.volumeAccessMode))",message="When create is true, size, storageClass, and volumeAccessMode are required"
type PVC struct {
	// Create indicates to create a new PVC
	Create *bool `json:"create,omitempty"`
	// Name is the name of the PVC
	// +kubebuilder:validation:Required
	Name *string `json:"name,omitempty"`
	// StorageClass to be used for PVC creation. Required when create is true.
	StorageClass string `json:"storageClass,omitempty"`
	// Size of the volume in Gi, used during PVC creation. Required when create is true.
	Size resource.Quantity `json:"size,omitempty"`
	// VolumeAccessMode is the volume access mode of the PVC. Required when create is true.
	VolumeAccessMode corev1.PersistentVolumeAccessMode `json:"volumeAccessMode,omitempty"`
}

// VolumeMount references a PVC defined at the top level for volumes to be mounted by the component
type VolumeMount struct {
	// Name references a PVC name defined in the top-level PVCs map
	// +kubebuilder:validation:Required
	Name string `json:"name,omitempty"`
	// MountPoint specifies where to mount the volume.
	// If useAsCompilationCache is true and mountPoint is not specified,
	// a backend-specific default will be used.
	MountPoint string `json:"mountPoint,omitempty"`
	// UseAsCompilationCache indicates this volume should be used as a compilation cache.
	// When true, backend-specific environment variables will be set and default mount points may be used.
	// +kubebuilder:default=false
	UseAsCompilationCache bool `json:"useAsCompilationCache,omitempty"`
}

type Autoscaling struct {
	Enabled     bool                                           `json:"enabled,omitempty"`
	MinReplicas int                                            `json:"minReplicas,omitempty"`
	MaxReplicas int                                            `json:"maxReplicas,omitempty"`
	Behavior    *autoscalingv2.HorizontalPodAutoscalerBehavior `json:"behavior,omitempty"`
	Metrics     []autoscalingv2.MetricSpec                     `json:"metrics,omitempty"`
}

type SharedMemorySpec struct {
	Disabled bool              `json:"disabled,omitempty"`
	Size     resource.Quantity `json:"size,omitempty"`
}
