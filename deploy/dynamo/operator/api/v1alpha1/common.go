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

type PVC struct {
	// Create indicates to create a new PVC
	Create *bool `json:"create,omitempty"`
	// Name is the name of the PVC
	Name *string `json:"name,omitempty"`
	// StorageClass to be used for PVC creation. Leave it as empty if the PVC is already created.
	StorageClass string `json:"storageClass,omitempty"`
	// Size of the NIM cache in Gi, used during PVC creation
	Size resource.Quantity `json:"size,omitempty"`
	// VolumeAccessMode is the volume access mode of the PVC
	VolumeAccessMode corev1.PersistentVolumeAccessMode `json:"volumeAccessMode,omitempty"`
	MountPoint       *string                           `json:"mountPoint,omitempty"`
}

type Autoscaling struct {
	Enabled     bool                                           `json:"enabled,omitempty"`
	MinReplicas int                                            `json:"minReplicas,omitempty"`
	MaxReplicas int                                            `json:"maxReplicas,omitempty"`
	Behavior    *autoscalingv2.HorizontalPodAutoscalerBehavior `json:"behavior,omitempty"`
	Metrics     []autoscalingv2.MetricSpec                     `json:"metrics,omitempty"`
}
