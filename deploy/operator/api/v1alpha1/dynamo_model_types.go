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
	"strings"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
)

// DynamoModelSpec defines the desired state of DynamoModel
type DynamoModelSpec struct {
	// ModelName is the full model identifier (e.g., "meta-llama/Llama-3.3-70B-Instruct-lora")
	// +kubebuilder:validation:Required
	ModelName string `json:"modelName"`

	// BaseModelName is the base model identifier that matches the service label
	// This is used to discover endpoints via headless services
	// +kubebuilder:validation:Required
	BaseModelName string `json:"baseModelName"`

	// ModelType specifies the type of model (e.g., "base", "lora", "adapter")
	// +kubebuilder:validation:Enum=base;lora;adapter
	// +kubebuilder:default=base
	// +optional
	ModelType string `json:"modelType,omitempty"`

	// Source specifies the model source location (only applicable for lora model type)
	// +optional
	Source *ModelSource `json:"source,omitempty"`
}

// ModelSource defines the source location of a model
type ModelSource struct {
	// URI is the model source URI
	// Supported formats:
	// - S3: s3://bucket/path/to/model
	// - HuggingFace: hf://org/model@revision_sha
	// +kubebuilder:validation:Required
	URI string `json:"uri"`
}

// EndpointInfo represents a single endpoint (pod) serving the model
type EndpointInfo struct {
	// Address is the full address of the endpoint (e.g., "http://10.0.1.5:9090")
	Address string `json:"address"`

	// PodName is the name of the pod serving this endpoint
	// +optional
	PodName string `json:"podName,omitempty"`

	// Ready indicates whether the endpoint is ready to serve traffic
	// For LoRA models: true if the POST /loras request succeeded with a 2xx status code
	// For base models: always false (no probing performed)
	Ready bool `json:"ready"`
}

// DynamoModelStatus defines the observed state of DynamoModel
type DynamoModelStatus struct {
	// Endpoints is the current list of all endpoints for this model
	// +optional
	Endpoints []EndpointInfo `json:"endpoints,omitempty"`

	// ReadyEndpoints is the count of endpoints that are ready
	ReadyEndpoints int `json:"readyEndpoints"`

	// TotalEndpoints is the total count of endpoints
	TotalEndpoints int `json:"totalEndpoints"`

	// Conditions represents the latest available observations of the model's state
	// +optional
	Conditions []metav1.Condition `json:"conditions,omitempty"`
}

// +genclient
// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:storageversion
// +kubebuilder:printcolumn:name="BaseModel",type="string",JSONPath=".spec.baseModelName",description="Base model name"
// +kubebuilder:printcolumn:name="Type",type="string",JSONPath=".spec.modelType",description="Model type"
// +kubebuilder:printcolumn:name="Ready",type="integer",JSONPath=".status.readyEndpoints",description="Ready endpoints"
// +kubebuilder:printcolumn:name="Total",type="integer",JSONPath=".status.totalEndpoints",description="Total endpoints"
// +kubebuilder:printcolumn:name="Age",type="date",JSONPath=".metadata.creationTimestamp"
// +kubebuilder:resource:shortName=dm
// DynamoModel is the Schema for the dynamo models API
type DynamoModel struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   DynamoModelSpec   `json:"spec,omitempty"`
	Status DynamoModelStatus `json:"status,omitempty"`
}

// +kubebuilder:object:root=true

// DynamoModelList contains a list of DynamoModel
type DynamoModelList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []DynamoModel `json:"items"`
}

func init() {
	SchemeBuilder.Register(&DynamoModel{}, &DynamoModelList{})
}

// IsLoRA returns true if this is a LoRA model (case-insensitive)
func (m *DynamoModel) IsLoRA() bool {
	return strings.EqualFold(m.Spec.ModelType, "lora")
}

// GetReadyEndpoints returns only the endpoints that are ready
func (m *DynamoModel) GetReadyEndpoints() []EndpointInfo {
	var ready []EndpointInfo
	for _, ep := range m.Status.Endpoints {
		if ep.Ready {
			ready = append(ready, ep)
		}
	}
	return ready
}

// HasEndpoints returns true if the model has any endpoints
func (m *DynamoModel) HasEndpoints() bool {
	return len(m.Status.Endpoints) > 0
}

// HasReadyEndpoints returns true if the model has any ready endpoints
func (m *DynamoModel) HasReadyEndpoints() bool {
	return m.Status.ReadyEndpoints > 0
}

// IsReady returns true if all endpoints are ready
func (m *DynamoModel) IsReady() (bool, string) {
	if m.Status.TotalEndpoints == 0 {
		return false, "No endpoints configured"
	}
	if m.Status.ReadyEndpoints == 0 {
		return false, "No endpoints ready"
	}
	if m.Status.ReadyEndpoints < m.Status.TotalEndpoints {
		return false, fmt.Sprintf("Only %d/%d endpoints ready", m.Status.ReadyEndpoints, m.Status.TotalEndpoints)
	}
	return true, ""
}

// GetState returns "ready" or "not_ready" based on endpoint status
func (m *DynamoModel) GetState() string {
	ready, _ := m.IsReady()
	if ready {
		return consts.ResourceStateReady
	}
	return consts.ResourceStateNotReady
}
