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

package crds

import (
	"github.com/dynemo-ai/dynemo/deploy/dynamo/api-server/api/schemas"
	autoscalingv2beta2 "k8s.io/api/autoscaling/v2beta2"
	corev1 "k8s.io/api/core/v1"
)

type Autoscaling struct {
	MinReplicas int32                                               `json:"minReplicas"`
	MaxReplicas int32                                               `json:"maxReplicas"`
	Metrics     []autoscalingv2beta2.MetricSpec                     `json:"metrics,omitempty"`
	Behavior    *autoscalingv2beta2.HorizontalPodAutoscalerBehavior `json:"behavior,omitempty"`
}

type DynamoNimVersionDeploymentIngressTLSSpec struct {
	SecretName string `json:"secretName,omitempty"`
}

type DynamoNimVersionDeploymentIngressSpec struct {
	Enabled     bool                                      `json:"enabled,omitempty"`
	Annotations map[string]string                         `json:"annotations,omitempty"`
	Labels      map[string]string                         `json:"labels,omitempty"`
	TLS         *DynamoNimVersionDeploymentIngressTLSSpec `json:"tls,omitempty"`
}

type MonitorExporterMountSpec struct {
	Path                string `json:"path,omitempty"`
	ReadOnly            bool   `json:"readOnly,omitempty"`
	corev1.VolumeSource `json:",inline"`
}

type MonitorExporterSpec struct {
	Enabled          bool                       `json:"enabled,omitempty"`
	Output           string                     `json:"output,omitempty"`
	Options          map[string]string          `json:"options,omitempty"`
	StructureOptions []corev1.EnvVar            `json:"structureOptions,omitempty"`
	Mounts           []MonitorExporterMountSpec `json:"mounts,omitempty"`
}

type DynamoNimDeploymentData struct {
	Annotations map[string]string `json:"annotations,omitempty"`
	Labels      map[string]string `json:"labels,omitempty"`

	DynamoNimVersion string `json:"dynamoNim"`

	Resources        schemas.Resources                  `json:"resources,omitempty"`
	Autoscaling      *Autoscaling                       `json:"autoscaling,omitempty"`
	Envs             []corev1.EnvVar                    `json:"envs,omitempty"`
	ExternalServices map[string]schemas.ExternalService `json:"externalServices,omitempty"`

	Ingress DynamoNimVersionDeploymentIngressSpec `json:"ingress,omitempty"`

	MonitorExporter *MonitorExporterSpec `json:"monitorExporter,omitempty"`

	ExtraPodMetadata *ExtraPodMetadata `json:"extraPodMetadata,omitempty"`
	ExtraPodSpec     *ExtraPodSpec     `json:"extraPodSpec,omitempty"`

	LivenessProbe  *corev1.Probe `json:"livenessProbe,omitempty"`
	ReadinessProbe *corev1.Probe `json:"readinessProbe,omitempty"`
}

type DynamoNimDeploymentConfigurationV1Alpha1 struct {
	Data    DynamoNimDeploymentData `json:"data"`
	Version string                  `json:"version"`
}
