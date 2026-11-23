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

// Package modelservice renders worker system-API Services for model discovery.
package modelservice

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"maps"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/validation"
)

// Params supplies resolved model identity, graph scope, and authored annotations.
type Params struct {
	Namespace    string
	ModelName    string
	GraphName    string // empty for legacy shared model Services
	Annotations  map[string]string
	ReadableName bool // selects readable naming for an eligible graph
}

// Render constructs a headless Service without mutating its inputs.
func Render(params Params) *corev1.Service {
	// Model identity remains the discovery contract across all naming modes.
	modelHash := HashModelName(params.ModelName)
	labels := map[string]string{
		consts.KubeLabelDynamoBaseModelHash: modelHash,
		"nvidia.com/managed-by":             "dynamo-operator",
	}
	selector := map[string]string{consts.KubeLabelDynamoBaseModelHash: modelHash}
	if params.GraphName != "" {
		labels[consts.KubeLabelDynamoGraphDeploymentName] = params.GraphName
		selector[consts.KubeLabelDynamoGraphDeploymentName] = params.GraphName
	}

	// Canonical model identity takes precedence without aliasing authored metadata.
	annotations := make(map[string]string, len(params.Annotations)+1)
	maps.Copy(annotations, params.Annotations)
	annotations[consts.KubeAnnotationDynamoBaseModel] = params.ModelName

	return &corev1.Service{
		Name:        Name(params.ModelName, params.GraphName, params.ReadableName),
		Namespace:   params.Namespace,
		Labels:      labels,
		Annotations: annotations,
		Spec: corev1.ServiceSpec{
			ClusterIP:                corev1.ClusterIPNone,
			Selector:                 selector,
			PublishNotReadyAddresses: false,
			Ports: []corev1.ServicePort{{
				Name:       consts.DynamoSystemPortName,
				Port:       consts.DynamoSystemPort,
				TargetPort: intstr.FromInt32(consts.DynamoSystemPort),
				Protocol:   corev1.ProtocolTCP,
			}},
		},
	}
}

// Name preserves legacy naming or derives a readable, graph-scoped Service name.
// An empty graphName retains the legacy shared model name.
func Name(modelName, graphName string, readableName bool) string {
	// Legacy shared names intentionally do not depend on a graph identity.
	modelHash := HashModelName(modelName)
	if graphName == "" {
		return fmt.Sprintf("dynamo-model-%s", modelHash)
	}

	// Retain exact graph identity rather than normalizing distinct names together.
	if readableName {
		readable := fmt.Sprintf("%s-model-%s", graphName, modelHash)
		if len(validation.IsDNS1035Label(readable)) == 0 {
			return readable
		}
	}
	return fmt.Sprintf("dynamo-model-%s-%s", modelHash, HashModelName(graphName))
}

// HashModelName preserves the eight-character model discovery hash.
func HashModelName(modelName string) string {
	hash := sha256.Sum256([]byte(modelName))
	return hex.EncodeToString(hash[:])[:8]
}
