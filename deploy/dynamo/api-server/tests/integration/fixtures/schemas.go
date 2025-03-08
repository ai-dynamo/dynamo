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

package fixtures

import "github.com/dynemo-ai/dynemo/deploy/dynamo/api-server/api/schemas"

func DefaultCreateClusterSchema() schemas.CreateClusterSchema {
	return schemas.CreateClusterSchema{
		Description: "description",
		KubeConfig:  "",
		Name:        "default",
	}
}

func DefaultUpdateClusterSchema() schemas.UpdateClusterSchema {
	d := "description"
	kc := "kubeconfig"
	return schemas.UpdateClusterSchema{
		Description: &d,
		KubeConfig:  &kc,
	}
}

func DefaultListQuerySchema() schemas.ListQuerySchema {
	return schemas.ListQuerySchema{
		Start:  0,
		Count:  20,
		Search: nil,
	}
}

// DefaultCreateDeploymentSchema generates a default CreateDeploymentSchema
func DefaultCreateDeploymentSchema() schemas.CreateDeploymentSchema {
	return schemas.CreateDeploymentSchema{
		Name:                   "default-deployment",
		KubeNamespace:          "default-namespace",
		UpdateDeploymentSchema: DefaultUpdateDeploymentSchema(),
	}
}

// DefaultUpdateDeploymentSchema generates a default UpdateDeploymentSchema
func DefaultUpdateDeploymentSchema() schemas.UpdateDeploymentSchema {
	description := "default deployment"
	return schemas.UpdateDeploymentSchema{
		Targets: []*schemas.CreateDeploymentTargetSchema{
			DefaultCreateDeploymentTargetSchema(),
		},
		Description: &description,
		DoNotDeploy: false,
	}
}

// DefaultCreateDeploymentTargetSchema generates a default CreateDeploymentTargetSchema
func DefaultCreateDeploymentTargetSchema() *schemas.CreateDeploymentTargetSchema {
	return &schemas.CreateDeploymentTargetSchema{
		DynamoNim: "default-dynamo-nim",
		Version:   "default-version",
		Config:    DefaultDeploymentTargetConfig(),
	}
}

// DefaultDeploymentTargetConfig generates a default DeploymentTargetConfig
func DefaultDeploymentTargetConfig() *schemas.DeploymentTargetConfig {
	return &schemas.DeploymentTargetConfig{
		KubeResourceUid:     "default-uid",
		KubeResourceVersion: "v1",
		Resources:           DefaultResources(),
		HPAConf:             DefaultDeploymentTargetHPAConf(),
		DeploymentStrategy:  DefaultDeploymentStrategy(),
	}
}

// DefaultResources generates a default Resources struct
func DefaultResources() *schemas.Resources {
	return &schemas.Resources{
		Requests: &schemas.ResourceItem{
			CPU:    "500m",
			Memory: "1Gi",
		},
		Limits: &schemas.ResourceItem{
			CPU:    "1",
			Memory: "2Gi",
		},
	}
}

// DefaultDeploymentTargetHPAConf generates a default DeploymentTargetHPAConf
func DefaultDeploymentTargetHPAConf() *schemas.DeploymentTargetHPAConf {
	qps := int64(1000)
	return &schemas.DeploymentTargetHPAConf{
		CPU:         nil,
		GPU:         nil,
		Memory:      nil,
		QPS:         &qps,
		MinReplicas: int32Ptr(1),
		MaxReplicas: int32Ptr(5),
	}
}

// DefaultDeploymentStrategy generates a default DeploymentStrategy
func DefaultDeploymentStrategy() *schemas.DeploymentStrategy {
	strategy := schemas.DeploymentStrategyRollingUpdate
	return &strategy
}

// Helper function to return a pointer to an int32
func int32Ptr(i int32) *int32 {
	return &i
}
