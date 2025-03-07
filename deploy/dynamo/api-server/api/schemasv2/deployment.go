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

package schemasv2

import "github.com/dynemo-ai/dynemo/deploy/dynamo/api-server/api/schemas"

type DeploymentSchema struct {
	schemas.ResourceSchema
	Creator        *schemas.UserSchema               `json:"creator"`
	Cluster        *ClusterSchema                    `json:"cluster"`
	Status         schemas.DeploymentStatus          `json:"status" enum:"unknown,non-deployed,running,unhealthy,failed,deploying"`
	URLs           []string                          `json:"urls"`
	LatestRevision *schemas.DeploymentRevisionSchema `json:"latest_revision"`
	KubeNamespace  string                            `json:"kube_namespace"`
}

type GetDeploymentSchema struct {
	DeploymentName string `uri:"deploymentName" binding:"required"`
}

func (s *GetDeploymentSchema) ToV1(clusterName string, namespace string) *schemas.GetDeploymentSchema {
	return &schemas.GetDeploymentSchema{
		GetClusterSchema: schemas.GetClusterSchema{
			ClusterName: clusterName,
		},
		KubeNamespace:  namespace,
		DeploymentName: s.DeploymentName,
	}
}

type CreateDeploymentSchema struct {
	UpdateDeploymentSchema
	Name string `json:"name"`
}

type UpdateDeploymentSchema struct {
	DeploymentConfigSchema
	CompoundNim string `json:"bento"`
}

type DeploymentConfigSchema struct {
	AccessAuthorization bool                   `json:"access_authorization"`
	Envs                interface{}            `json:"envs,omitempty"`
	Secrets             interface{}            `json:"secrets,omitempty"`
	Services            map[string]ServiceSpec `json:"services"`
}

type ServiceSpec struct {
	Scaling          ScalingSpec                        `json:"scaling"`
	ConfigOverrides  ConfigOverridesSpec                `json:"config_overrides"`
	ExternalServices map[string]schemas.ExternalService `json:"external_services,omitempty"`
	ColdStartTimeout *int32                             `json:"cold_start_timeout,omitempty"`
}

type ScalingSpec struct {
	MinReplicas int `json:"min_replicas"`
	MaxReplicas int `json:"max_replicas"`
}

type ConfigOverridesSpec struct {
	Resources schemas.Resources `json:"resources"`
}
