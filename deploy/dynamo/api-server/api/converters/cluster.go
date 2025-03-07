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

package converters

import (
	"github.com/dynemo-ai/dynemo/deploy/dynamo/api-server/api/mocks"
	"github.com/dynemo-ai/dynemo/deploy/dynamo/api-server/api/models"
	"github.com/dynemo-ai/dynemo/deploy/dynamo/api-server/api/schemas"
	"github.com/dynemo-ai/dynemo/deploy/dynamo/api-server/api/schemasv2"
)

func ToClusterSchemaList(clusters []*models.Cluster) []*schemas.ClusterSchema {
	clusterSchemas := make([]*schemas.ClusterSchema, 0)

	for _, cluster := range clusters {
		clusterSchemas = append(clusterSchemas, ToClusterSchema(cluster))
	}

	return clusterSchemas
}

func ToClusterSchema(cluster *models.Cluster) *schemas.ClusterSchema {
	return &schemas.ClusterSchema{
		Creator:     mocks.DefaultUser(),
		Description: cluster.Description,
		ResourceSchema: schemas.ResourceSchema{
			Name:         cluster.Name,
			ResourceType: schemas.ResourceTypeCluster,
			BaseSchema: schemas.BaseSchema{
				Uid:       cluster.GetUid(),
				CreatedAt: cluster.CreatedAt,
				UpdatedAt: cluster.UpdatedAt,
				DeletedAt: &cluster.DeletedAt.Time,
			},
		},
	}
}

func ToClusterFullSchema(cluster *models.Cluster) *schemas.ClusterFullSchema {
	clusterSchema := ToClusterSchema(cluster)

	return &schemas.ClusterFullSchema{
		ClusterSchema: *clusterSchema,
		KubeConfig:    &cluster.KubeConfig,
		Organization:  mocks.DefaultOrg(),
	}
}

func ToClusterSchemaV2(cluster *models.Cluster, creator *schemas.UserSchema) *schemasv2.ClusterSchema {
	return &schemasv2.ClusterSchema{
		Description:      cluster.Description,
		OrganizationName: "nvidia",
		Creator:          creator,
		ResourceSchema: schemas.ResourceSchema{
			Name:   cluster.Name,
			Labels: []schemas.LabelItemSchema{},
			BaseSchema: schemas.BaseSchema{
				Uid:       cluster.Resource.BaseModel.GetUid(),
				CreatedAt: cluster.Resource.BaseModel.CreatedAt,
				UpdatedAt: cluster.Resource.BaseModel.UpdatedAt,
				DeletedAt: nil, // Can assume that this is nil during creation
			},
		},
	}
}
