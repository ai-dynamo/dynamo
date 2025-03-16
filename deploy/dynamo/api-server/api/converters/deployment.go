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
	"context"
	"fmt"
	"os"

	"github.com/ai-dynamo/dynamo/deploy/dynamo/api-server/api/mocks"
	"github.com/ai-dynamo/dynamo/deploy/dynamo/api-server/api/models"
	"github.com/ai-dynamo/dynamo/deploy/dynamo/api-server/api/schemas"
	"github.com/ai-dynamo/dynamo/deploy/dynamo/api-server/api/schemasv2"
	"github.com/ai-dynamo/dynamo/deploy/dynamo/api-server/api/services"
)

const kDefaultIngressSuffix = "local"

func ToDeploymentSchema(ctx context.Context, deployment *models.Deployment) (*schemas.DeploymentSchema, error) {
	if deployment == nil {
		return nil, nil
	}
	ss, err := ToDeploymentSchemas(ctx, []*models.Deployment{deployment})
	if err != nil {
		return nil, err
	}
	return ss[0], nil
}

func ToDeploymentSchemas(ctx context.Context, deployments []*models.Deployment) ([]*schemas.DeploymentSchema, error) {
	status_ := schemas.DeploymentRevisionStatusActive
	deploymentIds := make([]uint, 0, len(deployments))

	for _, deployment := range deployments {
		deploymentIds = append(deploymentIds, deployment.ID)
	}

	deploymentRevisions, _, err := services.DeploymentRevisionService.List(ctx, services.ListDeploymentRevisionOption{
		DeploymentIds: &deploymentIds,
		Status:        &status_,
	})
	if err != nil {
		return nil, err
	}

	deploymentIdToDeploymentRevisionUid := make(map[uint]string)
	for _, deploymentRevision := range deploymentRevisions {
		deploymentIdToDeploymentRevisionUid[deploymentRevision.DeploymentId] = deploymentRevision.GetUid()
	}

	deploymentRevisionSchemas, err := ToDeploymentRevisionSchemas(ctx, deploymentRevisions)
	if err != nil {
		return nil, err
	}

	deploymentRevisionSchemasMap := make(map[string]*schemas.DeploymentRevisionSchema)
	for _, deploymentRevisionSchema := range deploymentRevisionSchemas {
		deploymentRevisionSchemasMap[deploymentRevisionSchema.Uid] = deploymentRevisionSchema
	}

	resourceSchemaMap := make(map[string]*schemas.ResourceSchema, len(deployments))
	for _, deployment := range deployments {
		resourceSchemaMap[deployment.GetUid()] = ToResourceSchema(&deployment.Resource, deployment.GetResourceType())
	}

	res := make([]*schemas.DeploymentSchema, 0, len(deployments))
	for _, deployment := range deployments {
		deploymentRevisionUid := deploymentIdToDeploymentRevisionUid[deployment.ID]
		deploymentRevisionSchema := deploymentRevisionSchemasMap[deploymentRevisionUid]

		creatorSchema := mocks.DefaultUser()
		cluster, err := services.ClusterService.Get(ctx, deployment.ClusterId)
		if err != nil {
			return nil, err
		}
		clusterSchema := ToClusterFullSchema(cluster)

		// TODO: Add Implementation for Ingress URLs after UI is integrated

		resourceSchema, ok := resourceSchemaMap[deployment.GetUid()]
		if !ok {
			return nil, fmt.Errorf("resourceSchema not found for deployment %s", deployment.GetUid())
		}
		res = append(res, &schemas.DeploymentSchema{
			ResourceSchema: *resourceSchema,
			Creator:        creatorSchema,
			Cluster:        clusterSchema,
			Status:         deployment.Status,
			LatestRevision: deploymentRevisionSchema,
			KubeNamespace:  deployment.KubeNamespace,
		})
	}
	return res, nil
}

func ToDeploymentSchemaV2(ctx context.Context, cluster *models.Cluster, deployment *models.Deployment, creator *schemas.UserSchema) (*schemasv2.DeploymentSchema, error) {
	clusterSchema := ToClusterSchemaV2(cluster, creator)
	status := schemas.DeploymentRevisionStatusActive

	deploymentRevisionListOpts := services.ListDeploymentRevisionOption{
		DeploymentId: &deployment.ID,
		Status:       &status,
	}

	deploymentRevisions, total, err := services.DeploymentRevisionService.List(ctx, deploymentRevisionListOpts)
	if err != nil {
		return nil, err
	}

	var revision *models.DeploymentRevision
	if total > 0 {
		revision = deploymentRevisions[0]
	}

	revisionSchema, err := ToDeploymentRevisionSchema(ctx, revision)
	if err != nil {
		return nil, err
	}

	urls := make([]string, 0)
	ingressSuffix, found := os.LookupEnv("INGRESS_SUFFIX")
	if !found || ingressSuffix == "" {
		ingressSuffix = kDefaultIngressSuffix
	}
	if deployment.KubeNamespace != "" && cluster != nil {
		urls = append(urls, fmt.Sprintf("https://%s.%s/api/v2/deployments/%s?cluster=%s",
			deployment.KubeNamespace,
			ingressSuffix,
			deployment.Name,
			cluster.Name))
	}

	return &schemasv2.DeploymentSchema{
		ResourceSchema: schemas.ResourceSchema{
			Name:         deployment.Resource.Name,
			Labels:       []schemas.LabelItemSchema{},
			ResourceType: deployment.GetResourceType(),
			BaseSchema: schemas.BaseSchema{
				Uid:       deployment.Resource.BaseModel.GetUid(),
				CreatedAt: deployment.Resource.BaseModel.CreatedAt,
				UpdatedAt: deployment.Resource.BaseModel.UpdatedAt,
				DeletedAt: nil, // Can assume that this is nil during creation
			},
		}, // Assuming ResourceSchema can be copied directly
		Creator:        creator,
		Cluster:        clusterSchema,
		Status:         deployment.Status,
		URLs:           urls,
		LatestRevision: revisionSchema,
		KubeNamespace:  deployment.KubeNamespace,
	}, nil
}
