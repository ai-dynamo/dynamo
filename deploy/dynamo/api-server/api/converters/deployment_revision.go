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

	"github.com/ai-dynamo/dynamo/deploy/dynamo/api-server/api/mocks"
	"github.com/ai-dynamo/dynamo/deploy/dynamo/api-server/api/models"
	"github.com/ai-dynamo/dynamo/deploy/dynamo/api-server/api/schemas"
	"github.com/ai-dynamo/dynamo/deploy/dynamo/api-server/api/services"
)

func ToDeploymentRevisionSchema(ctx context.Context, deploymentRevision *models.DeploymentRevision) (*schemas.DeploymentRevisionSchema, error) {
	if deploymentRevision == nil {
		return nil, nil
	}
	ss, err := ToDeploymentRevisionSchemas(ctx, []*models.DeploymentRevision{deploymentRevision})
	if err != nil {
		return nil, err
	}
	return ss[0], nil
}

func ToDeploymentRevisionSchemas(ctx context.Context, deploymentRevisions []*models.DeploymentRevision) ([]*schemas.DeploymentRevisionSchema, error) {
	deploymentRevisionIds := make([]uint, 0, len(deploymentRevisions))
	for _, deploymentRevision := range deploymentRevisions {
		deploymentRevisionIds = append(deploymentRevisionIds, deploymentRevision.ID)
	}
	deploymentTargets, _, err := services.DeploymentTargetService.List(ctx, services.ListDeploymentTargetOption{
		DeploymentRevisionIds: &deploymentRevisionIds,
	})
	if err != nil {
		return nil, err
	}
	deploymentTargetsMapping := make(map[uint][]*models.DeploymentTarget)
	for _, deploymentTarget := range deploymentTargets {
		deploymentTargets, ok := deploymentTargetsMapping[deploymentTarget.DeploymentRevisionId]
		if !ok {
			deploymentTargets = make([]*models.DeploymentTarget, 0)
		}
		deploymentTargets = append(deploymentTargets, deploymentTarget)
		deploymentTargetsMapping[deploymentTarget.DeploymentRevisionId] = deploymentTargets
	}

	resourceSchemasMap := make(map[string]*schemas.ResourceSchema, len(deploymentRevisions))
	for _, revision := range deploymentRevisions {
		resourceSchemasMap[revision.GetUid()] = ToResourceSchema(revisionToResource(revision), revision.GetResourceType())
	}

	res := make([]*schemas.DeploymentRevisionSchema, 0, len(deploymentRevisions))
	for _, deploymentRevision := range deploymentRevisions {
		creatorSchema := mocks.DefaultUser()

		deploymentTargets := deploymentTargetsMapping[deploymentRevision.ID]
		deploymentTargetSchemas, err := ToDeploymentTargetSchemas(ctx, deploymentTargets)
		if err != nil {
			return nil, err
		}
		resourceSchema, ok := resourceSchemasMap[deploymentRevision.GetUid()]
		if !ok {
			return nil, fmt.Errorf("resourceSchema not found for deploymentRevision %s", deploymentRevision.GetUid())
		}
		res = append(res, &schemas.DeploymentRevisionSchema{
			ResourceSchema: *resourceSchema,
			Creator:        creatorSchema,
			Status:         deploymentRevision.Status,
			Targets:        deploymentTargetSchemas,
		})
	}
	return res, nil
}

func revisionToResource(deploymentTarget *models.DeploymentRevision) *models.Resource {
	return &models.Resource{
		BaseModel: deploymentTarget.BaseModel,
		Name:      deploymentTarget.GetUid(),
	}
}
