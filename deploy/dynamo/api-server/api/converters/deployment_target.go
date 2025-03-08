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
	"strings"

	"github.com/dynemo-ai/dynemo/deploy/dynamo/api-server/api/mocks"
	"github.com/dynemo-ai/dynemo/deploy/dynamo/api-server/api/models"
	"github.com/dynemo-ai/dynemo/deploy/dynamo/api-server/api/schemas"
	"github.com/dynemo-ai/dynemo/deploy/dynamo/api-server/api/services"
	"github.com/pkg/errors"
)

func ToDeploymentTargetSchema(ctx context.Context, deploymentTarget *models.DeploymentTarget) (*schemas.DeploymentTargetSchema, error) {
	if deploymentTarget == nil {
		return nil, nil
	}
	ss, err := ToDeploymentTargetSchemas(ctx, []*models.DeploymentTarget{deploymentTarget})
	if err != nil {
		return nil, err
	}
	return ss[0], nil
}

func ToDeploymentTargetSchemas(ctx context.Context, deploymentTargets []*models.DeploymentTarget) ([]*schemas.DeploymentTargetSchema, error) {
	resourceSchemasMap := make(map[string]*schemas.ResourceSchema, len(deploymentTargets))
	for _, target := range deploymentTargets {
		resourceSchemasMap[target.GetUid()] = ToResourceSchema(targetToResource(target), target.GetResourceType())
	}

	res := make([]*schemas.DeploymentTargetSchema, 0, len(deploymentTargets))
	for _, deploymentTarget := range deploymentTargets {
		creatorSchema := mocks.DefaultUser()

		dynamoNimParts := strings.Split(deploymentTarget.DynamoNimVersionTag, ":")
		if len(dynamoNimParts) != 2 {
			return nil, errors.Errorf("Invalid format for DynamoNIM version tag %s. Expected 2 parts got %d", deploymentTarget.DynamoNimVersionTag, len(dynamoNimParts))
		}

		dynamoNimVersionFullSchema, err := services.DatastoreService.GetDynamoNimVersion(ctx, dynamoNimParts[0], dynamoNimParts[1])
		if err != nil {
			dynamoNimVersionFullSchema = nil // We shouldn't fail the request if this info is missing
		}

		resourceSchema, ok := resourceSchemasMap[deploymentTarget.GetUid()]
		if !ok {
			return nil, fmt.Errorf("resourceSchema not found for deploymentTarget %s", deploymentTarget.GetUid())
		}
		res = append(res, &schemas.DeploymentTargetSchema{
			ResourceSchema: *resourceSchema,
			DeploymentTargetTypeSchema: schemas.DeploymentTargetTypeSchema{
				Type: "stable",
			},
			Creator:          creatorSchema,
			DynamoNimVersion: dynamoNimVersionFullSchema,
			Config:           deploymentTarget.Config,
		})
	}
	return res, nil
}

func targetToResource(deploymentTarget *models.DeploymentTarget) *models.Resource {
	return &models.Resource{
		BaseModel: deploymentTarget.BaseModel,
		Name:      deploymentTarget.GetUid(),
	}
}
