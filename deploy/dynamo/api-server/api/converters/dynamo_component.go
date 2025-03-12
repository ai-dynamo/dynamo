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

	"github.com/ai-dynamo/dynamo/deploy/dynamo/api-server/api/models"
	"github.com/ai-dynamo/dynamo/deploy/dynamo/api-server/api/schemas"
	"github.com/ai-dynamo/dynamo/deploy/dynamo/api-server/api/services"
	"github.com/pkg/errors"
)

func ToDynamoComponentSchema(ctx context.Context, dynamoComponent *models.DynamoComponent) (*schemas.DynamoComponentSchema, error) {
	if dynamoComponent == nil {
		return nil, nil
	}
	ss, err := ToDynamoComponentSchemas(ctx, []*models.DynamoComponent{dynamoComponent})
	if err != nil {
		return nil, errors.Wrap(err, "ToDynamoComponentSchemas")
	}
	return ss[0], nil
}

func ToDynamoComponentSchemas(ctx context.Context, dynamoComponents []*models.DynamoComponent) ([]*schemas.DynamoComponentSchema, error) {
	resourceSchemasMap := make(map[string]*schemas.ResourceSchema, len(dynamoComponents))
	for _, component := range dynamoComponents {
		resourceSchemasMap[component.GetUid()] = ToResourceSchema(&component.Resource, component.GetResourceType())
	}

	res := make([]*schemas.DynamoComponentSchema, 0, len(dynamoComponents))
	for _, dynamoComponent := range dynamoComponents {
		cluster, err := services.ClusterService.Get(ctx, dynamoComponent.ClusterId)
		if err != nil {
			return nil, err
		}
		clusterSchema := ToClusterFullSchema(cluster)
		resourceSchema, ok := resourceSchemasMap[dynamoComponent.GetUid()]
		if !ok {
			return nil, errors.Errorf("resource schema not found for DynamoComponent %s", dynamoComponent.GetUid())
		}
		res = append(res, &schemas.DynamoComponentSchema{
			ResourceSchema:    *resourceSchema,
			Cluster:           clusterSchema,
			Manifest:          dynamoComponent.Manifest,
			Version:           dynamoComponent.Version,
			KubeNamespace:     dynamoComponent.KubeNamespace,
			LatestHeartbeatAt: dynamoComponent.LatestHeartbeatAt,
			LatestInstalledAt: dynamoComponent.LatestInstalledAt,
		})
	}
	return res, nil
}
