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

	"github.com/dynemo-ai/dynemo/deploy/compoundai/api-server/api/models"
	"github.com/dynemo-ai/dynemo/deploy/compoundai/api-server/api/schemas"
	"github.com/dynemo-ai/dynemo/deploy/compoundai/api-server/api/services"
	"github.com/pkg/errors"
)

func ToCompoundComponentSchema(ctx context.Context, compoundComponent *models.CompoundComponent) (*schemas.CompoundComponentSchema, error) {
	if compoundComponent == nil {
		return nil, nil
	}
	ss, err := ToCompoundComponentSchemas(ctx, []*models.CompoundComponent{compoundComponent})
	if err != nil {
		return nil, errors.Wrap(err, "ToCompoundComponentSchemas")
	}
	return ss[0], nil
}

func ToCompoundComponentSchemas(ctx context.Context, compoundComponents []*models.CompoundComponent) ([]*schemas.CompoundComponentSchema, error) {
	resourceSchemasMap := make(map[string]*schemas.ResourceSchema, len(compoundComponents))
	for _, component := range compoundComponents {
		resourceSchemasMap[component.GetUid()] = ToResourceSchema(&component.Resource, component.GetResourceType())
	}

	res := make([]*schemas.CompoundComponentSchema, 0, len(compoundComponents))
	for _, compoundComponent := range compoundComponents {
		cluster, err := services.ClusterService.Get(ctx, compoundComponent.ClusterId)
		if err != nil {
			return nil, err
		}
		clusterSchema := ToClusterFullSchema(cluster)
		resourceSchema, ok := resourceSchemasMap[compoundComponent.GetUid()]
		if !ok {
			return nil, errors.Errorf("resource schema not found for CompoundComponent %s", compoundComponent.GetUid())
		}
		res = append(res, &schemas.CompoundComponentSchema{
			ResourceSchema:    *resourceSchema,
			Cluster:           clusterSchema,
			Manifest:          compoundComponent.Manifest,
			Version:           compoundComponent.Version,
			KubeNamespace:     compoundComponent.KubeNamespace,
			LatestHeartbeatAt: compoundComponent.LatestHeartbeatAt,
			LatestInstalledAt: compoundComponent.LatestInstalledAt,
		})
	}
	return res, nil
}
