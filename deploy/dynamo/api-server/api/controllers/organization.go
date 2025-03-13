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

package controllers

import (
	"context"

	"github.com/ai-dynamo/dynamo/deploy/dynamo/api-server/api/common/consts"
	"github.com/ai-dynamo/dynamo/deploy/dynamo/api-server/api/converters"
	"github.com/ai-dynamo/dynamo/deploy/dynamo/api-server/api/mocks"
	"github.com/ai-dynamo/dynamo/deploy/dynamo/api-server/api/schemas"
	"github.com/gin-gonic/gin"
	"github.com/pkg/errors"
	"github.com/rs/zerolog/log"
)

type organizationController struct{}

var OrganizationController = organizationController{}

const CurrentOrganizationKey = "currentOrganization"
const CurrentOrganizationIdKey = "currentOrganizationId"

func GetCurrentOrganization(ctx context.Context) (*schemas.OrganizationSchema, error) {
	org_ := ctx.Value(CurrentOrganizationKey)
	if org_ == nil {
		return nil, consts.ErrNotFound
	}
	org, ok := org_.(*schemas.OrganizationSchema)
	if !ok {
		return nil, errors.New("current organization is not a organization")
	}
	return org, nil
}

func (c *organizationController) Create(ctx *gin.Context) {
	ctx.JSON(501, gin.H{"error": "not supported."})
}

func (c *organizationController) Update(ctx *gin.Context) {
	ctx.JSON(501, gin.H{"error": "not supported."})
}

func (c *organizationController) Get(ctx *gin.Context) {
	organization, err := GetCurrentOrganization(ctx)
	if err != nil {
		ctx.JSON(400, gin.H{"error": err.Error()})
		return
	}

	ctx.JSON(200, organization)
}

func (c *organizationController) GetMajorCluster(ctx *gin.Context) {
	cluster, err := ClusterController.GetCluster(ctx, "default")
	if err != nil {
		log.Info().Msgf("Failed to get default cluster: %s", err.Error())
		ctx.JSON(404, gin.H{"error": err.Error()})
		return
	}

	ctx.JSON(200, converters.ToClusterFullSchema(cluster))
}

func (c *organizationController) List(ctx *gin.Context) {
	var schema schemas.ListQuerySchema
	if err := ctx.ShouldBindQuery(&schema); err != nil {
		ctx.JSON(400, gin.H{"error": err.Error()})
		return
	}

	organizationSchemas := []*schemas.OrganizationSchema{mocks.DefaultOrg()}
	organizationListSchema := schemas.OrganizationListSchema{
		BaseListSchema: schemas.BaseListSchema{
			Total: 1,
			Start: schema.Start,
			Count: schema.Count,
		},
		Items: organizationSchemas,
	}

	ctx.JSON(200, organizationListSchema)
}

func (c *organizationController) ListEventOperationNames(ctx *gin.Context) {
	ctx.JSON(200, []string{})
}

func (c *organizationController) ListEvents(ctx *gin.Context) {
	ctx.JSON(200, schemas.EventListSchema{
		Items: []*schemas.EventSchema{},
	})
}
