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
	"fmt"

	"github.com/gin-gonic/gin"

	"github.com/rs/zerolog/log"

	"github.com/dynemo-ai/dynemo/deploy/dynamo/api-server/api/converters"
	"github.com/dynemo-ai/dynemo/deploy/dynamo/api-server/api/models"
	"github.com/dynemo-ai/dynemo/deploy/dynamo/api-server/api/schemas"
	"github.com/dynemo-ai/dynemo/deploy/dynamo/api-server/api/services"
)

type clusterController struct{}

var ClusterController = clusterController{}

func (s *clusterController) GetCluster(ctx *gin.Context, clusterName string) (*models.Cluster, error) {
	ownership, err := GetOwnershipInfo(ctx)
	if err != nil {
		return nil, err
	}
	cluster, err := services.ClusterService.GetByName(ctx, ownership.OrganizationId, clusterName)
	if err != nil {
		return nil, err
	}
	return cluster, nil
}

func (c *clusterController) Create(ctx *gin.Context) {
	var schema schemas.CreateClusterSchema
	if err := ctx.ShouldBindJSON(&schema); err != nil {
		ctx.JSON(400, gin.H{"error": err.Error()})
		return
	}

	ownership, err := GetOwnershipInfo(ctx)
	if err != nil {
		ctx.JSON(400, err)
	}

	cluster, err := services.ClusterService.Create(ctx, services.CreateClusterOption{
		Name:           schema.Name,
		OrganizationId: ownership.OrganizationId,
		CreatorId:      ownership.UserId,
		Description:    schema.Description,
		KubeConfig:     schema.KubeConfig,
	})

	if err != nil {
		log.Info().Msgf("Failed to create cluster: %s", err.Error())
		ctx.JSON(500, gin.Error{Err: err})
		return
	}

	ctx.JSON(200, converters.ToClusterFullSchema(cluster))
}

func (c *clusterController) Update(ctx *gin.Context) {
	var schema schemas.UpdateClusterSchema
	clusterName := ctx.Param("clusterName")

	if err := ctx.ShouldBindJSON(&schema); err != nil {
		ctx.JSON(400, gin.H{"error": err.Error()})
		return
	}

	cluster, err := c.GetCluster(ctx, clusterName)
	if err != nil {
		ctx.JSON(404, gin.H{"error": fmt.Sprintf("Could not find cluster with the name %s", clusterName)})
		return
	}

	cluster, err = services.ClusterService.Update(ctx, cluster, services.UpdateClusterOption{
		Description: schema.Description,
		KubeConfig:  schema.KubeConfig,
	})

	if err != nil {
		log.Info().Msgf("Failed to update cluster: %s", err.Error())
		ctx.JSON(500, gin.H{"error": fmt.Sprintf("Error updating cluster %s", err.Error())})
		return
	}

	ctx.JSON(200, converters.ToClusterFullSchema(cluster))
}

func (c *clusterController) Get(ctx *gin.Context) {
	clusterName := ctx.Param("clusterName")

	cluster, err := c.GetCluster(ctx, clusterName)
	if err != nil {
		ctx.JSON(404, gin.H{"error": fmt.Sprintf("Could not find cluster with the name %s", clusterName)})
		return
	}

	ctx.JSON(200, converters.ToClusterFullSchema(cluster))
}

func (c *clusterController) List(ctx *gin.Context) {
	var schema schemas.ListQuerySchema

	if err := ctx.ShouldBindQuery(&schema); err != nil {
		ctx.JSON(400, gin.H{"error": err.Error()})
		return
	}

	ownership, err := GetOwnershipInfo(ctx)
	if err != nil {
		ctx.JSON(500, gin.H{"error": err.Error()})
		return
	}

	clusters, total, err := services.ClusterService.List(ctx, services.ListClusterOption{
		BaseListOption: services.BaseListOption{
			Start:  &schema.Start,
			Count:  &schema.Count,
			Search: schema.Search,
		},
		OrganizationId: &ownership.OrganizationId,
	})
	if err != nil {
		log.Info().Msgf("Failed to list clusters: %s", err.Error())
		ctx.JSON(400, gin.H{"Error": fmt.Sprintf("List clusters %s", err.Error())})
		return
	}

	clusterList := schemas.ClusterListSchema{
		BaseListSchema: schemas.BaseListSchema{
			Start: schema.Start,
			Count: schema.Count,
			Total: total,
		},
		Items: converters.ToClusterSchemaList(clusters),
	}

	ctx.JSON(200, clusterList)
}
