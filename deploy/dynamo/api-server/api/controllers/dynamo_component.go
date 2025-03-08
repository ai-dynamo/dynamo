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
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/dynemo-ai/dynemo/deploy/dynamo/api-server/api/common/consts"
	"github.com/dynemo-ai/dynemo/deploy/dynamo/api-server/api/converters"
	"github.com/dynemo-ai/dynemo/deploy/dynamo/api-server/api/database"
	"github.com/dynemo-ai/dynemo/deploy/dynamo/api-server/api/models"
	"github.com/dynemo-ai/dynemo/deploy/dynamo/api-server/api/schemas"
	"github.com/dynemo-ai/dynemo/deploy/dynamo/api-server/api/services"
	"github.com/gin-gonic/gin"
	"github.com/rs/zerolog/log"
)

type dynamoComponentController struct{}

var DynamoComponentController = dynamoComponentController{}

func (c *dynamoComponentController) Register(ctx *gin.Context) {
	var getCluster schemas.GetClusterSchema
	var registerDynamoComponentSchema schemas.RegisterDynamoComponentSchema

	if err := ctx.ShouldBindUri(&getCluster); err != nil {
		ctx.JSON(400, gin.H{"error": err.Error()})
		return
	}

	if err := ctx.ShouldBindJSON(&registerDynamoComponentSchema); err != nil {
		ctx.JSON(400, gin.H{"error": err.Error()})
		return
	}

	names := []string{getCluster.ClusterName}
	clusters, _, err := services.ClusterService.List(ctx, services.ListClusterOption{
		Names: &names,
	})

	if err != nil {
		errMsg := fmt.Sprintf("Failed to get clusters %s when registering Dynamo Component: %s", getCluster.ClusterName, err.Error())
		log.Error().Msg(errMsg)
		ctx.JSON(500, gin.H{"error": errMsg})
		return
	}

	kubeNamespace := strings.TrimSpace(registerDynamoComponentSchema.KubeNamespace)

	// nolint: ineffassign, staticcheck
	tx, ctx_, df, err := database.DatabaseUtil.StartTransaction(ctx)
	defer func() { df(err) }()

	log.Info().Msgf("Registering dynamo component for %d clusters", len(clusters))
	var dynamoComponent *models.DynamoComponent
	for _, cluster := range clusters {
		dynamoComponent, err = services.DynamoComponentService.GetByName(ctx_, cluster.ID, string(registerDynamoComponentSchema.Name))
		isNotFound := errors.Is(err, consts.ErrNotFound)
		if err != nil && !isNotFound {
			log.Error().Msgf("Failed to get dynamoComponent: %s", err.Error())
			ctx.JSON(500, gin.H{"error": "failed to get dynamoComponent"})
			return
		}

		manifest := &schemas.DynamoComponentManifestSchema{
			SelectorLabels: registerDynamoComponentSchema.SelectorLabels,
		}
		if registerDynamoComponentSchema.Manifest != nil {
			manifest = registerDynamoComponentSchema.Manifest
		}

		if isNotFound {
			dynamoComponent, err = services.DynamoComponentService.Create(ctx_, services.CreateDynamoComponentOption{
				ClusterId:     cluster.ID,
				Name:          string(registerDynamoComponentSchema.Name),
				KubeNamespace: kubeNamespace,
				Version:       registerDynamoComponentSchema.Version,
				Manifest:      manifest,
			})
		} else {
			now := time.Now()
			now_ := &now
			opt := services.UpdateDynamoComponentOption{
				LatestHeartbeatAt: &now_,
				Version:           &registerDynamoComponentSchema.Version,
				Manifest:          &manifest,
			}
			if dynamoComponent.Version != registerDynamoComponentSchema.Version {
				opt.LatestInstalledAt = &now_
			}
			dynamoComponent, err = services.DynamoComponentService.Update(ctx_, dynamoComponent, opt)
		}

		if err != nil {
			log.Error().Msgf("Failed to register dynamoComponent: %s", err.Error())
			ctx.JSON(500, gin.H{"error": "failed to register dynamoComponent"})
			return
		}
	}

	tx.Commit()
	dynamoComponentSchema, err := converters.ToDynamoComponentSchema(ctx, dynamoComponent)
	if err != nil {
		log.Error().Msgf("Failed to convert dynamo component model to schema: %s", err.Error())
		ctx.JSON(500, gin.H{"error": err.Error()})
		return
	}

	ctx.JSON(200, dynamoComponentSchema)

}

func (c *dynamoComponentController) ListAll(ctx *gin.Context) {
	dynamoComponents, err := services.DynamoComponentService.List(ctx, services.ListDynamoComponentOption{})
	if err != nil {
		errMsg := fmt.Sprintf("Failed to get all dynamoComponents: %s", err.Error())
		log.Error().Msg(errMsg)
		ctx.JSON(400, gin.H{"error": errMsg})
		return
	}

	dynamoComponentSchema, err := converters.ToDynamoComponentSchemas(ctx, dynamoComponents)
	if err != nil {
		log.Error().Msgf("Failed to convert dynamo component model to schema: %s", err.Error())
		ctx.JSON(500, gin.H{"error": err.Error()})
		return
	}

	ctx.JSON(200, dynamoComponentSchema)
}

func (c *dynamoComponentController) List(ctx *gin.Context) {
	var getCluster schemas.GetClusterSchema

	if err := ctx.ShouldBindUri(&getCluster); err != nil {
		ctx.JSON(400, gin.H{"error": err.Error()})
		return
	}

	names := []string{getCluster.ClusterName}
	clusters, _, err := services.ClusterService.List(ctx, services.ListClusterOption{
		Names: &names,
	})

	if err != nil {
		errMsg := fmt.Sprintf("Failed to get clusters %s when registering Dynamo Component: %s", getCluster.ClusterName, err.Error())
		log.Error().Msg(errMsg)
		ctx.JSON(500, gin.H{"error": errMsg})
		return
	}

	clusterIds := []uint{}
	for _, cluster := range clusters {
		clusterIds = append(clusterIds, cluster.ID)
	}

	dynamoComponents, err := services.DynamoComponentService.List(ctx, services.ListDynamoComponentOption{
		ClusterIds: &clusterIds,
	})

	if err != nil {
		errMsg := fmt.Sprintf("Failed to get dynamoComponents for the cluster %s: %s", getCluster.ClusterName, err.Error())
		log.Error().Msg(errMsg)
		ctx.JSON(500, gin.H{"error": errMsg})
		return
	}

	dynamoComponentSchema, err := converters.ToDynamoComponentSchemas(ctx, dynamoComponents)
	if err != nil {
		log.Error().Msgf("Failed to convert dynamo component model to schema: %s", err.Error())
		ctx.JSON(500, gin.H{"error": err.Error()})
		return
	}

	ctx.JSON(200, dynamoComponentSchema)
}
