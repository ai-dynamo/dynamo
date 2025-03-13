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

	"github.com/ai-dynamo/dynamo/deploy/dynamo/api-server/api/converters"
	"github.com/ai-dynamo/dynamo/deploy/dynamo/api-server/api/schemas"
	"github.com/ai-dynamo/dynamo/deploy/dynamo/api-server/api/services"
	"github.com/gin-gonic/gin"
	"github.com/rs/zerolog/log"
)

type deploymentRevisionController struct{}

var DeploymentRevisionController = deploymentRevisionController{}

func (c *deploymentRevisionController) List(ctx *gin.Context) {
	var schema schemas.ListQuerySchema
	var getSchema schemas.GetDeploymentSchema

	if err := ctx.ShouldBindUri(&getSchema); err != nil {
		log.Error().Msgf("Error binding: %s", err.Error())
		ctx.JSON(400, gin.H{"error": err.Error()})
		return
	}

	if err := ctx.ShouldBindQuery(&schema); err != nil {
		ctx.JSON(400, gin.H{"error": err.Error()})
		return
	}

	deployment, err := getDeployment(ctx, &getSchema)
	if err != nil {
		log.Error().Msgf("Could not find deployment with the name %s: %s", getSchema.DeploymentName, err.Error())
		ctx.JSON(404, gin.H{"error": fmt.Sprintf("Could not find deployment with the name %s", getSchema.DeploymentName)})
		return
	}

	deploymentRevisions, total, err := services.DeploymentRevisionService.List(ctx, services.ListDeploymentRevisionOption{
		BaseListOption: services.BaseListOption{
			Start:  &schema.Start,
			Count:  &schema.Count,
			Search: schema.Search,
		},
		DeploymentId: &deployment.ID,
	})

	if err != nil {
		errMsg := fmt.Sprintf("Failed to get deployment revisions %s", err.Error())
		log.Error().Msgf(errMsg)
		ctx.JSON(500, gin.H{"error": errMsg})
		return
	}

	deploymentRevisionSchemas, err := converters.ToDeploymentRevisionSchemas(ctx, deploymentRevisions)
	if err != nil {
		errMsg := fmt.Sprintf("Failed to convert models to deployment revision schemas %s", err.Error())
		log.Error().Msgf(errMsg)
		ctx.JSON(500, gin.H{"error": errMsg})
		return
	}
	log.Info().Msgf("Got %d deployment revisions", len(deploymentRevisionSchemas))

	deploymentRevisionListSchema := schemas.DeploymentRevisionListSchema{
		BaseListSchema: schemas.BaseListSchema{
			Total: total,
			Start: schema.Start,
			Count: schema.Count,
		},
		Items: deploymentRevisionSchemas,
	}

	ctx.JSON(200, deploymentRevisionListSchema)
}

func (c *deploymentRevisionController) Get(ctx *gin.Context) {
	var schema schemas.GetDeploymentRevisionSchema

	if err := ctx.ShouldBindUri(&schema); err != nil {
		log.Error().Msgf("Error binding: %s", err.Error())
		ctx.JSON(400, gin.H{"error": err.Error()})
		return
	}

	_, err := getDeployment(ctx, &schema.GetDeploymentSchema)
	if err != nil {
		log.Error().Msgf("Could not find deployment with the name %s: %s", schema.DeploymentName, err.Error())
		ctx.JSON(404, gin.H{"error": fmt.Sprintf("Could not find deployment with the name %s", schema.DeploymentName)})
		return
	}

	deploymentRevision, err := services.DeploymentRevisionService.GetByUid(ctx, schema.RevisionUid)
	if err != nil {
		errMsg := fmt.Sprintf("Failed to get deployment revisions %s for %s", schema.DeploymentName, err.Error())
		log.Error().Msgf(errMsg)
		ctx.JSON(404, gin.H{"error": errMsg})
		return
	}

	deploymentRevisionSchema, err := converters.ToDeploymentRevisionSchema(ctx, deploymentRevision)
	if err != nil {
		errMsg := fmt.Sprintf("Failed to convert model to deployment revision schema %s", err.Error())
		log.Error().Msgf(errMsg)
		ctx.JSON(500, gin.H{"error": errMsg})
		return
	}

	ctx.JSON(200, deploymentRevisionSchema)
}
