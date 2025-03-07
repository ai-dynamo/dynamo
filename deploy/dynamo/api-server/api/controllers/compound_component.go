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

type compoundComponentController struct{}

var CompoundComponentController = compoundComponentController{}

func (c *compoundComponentController) Register(ctx *gin.Context) {
	var getCluster schemas.GetClusterSchema
	var registerCompoundComponentSchema schemas.RegisterCompoundComponentSchema

	if err := ctx.ShouldBindUri(&getCluster); err != nil {
		ctx.JSON(400, gin.H{"error": err.Error()})
		return
	}

	if err := ctx.ShouldBindJSON(&registerCompoundComponentSchema); err != nil {
		ctx.JSON(400, gin.H{"error": err.Error()})
		return
	}

	names := []string{getCluster.ClusterName}
	clusters, _, err := services.ClusterService.List(ctx, services.ListClusterOption{
		Names: &names,
	})

	if err != nil {
		errMsg := fmt.Sprintf("Failed to get clusters %s when registering Compound Component: %s", getCluster.ClusterName, err.Error())
		log.Error().Msg(errMsg)
		ctx.JSON(500, gin.H{"error": errMsg})
		return
	}

	kubeNamespace := strings.TrimSpace(registerCompoundComponentSchema.KubeNamespace)

	// nolint: ineffassign, staticcheck
	tx, ctx_, df, err := database.DatabaseUtil.StartTransaction(ctx)
	defer func() { df(err) }()

	log.Info().Msgf("Registering compound component for %d clusters", len(clusters))
	var compoundComponent *models.CompoundComponent
	for _, cluster := range clusters {
		compoundComponent, err = services.CompoundComponentService.GetByName(ctx_, cluster.ID, string(registerCompoundComponentSchema.Name))
		isNotFound := errors.Is(err, consts.ErrNotFound)
		if err != nil && !isNotFound {
			log.Error().Msgf("Failed to get compoundComponent: %s", err.Error())
			ctx.JSON(500, gin.H{"error": "failed to get compoundComponent"})
			return
		}

		manifest := &schemas.CompoundComponentManifestSchema{
			SelectorLabels: registerCompoundComponentSchema.SelectorLabels,
		}
		if registerCompoundComponentSchema.Manifest != nil {
			manifest = registerCompoundComponentSchema.Manifest
		}

		if isNotFound {
			compoundComponent, err = services.CompoundComponentService.Create(ctx_, services.CreateCompoundComponentOption{
				ClusterId:     cluster.ID,
				Name:          string(registerCompoundComponentSchema.Name),
				KubeNamespace: kubeNamespace,
				Version:       registerCompoundComponentSchema.Version,
				Manifest:      manifest,
			})
		} else {
			now := time.Now()
			now_ := &now
			opt := services.UpdateCompoundComponentOption{
				LatestHeartbeatAt: &now_,
				Version:           &registerCompoundComponentSchema.Version,
				Manifest:          &manifest,
			}
			if compoundComponent.Version != registerCompoundComponentSchema.Version {
				opt.LatestInstalledAt = &now_
			}
			compoundComponent, err = services.CompoundComponentService.Update(ctx_, compoundComponent, opt)
		}

		if err != nil {
			log.Error().Msgf("Failed to register compoundComponent: %s", err.Error())
			ctx.JSON(500, gin.H{"error": "failed to register compoundComponent"})
			return
		}
	}

	tx.Commit()
	compoundComponentSchema, err := converters.ToCompoundComponentSchema(ctx, compoundComponent)
	if err != nil {
		log.Error().Msgf("Failed to convert compound component model to schema: %s", err.Error())
		ctx.JSON(500, gin.H{"error": err.Error()})
		return
	}

	ctx.JSON(200, compoundComponentSchema)

}

func (c *compoundComponentController) ListAll(ctx *gin.Context) {
	compoundComponents, err := services.CompoundComponentService.List(ctx, services.ListCompoundComponentOption{})
	if err != nil {
		errMsg := fmt.Sprintf("Failed to get all compoundComponents: %s", err.Error())
		log.Error().Msg(errMsg)
		ctx.JSON(400, gin.H{"error": errMsg})
		return
	}

	compoundComponentSchema, err := converters.ToCompoundComponentSchemas(ctx, compoundComponents)
	if err != nil {
		log.Error().Msgf("Failed to convert compound component model to schema: %s", err.Error())
		ctx.JSON(500, gin.H{"error": err.Error()})
		return
	}

	ctx.JSON(200, compoundComponentSchema)
}

func (c *compoundComponentController) List(ctx *gin.Context) {
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
		errMsg := fmt.Sprintf("Failed to get clusters %s when registering Compound Component: %s", getCluster.ClusterName, err.Error())
		log.Error().Msg(errMsg)
		ctx.JSON(500, gin.H{"error": errMsg})
		return
	}

	clusterIds := []uint{}
	for _, cluster := range clusters {
		clusterIds = append(clusterIds, cluster.ID)
	}

	compoundComponents, err := services.CompoundComponentService.List(ctx, services.ListCompoundComponentOption{
		ClusterIds: &clusterIds,
	})

	if err != nil {
		errMsg := fmt.Sprintf("Failed to get compoundComponents for the cluster %s: %s", getCluster.ClusterName, err.Error())
		log.Error().Msg(errMsg)
		ctx.JSON(500, gin.H{"error": errMsg})
		return
	}

	compoundComponentSchema, err := converters.ToCompoundComponentSchemas(ctx, compoundComponents)
	if err != nil {
		log.Error().Msgf("Failed to convert compound component model to schema: %s", err.Error())
		ctx.JSON(500, gin.H{"error": err.Error()})
		return
	}

	ctx.JSON(200, compoundComponentSchema)
}
