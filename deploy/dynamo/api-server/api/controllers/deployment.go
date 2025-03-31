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
	"errors"
	"fmt"
	"os"
	"strings"

	"github.com/ai-dynamo/dynamo/deploy/dynamo/api-server/api/common/env"
	"github.com/ai-dynamo/dynamo/deploy/dynamo/api-server/api/converters"
	"github.com/ai-dynamo/dynamo/deploy/dynamo/api-server/api/database"
	"github.com/ai-dynamo/dynamo/deploy/dynamo/api-server/api/mocks"
	"github.com/ai-dynamo/dynamo/deploy/dynamo/api-server/api/models"
	"github.com/ai-dynamo/dynamo/deploy/dynamo/api-server/api/schemas"
	"github.com/ai-dynamo/dynamo/deploy/dynamo/api-server/api/schemasv2"
	"github.com/ai-dynamo/dynamo/deploy/dynamo/api-server/api/services"
	dynamov1alpha1 "github.com/ai-dynamo/dynamo/deploy/dynamo/operator/api/v1alpha1"
	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
	"github.com/invopop/jsonschema"
	"github.com/rs/zerolog/log"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

type deploymentController struct{}

var DeploymentController = deploymentController{}

type CreateDeploymentSchema struct {
	schemas.CreateDeploymentSchema
}

func (c *deploymentController) Create(ctx *gin.Context) {
	clusterName := ctx.Param("clusterName")
	var schema CreateDeploymentSchema

	if err := ctx.ShouldBindJSON(&schema); err != nil {
		ctx.JSON(400, gin.H{"error": err.Error()})
		return
	}

	cluster, err := ClusterController.GetCluster(ctx, clusterName)
	if err != nil {
		ctx.JSON(404, fmt.Sprintf("Could not find cluster with the name %s", clusterName))
		return
	}

	deployment, err := c.createDeploymentHelper(ctx, cluster, schema)
	if err != nil {
		ctx.JSON(400, gin.H{"error": err.Error()})
		return
	}

	log.Info().Msgf("Created deployment: %+v", deployment)
	deploymentSchema, err := converters.ToDeploymentSchema(ctx, deployment)
	if err != nil {
		log.Error().Msgf("Failed to convert deployment model to schema: %s", err.Error())
		ctx.JSON(500, gin.H{"error": err.Error()})
		return
	}

	ctx.JSON(200, deploymentSchema)
}

func (c *deploymentController) createDeploymentHelper(ctx *gin.Context, cluster *models.Cluster, schema CreateDeploymentSchema) (*models.Deployment, error) {
	description := ""
	if schema.Description != nil {
		description = *schema.Description
	}

	_, ctx_, df, err := database.DatabaseUtil.StartTransaction(ctx)
	defer func() { df(err) }() // Clean up the transaction

	ownership, err := GetOwnershipInfo(ctx)
	if err != nil {
		return nil, err
	}

	deployment, err := services.DeploymentService.Create(ctx_, services.CreateDeploymentOption{
		CreatorId:     ownership.UserId,
		ClusterId:     cluster.ID,
		Name:          schema.Name,
		Description:   description,
		KubeNamespace: schema.KubeNamespace,
	})
	if err != nil {
		log.Error().Msgf("Creating deployment failed: %s", err.Error())
		return nil, fmt.Errorf("creating deployment failed: %s", err.Error())
	}

	_, err = c.updateDeploymentEntities(ctx_, schema.UpdateDeploymentSchema, deployment, ownership)
	if err != nil {
		log.Error().Msgf("Failed to update deployment %s entities %s", deployment.Name, err.Error())
		return nil, fmt.Errorf("failed to update deployment %s entities %s", deployment.Name, err.Error())
	}

	return deployment, nil
}

func (c *deploymentController) SyncStatus(ctx *gin.Context) {
	var schema schemas.GetDeploymentSchema

	if err := ctx.ShouldBindUri(&schema); err != nil {
		ctx.JSON(400, gin.H{"error": err.Error()})
		return
	}

	deployment, err := getDeployment(ctx, &schema)

	if err != nil {
		log.Error().Msgf("Could not find deployment with the name %s: %s", schema.DeploymentName, err.Error())
		ctx.JSON(404, fmt.Sprintf("Could not find deployment with the name %s", schema.DeploymentName))
		return
	}

	status, err := services.DeploymentService.SyncStatus(ctx, deployment)
	if err != nil {
		log.Error().Msgf("Failed to sync deployment %s status: %s", deployment.Name, err.Error())
		ctx.JSON(500, gin.H{"error": err.Error()})
		return
	}

	deployment.Status = status

	deploymentSchema, err := converters.ToDeploymentSchema(ctx, deployment)
	if err != nil {
		log.Error().Msgf("Failed to convert deployment model to schema: %s", err.Error())
		ctx.JSON(500, gin.H{"error": err.Error()})
		return
	}

	ctx.JSON(200, deploymentSchema)
}

func (c *deploymentController) Update(ctx *gin.Context) {
	var updateSchema schemas.UpdateDeploymentSchema
	var getSchema schemas.GetDeploymentSchema

	if err := ctx.ShouldBindUri(&getSchema); err != nil {
		log.Error().Msgf("Error binding: %s", err.Error())
		ctx.JSON(400, gin.H{"error": err.Error()})
		return
	}

	if err := ctx.ShouldBindJSON(&updateSchema); err != nil {
		log.Error().Msgf("Error binding: %s", err.Error())
		ctx.JSON(400, gin.H{"error": err.Error()})
		return
	}

	ownership, err := GetOwnershipInfo(ctx)
	if err != nil {
		ctx.JSON(500, err)
		return
	}

	deployment, err := getDeployment(ctx, &getSchema)

	if err != nil {
		log.Error().Msgf("Could not find deployment with the name %s: %s", getSchema.DeploymentName, err.Error())
		ctx.JSON(404, fmt.Sprintf("Could not find deployment with the name %s", getSchema.DeploymentName))
		return
	}

	tx, ctx_, df, err := database.DatabaseUtil.StartTransaction(ctx)
	defer func() { df(err) }() // Clean up the transaction

	deployment, err = services.DeploymentService.Update(ctx_, deployment, services.UpdateDeploymentOption{
		Description: updateSchema.Description,
	})

	if err != nil {
		log.Error().Msgf("Could not update deployment with the name %s: %s", getSchema.DeploymentName, err.Error())
		ctx.JSON(500, fmt.Sprintf("Could not update deployment with the name %s", getSchema.DeploymentName))
		return
	}

	if updateSchema.DoNotDeploy {
		deployment, err = c.updateDeploymentInformation(ctx_, updateSchema, deployment)
		if err != nil {
			log.Error().Msgf("Could not update deployment information %s: %s", getSchema.DeploymentName, err.Error())
			ctx.JSON(500, err.Error())
			return
		}
	} else {
		deployment, err = c.updateDeploymentEntities(ctx_, updateSchema, deployment, ownership)
		if err != nil {
			log.Error().Msgf("Could not update deployment entities %s: %s", getSchema.DeploymentName, err.Error())
			ctx.JSON(500, err.Error())
			return
		}
	}
	tx.Commit()

	deploymentSchema, err := converters.ToDeploymentSchema(ctx, deployment)
	if err != nil {
		log.Error().Msgf("Failed to convert deployment model to schema: %s", err.Error())
		ctx.JSON(500, gin.H{"error": err.Error()})
		return
	}

	ctx.JSON(200, deploymentSchema)
}

func (c *deploymentController) updateDeploymentEntities(ctx context.Context, schema schemas.UpdateDeploymentSchema, deployment *models.Deployment, ownership *schemas.OwnershipSchema) (*models.Deployment, error) {
	dynamoNimVersions := map[string]*schemas.DynamoNimVersionFullSchema{}
	for _, target := range schema.Targets {
		dynamoNimVersionSchema, err := services.BackendService.GetDynamoNimVersion(ctx, target.DynamoNim, target.Version)
		if err != nil {
			return nil, err
		}
		dynamoNimVersions[fmt.Sprintf("%s:%s", target.DynamoNim, target.Version)] = dynamoNimVersionSchema
	}
	log.Info().Msgf("Found %d Dynamo NIM versions", len(dynamoNimVersions))

	// Mark previous revisions as inactive...
	status_ := schemas.DeploymentRevisionStatusActive
	oldDeploymentRevisions, total, err := services.DeploymentRevisionService.List(ctx, services.ListDeploymentRevisionOption{
		DeploymentId: &deployment.ID,
		Status:       &status_,
	})
	if err != nil {
		return nil, err
	}

	var oldDeploymentTargets = make([]*models.DeploymentTarget, 0)

	log.Info().Msgf("Marking %d version as inactive", total)
	for _, oldDeploymentRevision := range oldDeploymentRevisions {
		_, err = services.DeploymentRevisionService.Update(ctx, oldDeploymentRevision, services.UpdateDeploymentRevisionOption{
			Status: schemas.DeploymentRevisionStatusPtr(schemas.DeploymentRevisionStatusInactive),
		})

		if err != nil {
			return nil, err
		}

		_oldDeploymentTargets, _, err := services.DeploymentTargetService.List(ctx, services.ListDeploymentTargetOption{
			DeploymentRevisionId: &oldDeploymentRevision.ID,
		})

		oldDeploymentTargets = append(oldDeploymentTargets, _oldDeploymentTargets...)

		if err != nil {
			return nil, err
		}
	}

	// Create a new revision
	deploymentRevision, err := services.DeploymentRevisionService.Create(ctx, services.CreateDeploymentRevisionOption{
		CreatorId:    ownership.UserId,
		DeploymentId: deployment.ID,
		Status:       schemas.DeploymentRevisionStatusActive,
	})
	if err != nil {
		return nil, err
	}

	// Create deployment targets
	deploymentTargets := make([]*models.DeploymentTarget, 0, len(schema.Targets))
	for _, createDeploymentTargetSchema := range schema.Targets {
		createDeploymentTargetSchema.Config.KubeResourceVersion = ""
		createDeploymentTargetSchema.Config.KubeResourceUid = ""

		dynamoNimTag := fmt.Sprintf("%s:%s", createDeploymentTargetSchema.DynamoNim, createDeploymentTargetSchema.Version)
		deploymentTarget, err := services.DeploymentTargetService.Create(ctx, services.CreateDeploymentTargetOption{
			CreatorId:            ownership.UserId,
			DeploymentId:         deployment.ID,
			DeploymentRevisionId: deploymentRevision.ID,
			DynamoNimVersionId:   dynamoNimVersions[dynamoNimTag].Uid,
			DynamoNimVersionTag:  dynamoNimTag,
			Config:               createDeploymentTargetSchema.Config,
		})
		if err != nil {
			return nil, err
		}
		deploymentTargets = append(deploymentTargets, deploymentTarget)
	}

	log.Info().Msgf("Terminating %d inactive deployment targets", len(oldDeploymentTargets))
	for _, oldDeploymentTarget := range oldDeploymentTargets {
		_, err := services.DeploymentTargetService.Terminate(ctx, oldDeploymentTarget)

		if err != nil {
			return nil, err
		}
	}

	// Deploy new revision
	err = services.DeploymentRevisionService.Deploy(ctx, deploymentRevision, deploymentTargets, ownership, false)
	if err != nil {
		return nil, err
	}

	return deployment, nil
}

func (c *deploymentController) updateDeploymentInformation(ctx context.Context, schema schemas.UpdateDeploymentSchema, deployment *models.Deployment) (*models.Deployment, error) {
	status_ := schemas.DeploymentRevisionStatusActive
	activeReploymentRevisions, _, err := services.DeploymentRevisionService.List(ctx, services.ListDeploymentRevisionOption{
		DeploymentId: &deployment.ID,
		Status:       &status_,
	})
	if err != nil {
		return nil, err
	}

	targetSchemaDynamoVersionNims := map[string]*schemas.CreateDeploymentTargetSchema{}
	for _, targetSchema := range schema.Targets {
		targetSchemaDynamoVersionNims[fmt.Sprintf("%s:%s", targetSchema.DynamoNim, targetSchema.Version)] = targetSchema
	}

	var activeDeploymentTargets = make([]*models.DeploymentTarget, 0)

	for _, activeReploymentRevision := range activeReploymentRevisions {
		_activeDeploymentTargets, _, err := services.DeploymentTargetService.List(ctx, services.ListDeploymentTargetOption{
			DeploymentRevisionId: &activeReploymentRevision.ID,
		})

		activeDeploymentTargets = append(activeDeploymentTargets, _activeDeploymentTargets...)

		if err != nil {
			return nil, err
		}
	}

	for _, activeDeploymentTarget := range activeDeploymentTargets {
		if createDeploymentTargetSchema, ok := targetSchemaDynamoVersionNims[activeDeploymentTarget.DynamoNimVersionTag]; ok {
			config := activeDeploymentTarget.Config
			config.KubeResourceUid = createDeploymentTargetSchema.Config.KubeResourceUid
			config.KubeResourceVersion = createDeploymentTargetSchema.Config.KubeResourceVersion

			_, err = services.DeploymentTargetService.Update(ctx, activeDeploymentTarget, services.UpdateDeploymentTargetOption{
				Config: &config,
			})
			if err != nil {
				return nil, err
			}
		}
	}

	return deployment, nil
}

func (c *deploymentController) Get(ctx *gin.Context) {
	var schema schemas.GetDeploymentSchema

	if err := ctx.ShouldBindUri(&schema); err != nil {
		ctx.JSON(400, gin.H{"error": err.Error()})
		return
	}

	deployment, err := getDeployment(ctx, &schema)

	if err != nil {
		log.Error().Msgf("Could not find deployment with the name %s: %s", schema.DeploymentName, err.Error())
		ctx.JSON(404, fmt.Sprintf("Could not find deployment with the name %s", schema.DeploymentName))
		return
	}

	deploymentSchema, err := converters.ToDeploymentSchema(ctx, deployment)
	if err != nil {
		log.Error().Msgf("Failed to convert deployment model to schema: %s", err.Error())
		ctx.JSON(500, gin.H{"error": err.Error()})
		return
	}

	ctx.JSON(200, deploymentSchema)
}

func (c *deploymentController) Terminate(ctx *gin.Context) {
	var schema schemas.GetDeploymentSchema

	if err := ctx.ShouldBindUri(&schema); err != nil {
		ctx.JSON(400, gin.H{"error": err.Error()})
		return
	}

	deployment, err := getDeployment(ctx, &schema)
	if err != nil {
		log.Error().Msgf("Could not find deployment with the name %s: %s", schema.DeploymentName, err.Error())
		ctx.JSON(404, fmt.Sprintf("Could not find deployment with the name %s", schema.DeploymentName))
		return
	}

	deployment, err = c.doTerminate(ctx, deployment)
	if err != nil {
		ctx.JSON(500, gin.H{"error": err.Error()})
		return
	}

	deploymentSchema, err := converters.ToDeploymentSchema(ctx, deployment)
	if err != nil {
		log.Error().Msgf("Failed to convert deployment model to schema: %s", err.Error())
		ctx.JSON(500, gin.H{"error": err.Error()})
		return
	}

	ctx.JSON(200, deploymentSchema)
}

func (c *deploymentController) doTerminate(ctx *gin.Context, deployment *models.Deployment) (*models.Deployment, error) {
	tx, ctx_, df, err := database.DatabaseUtil.StartTransaction(ctx)
	defer func() { df(err) }() // Clean up the transaction

	deployment, err = services.DeploymentService.Terminate(ctx_, deployment)
	if err != nil {
		errMsg := fmt.Sprintf("Could not terminate deployment with the name: %s", err.Error())
		log.Error().Msgf(errMsg)
		return nil, errors.New(errMsg)
	}
	tx.Commit()

	return deployment, nil
}

func (c *deploymentController) Delete(ctx *gin.Context) {
	var schema schemas.GetDeploymentSchema

	if err := ctx.ShouldBindUri(&schema); err != nil {
		ctx.JSON(400, gin.H{"error": err.Error()})
		return
	}

	deployment, err := getDeployment(ctx, &schema)

	if err != nil {
		log.Error().Msgf("Could not find deployment with the name %s: %s", schema.DeploymentName, err.Error())
		ctx.JSON(404, fmt.Sprintf("Could not find deployment with the name %s", schema.DeploymentName))
		return
	}

	deployment, err = services.DeploymentService.Delete(ctx, deployment)
	if err != nil {
		log.Error().Msgf("Could not delete deployment with the name %s: %s", schema.DeploymentName, err.Error())
		ctx.JSON(500, gin.H{"error": fmt.Sprintf("Could not delete deployment with the name %s", schema.DeploymentName)})
		return
	}

	deploymentSchema, err := converters.ToDeploymentSchema(ctx, deployment)
	if err != nil {
		log.Error().Msgf("Failed to convert deployment model to schema: %s", err.Error())
		ctx.JSON(500, gin.H{"error": err.Error()})
		return
	}

	ctx.JSON(200, deploymentSchema)
}

func setListDeploymentOptionsScope(opt *services.ListDeploymentOption, ownership *schemas.OwnershipSchema) {
	opt.OrganizationId = &ownership.OrganizationId
	if env.ApplicationScope == env.UserScope {
		opt.CreatorId = &ownership.UserId
	}
}

func (c *deploymentController) ListClusterDeployments(ctx *gin.Context) {
	var schema schemas.ListQuerySchema
	var getCluster schemas.GetClusterSchema

	if err := ctx.ShouldBindUri(&getCluster); err != nil {
		ctx.JSON(400, gin.H{"error": err.Error()})
		return
	}

	if err := ctx.ShouldBindQuery(&schema); err != nil {
		ctx.JSON(400, gin.H{"error": err.Error()})
		return
	}

	cluster, err := ClusterController.GetCluster(ctx, getCluster.ClusterName)

	if err != nil {
		log.Error().Msgf("Could not find cluster with the name %s: %s", getCluster.ClusterName, err.Error())
		ctx.JSON(404, gin.H{"error": fmt.Sprintf("Could not find cluster with the name %s", getCluster.ClusterName)})
		return
	}

	ownership, err := GetOwnershipInfo(ctx)
	if err != nil {
		ctx.JSON(500, gin.H{"error": err.Error()})
		return
	}

	listOpt := services.ListDeploymentOption{
		BaseListOption: services.BaseListOption{
			Start:  &schema.Start,
			Count:  &schema.Count,
			Search: schema.Search,
		},
		ClusterId: &cluster.ID,
	}

	setListDeploymentOptionsScope(&listOpt, ownership)

	deployments, total, err := services.DeploymentService.List(ctx, listOpt)
	if err != nil {
		log.Error().Msgf("Could not find deployments for the cluster %s with the following opts %+v: %s", getCluster.ClusterName, listOpt, err.Error())
		ctx.JSON(500, gin.H{"error": fmt.Sprintf("Could not find deployments %s", err.Error())})
		return
	}

	deploymentSchemas, err := converters.ToDeploymentSchemas(ctx, deployments)
	if err != nil {
		log.Error().Msgf("Failed to convert deployment model list to schema: %s", err.Error())
		ctx.JSON(500, gin.H{"error": err.Error()})
		return
	}

	deploymentListSchema := &schemas.DeploymentListSchema{
		BaseListSchema: schemas.BaseListSchema{
			Total: total,
			Start: schema.Start,
			Count: schema.Count,
		},
		Items: deploymentSchemas,
	}

	ctx.JSON(200, deploymentListSchema)
}

func (c *deploymentController) ListDynamoNimDeployments(ctx *gin.Context) {
	var schema schemas.ListQuerySchema
	var getDynamoNimSchema schemas.GetDynamoNimSchema

	if err := ctx.ShouldBindUri(&getDynamoNimSchema); err != nil {
		ctx.JSON(400, gin.H{"error": err.Error()})
		return
	}

	if err := ctx.ShouldBindQuery(&schema); err != nil {
		ctx.JSON(400, gin.H{"error": err.Error()})
		return
	}

	ownership, err := GetOwnershipInfo(ctx)
	if err != nil {
		ctx.JSON(500, gin.H{"error": err.Error()})
		return
	}

	deploymentOpt := services.ListDeploymentOption{
		BaseListOption: services.BaseListOption{
			Start:  &schema.Start,
			Count:  &schema.Count,
			Search: schema.Search,
		},
		DynamoNimName: &getDynamoNimSchema.DynamoNimName,
	}

	setListDeploymentOptionsScope(&deploymentOpt, ownership)

	deployments, total, err := services.DeploymentService.List(ctx, deploymentOpt)
	if err != nil {
		log.Error().Msgf("Could not find deployments for the dynamo nim %s with the following opts %+v: %s", getDynamoNimSchema.DynamoNimName, deploymentOpt, err.Error())
		ctx.JSON(500, gin.H{"error": fmt.Sprintf("Could not find deployments %s", err.Error())})
		return
	}

	deploymentSchemas, err := converters.ToDeploymentSchemas(ctx, deployments)
	if err != nil {
		log.Error().Msgf("Failed to convert deployment model list to schema: %s", err.Error())
		ctx.JSON(500, gin.H{"error": err.Error()})
		return
	}

	deploymentListSchema := &schemas.DeploymentListSchema{
		BaseListSchema: schemas.BaseListSchema{
			Total: total,
			Start: schema.Start,
			Count: schema.Count,
		},
		Items: deploymentSchemas,
	}

	ctx.JSON(200, deploymentListSchema)
}

func (c *deploymentController) ListDynamoNimVersionDeployments(ctx *gin.Context) {
	var schema schemas.ListQuerySchema
	var getDynamoNimVersionSchema schemas.GetDynamoNimVersionSchema

	if err := ctx.ShouldBindUri(&getDynamoNimVersionSchema); err != nil {
		ctx.JSON(400, gin.H{"error": err.Error()})
		return
	}

	if err := ctx.ShouldBindQuery(&schema); err != nil {
		ctx.JSON(400, gin.H{"error": err.Error()})
		return
	}

	ownership, err := GetOwnershipInfo(ctx)
	if err != nil {
		ctx.JSON(500, err.Error())
		return
	}

	deploymentOpt := services.ListDeploymentOption{
		BaseListOption: services.BaseListOption{
			Start:  &schema.Start,
			Count:  &schema.Count,
			Search: schema.Search,
		},
		DynamoNimTag: getDynamoNimVersionSchema.Tag(),
	}

	setListDeploymentOptionsScope(&deploymentOpt, ownership)

	deployments, total, err := services.DeploymentService.List(ctx, deploymentOpt)
	if err != nil {
		log.Error().Msgf("Could not find deployments for the dynamo nim version %s with the following opts %+v: %s", *getDynamoNimVersionSchema.Tag(), deploymentOpt, err.Error())
		ctx.JSON(500, gin.H{"error": fmt.Sprintf("Could not find deployments %s", err.Error())})
		return
	}

	deploymentSchemas, err := converters.ToDeploymentSchemas(ctx, deployments)
	if err != nil {
		log.Error().Msgf("Failed to convert deployment model list to schema: %s", err.Error())
		ctx.JSON(500, gin.H{"error": err.Error()})
		return
	}

	deploymentListSchema := &schemas.DeploymentListSchema{
		BaseListSchema: schemas.BaseListSchema{
			Total: total,
			Start: schema.Start,
			Count: schema.Count,
		},
		Items: deploymentSchemas,
	}

	ctx.JSON(200, deploymentListSchema)
}

func (c *deploymentController) ListDeployments(ctx *gin.Context) {
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

	listOpt := services.ListDeploymentOption{
		BaseListOption: services.BaseListOption{
			Start:  &schema.Start,
			Count:  &schema.Count,
			Search: schema.Search,
		},
	}

	setListDeploymentOptionsScope(&listOpt, ownership)

	deployments, total, err := services.DeploymentService.List(ctx, listOpt)
	if err != nil {
		log.Error().Msgf("Could not get all deployments for the cluster with the following opts %+v: %s", listOpt, err.Error())
		ctx.JSON(500, gin.H{"error": fmt.Sprintf("Could not find deployments %s", err.Error())})
		return
	}

	deploymentSchemas, err := converters.ToDeploymentSchemas(ctx, deployments)
	if err != nil {
		log.Error().Msgf("Failed to convert deployment model list to schema: %s", err.Error())
		ctx.JSON(500, gin.H{"error": err.Error()})
		return
	}

	deploymentListSchema := &schemas.DeploymentListSchema{
		BaseListSchema: schemas.BaseListSchema{
			Total: total,
			Start: schema.Start,
			Count: schema.Count,
		},
		Items: deploymentSchemas,
	}

	ctx.JSON(200, deploymentListSchema)
}

func (c *deploymentController) CreationJSONSchema(ctx *gin.Context) {
	reflector := jsonschema.Reflector{}
	res := reflector.Reflect(schemas.CreateDeploymentSchema{})
	if res != nil {
		res.Version = "http://json-schema.org/draft-04/schema#"
	}
	ctx.JSON(200, res)
}

func getDeployment(ctx *gin.Context, s *schemas.GetDeploymentSchema) (*models.Deployment, error) {
	cluster, err := ClusterController.GetCluster(ctx, s.ClusterName)

	if err != nil {
		return nil, err
	}

	ownership, err := GetOwnershipInfo(ctx)
	if err != nil {
		return nil, err
	}

	var deployment *models.Deployment
	switch env.ApplicationScope {
	case env.UserScope:
		deployment, err = services.DeploymentService.GetByNameAndCreator(
			ctx, cluster.ID, s.KubeNamespace, s.DeploymentName, ownership.UserId,
		)
		if err != nil {
			return nil, err
		}
	case env.OrganizationScope:
		deployment, err = services.DeploymentService.GetByName(
			ctx, cluster.ID, s.KubeNamespace, s.DeploymentName,
		)
		if err != nil {
			return nil, err
		}
	default:
		return nil, fmt.Errorf("unknown application scope: %s", env.ApplicationScope)
	}

	if err != nil {
		return nil, err
	}

	return deployment, nil
}

// The start of the V2 deployment APIs
func (c *deploymentController) CreateV2(ctx *gin.Context) {
	_, kubeNamespace, err := getClusterAndInfo(ctx)
	if err != nil {
		return // ctx set in helper function
	}

	var schema schemasv2.CreateDeploymentSchema
	if err := ctx.ShouldBindJSON(&schema); err != nil {
		ctx.JSON(400, gin.H{"error": err.Error()})
		return
	}

	dynamoNim, dynamoNimVersion, err := parseDynamoNimVersion(schema.DynamoNim)
	if err != nil {
		ctx.JSON(400, gin.H{"error": err.Error()})
		return
	}

	// Determine the deployment name
	deploymentName := schema.Name
	if deploymentName == "" {
		deploymentName = fmt.Sprintf("dep-%s-%s--%s", dynamoNim, dynamoNimVersion, uuid.New().String())
		deploymentName = deploymentName[:63] // Max label length for k8s
	}
	log.Info().Msgf("Creating deployment with name: %s", deploymentName)

	// Get ownership info for labels
	ownership, err := GetOwnershipInfo(ctx)
	if err != nil {
		ctx.JSON(500, gin.H{"error": err.Error()})
		return
	}

	// Create DynamoDeployment CR
	dynamoDeployment := &dynamov1alpha1.DynamoDeployment{
		TypeMeta: metav1.TypeMeta{
			APIVersion: "nvidia.com/v1alpha1",
			Kind:       "DynamoDeployment",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:      deploymentName,
			Namespace: kubeNamespace,
			Labels: map[string]string{
				"ngc-organization": ownership.OrganizationId,
				"ngc-user":         ownership.UserId,
			},
		},
		Spec: dynamov1alpha1.DynamoDeploymentSpec{
			DynamoNim: schema.DynamoNim,
			Services:  make(map[string]*dynamov1alpha1.DynamoNimDeployment),
		},
	}

	// Create the DynamoDeployment CR
	log.Info().Msgf("Creating DynamoDeployment CR: %+v", dynamoDeployment)
	err = services.K8sService.CreateDynamoDeployment(ctx, dynamoDeployment)
	if err != nil {
		log.Error().Msgf("Failed to create DynamoDeployment CR: %s", err.Error())
		ctx.JSON(500, gin.H{"error": fmt.Sprintf("Failed to create DynamoDeployment CR: %v", err)})
		return
	}

	// Return success response
	ctx.JSON(200, gin.H{
		"status":    "success",
		"message":   "Deployment created successfully",
		"name":      deploymentName,
		"namespace": kubeNamespace,
		"dynamoNim": schema.DynamoNim,
		"ingress":   fmt.Sprintf("https://%s.%s.%s", deploymentName, kubeNamespace, "compoundai.cloud"),
	})
}

func (c *deploymentController) GetV2(ctx *gin.Context) {
	cluster, deployment, err := getDeploymentAndInfo(ctx)
	if err != nil {
		return // ctx set in helper function
	}

	// Getting mocked creator
	creator := mocks.DefaultUser()

	deploymentSchema, err := converters.ToDeploymentSchemaV2(ctx, cluster, deployment, creator)
	if err != nil {
		ctx.JSON(500, gin.H{"error": err.Error()})
		return
	}
	ctx.JSON(200, deploymentSchema)
}

func (c *deploymentController) UpdateV2(ctx *gin.Context) {
	cluster, deployment, err := getDeploymentAndInfo(ctx)
	if err != nil {
		return // ctx set in helper function
	}

	var schema schemasv2.UpdateDeploymentSchema
	if err := ctx.ShouldBindJSON(&schema); err != nil {
		ctx.JSON(400, gin.H{"error": err.Error()})
		return
	}

	ownership, err := GetOwnershipInfo(ctx)
	if err != nil {
		ctx.JSON(500, gin.H{"error": err.Error()})
		return
	}

	// Getting the k8s namespace
	kubeNamespace := os.Getenv("DEFAULT_KUBE_NAMESPACE")
	if kubeNamespace == "" {
		kubeNamespace = "dynamo"
	}

	dynamoNim, dynamoNimVersion, err := parseDynamoNimVersion(schema.DynamoNim)
	if err != nil {
		ctx.JSON(400, gin.H{"error": err.Error()})
		return
	}

	createDeploymentTarget, err := c.buildDeploymentTargetConfiguration(&schema, dynamoNim, dynamoNimVersion)
	if err != nil {
		log.Error().Msgf("Failed to build createDeploymentTarget schema %s", err.Error())
		ctx.JSON(400, gin.H{"error": err.Error()})
		return
	}

	updateDeploymentSchema := schemas.UpdateDeploymentSchema{
		Targets: []*schemas.CreateDeploymentTargetSchema{
			createDeploymentTarget,
		},
	}

	tx, ctx_, df, err := database.DatabaseUtil.StartTransaction(ctx)
	defer func() { df(err) }() // Clean up the transaction

	deployment, err = c.updateDeploymentEntities(ctx_, updateDeploymentSchema, deployment, ownership)
	if err != nil {
		log.Error().Msgf("Could not update deployment entities %s: %s", deployment.Name, err.Error())
		ctx.JSON(500, err.Error())
		return
	}

	tx.Commit()

	// Getting mocked creator
	creator := mocks.DefaultUser()

	deploymentSchema, err := converters.ToDeploymentSchemaV2(ctx, cluster, deployment, creator)
	if err != nil {
		ctx.JSON(500, gin.H{"error": err.Error()})
		return
	}
	ctx.JSON(200, deploymentSchema)
}

func parseDynamoNimVersion(dynamoNimTag string) (string, string, error) {
	var dynamoNim, dynamoNimVersion string
	dynamoNimParts := strings.Split(dynamoNimTag, ":")
	if len(dynamoNimParts) == 2 {
		dynamoNim = dynamoNimParts[0]
		dynamoNimVersion = dynamoNimParts[1]
	} else {
		return "", "", fmt.Errorf("invalid Dynamo Nim format, expected 'dynamonim:version'")
	}
	fmt.Println("Dynamo Nim:", dynamoNim)
	fmt.Println("Dynamo Nim Version:", dynamoNimVersion)

	return dynamoNim, dynamoNimVersion, nil
}

func (c *deploymentController) buildDeploymentTargetConfiguration(schema *schemasv2.UpdateDeploymentSchema, dynamoNim, dynamoNimVersion string) (*schemas.CreateDeploymentTargetSchema, error) {
	// Extract the first service from Services map
	var firstServiceSpec schemasv2.ServiceSpec
	for _, serviceSpec := range schema.Services {
		firstServiceSpec = serviceSpec
		break
	}

	hpaMinReplica := int32(firstServiceSpec.Scaling.MinReplicas)
	hpaMaxRepica := int32(firstServiceSpec.Scaling.MaxReplicas)
	enableIngress := false

	// Convert service configuration into CreateDeploymentTargetSchema
	createDeploymentTarget := &schemas.CreateDeploymentTargetSchema{
		DynamoNim: dynamoNim,
		Version:   dynamoNimVersion,
		Config: &schemas.DeploymentTargetConfig{
			HPAConf: &schemas.DeploymentTargetHPAConf{
				MinReplicas: &hpaMinReplica,
				MaxReplicas: &hpaMaxRepica,
			},
			Resources: &schemas.Resources{
				Requests: &schemas.ResourceItem{
					CPU:    firstServiceSpec.ConfigOverrides.Resources.Requests.CPU,
					GPU:    firstServiceSpec.ConfigOverrides.Resources.Requests.GPU,
					Memory: firstServiceSpec.ConfigOverrides.Resources.Requests.Memory,
				},
				Limits: &schemas.ResourceItem{
					CPU:    firstServiceSpec.ConfigOverrides.Resources.Limits.CPU,
					GPU:    firstServiceSpec.ConfigOverrides.Resources.Limits.GPU,
					Memory: firstServiceSpec.ConfigOverrides.Resources.Limits.Memory,
				},
			},
			DeploymentOverrides: &schemas.DeploymentOverrides{
				ColdStartTimeout: firstServiceSpec.ColdStartTimeout,
			},
			// Assuming Envs, Runners, EnableIngress, DeploymentStrategy are default values or nil
			EnableIngress:      &enableIngress, // Assuming false as default
			DeploymentStrategy: nil,            // Assuming no specific strategy as default
			ExternalServices:   firstServiceSpec.ExternalServices,
		},
	}

	return createDeploymentTarget, nil
}

func (c *deploymentController) TerminateV2(ctx *gin.Context) {
	cluster, deployment, err := getDeploymentAndInfo(ctx)
	if err != nil {
		return // ctx set in helper function
	}

	deployment, err = c.doTerminate(ctx, deployment)
	if err != nil {
		ctx.JSON(500, gin.H{"error": err.Error()})
		return
	}

	// Getting mocked creator
	creator := mocks.DefaultUser()
	deploymentSchema, err := converters.ToDeploymentSchemaV2(ctx, cluster, deployment, creator)
	if err != nil {
		ctx.JSON(500, gin.H{"error": err.Error()})
		return
	}
	ctx.JSON(200, deploymentSchema)
}

func (c *deploymentController) DeleteV2(ctx *gin.Context) {
	cluster, deployment, err := getDeploymentAndInfo(ctx)
	if err != nil {
		return // ctx set in helper function
	}

	deployment, err = services.DeploymentService.Delete(ctx, deployment)
	if err != nil {
		log.Error().Msgf("Could not delete deployment with the name %s: %s", deployment.Name, err.Error())
		ctx.JSON(500, gin.H{"error": fmt.Sprintf("Could not delete deployment with the name %s", deployment.Name)})
		return
	}

	// Getting mocked creator
	creator := mocks.DefaultUser()
	deploymentSchema, err := converters.ToDeploymentSchemaV2(ctx, cluster, deployment, creator)
	if err != nil {
		ctx.JSON(500, gin.H{"error": err.Error()})
		return
	}
	ctx.JSON(200, deploymentSchema)
}

func getDeploymentAndInfo(ctx *gin.Context) (*models.Cluster, *models.Deployment, error) {
	cluster, kubeNamespace, err := getClusterAndInfo(ctx)
	if err != nil {
		return nil, nil, err
	}

	var getSchema schemasv2.GetDeploymentSchema
	if err := ctx.ShouldBindUri(&getSchema); err != nil {
		ctx.JSON(400, gin.H{"error": err.Error()})
		return nil, nil, err
	}

	deployment, err := getDeployment(ctx, getSchema.ToV1(cluster.Name, kubeNamespace))
	if err != nil {
		log.Error().Msgf("Could not find deployment with the name %s: %s", getSchema.DeploymentName, err.Error())
		ctx.JSON(404, fmt.Sprintf("Could not find deployment with the name %s", getSchema.DeploymentName))
		return nil, nil, err
	}

	return cluster, deployment, err
}

func getClusterAndInfo(ctx *gin.Context) (*models.Cluster, string, error) {
	clusterName := ctx.Query("cluster")
	if clusterName == "" {
		clusterName = "default"
	}
	log.Info().Msgf("Got clusterName: %s", clusterName)

	cluster, err := ClusterController.GetCluster(ctx, clusterName)
	if err != nil {
		log.Error().Msgf("Could not find cluster with the name %s: %s", clusterName, err.Error())
		ctx.JSON(404, gin.H{"error": fmt.Sprintf("Could not find cluster with the name %s", clusterName)})
		return nil, "", err
	}

	// Getting the k8s namespace
	kubeNamespace := os.Getenv("DEFAULT_KUBE_NAMESPACE")
	if kubeNamespace == "" {
		kubeNamespace = "dynamo"
	}

	return cluster, kubeNamespace, nil
}
