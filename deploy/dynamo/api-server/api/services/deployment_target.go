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

package services

import (
	"context"
	"fmt"

	"github.com/dynemo-ai/dynemo/deploy/dynamo/api-server/api/common/consts"
	"github.com/dynemo-ai/dynemo/deploy/dynamo/api-server/api/database"
	"github.com/dynemo-ai/dynemo/deploy/dynamo/api-server/api/models"
	"github.com/dynemo-ai/dynemo/deploy/dynamo/api-server/api/schemas"
	"github.com/rs/zerolog/log"
	"gorm.io/gorm"
)

type deploymentTargetService struct{}

var DeploymentTargetService = deploymentTargetService{}

type CreateDeploymentTargetOption struct {
	CreatorId             string
	DeploymentId          uint
	DeploymentRevisionId  uint
	DynamoNimVersionId  string
	DynamoNimVersionTag string
	Config                *schemas.DeploymentTargetConfig
}

type UpdateDeploymentTargetOption struct {
	Config **schemas.DeploymentTargetConfig
}

type ListDeploymentTargetOption struct {
	BaseListOption
	DeploymentRevisionStatus *schemas.DeploymentRevisionStatus
	DeploymentId             *uint
	DeploymentIds            *[]uint
	DeploymentRevisionId     *uint
	DeploymentRevisionIds    *[]uint
	Type                     *schemas.DeploymentTargetType
}

func (s *deploymentTargetService) Create(ctx context.Context, opt CreateDeploymentTargetOption) (*models.DeploymentTarget, error) {
	if opt.Config == nil {
		defaultCPU := int32(80)
		defaultGPU := int32(80)
		defaultMinReplicas := int32(2)
		defaultMaxReplicas := int32(10)

		opt.Config = &schemas.DeploymentTargetConfig{
			Resources: &schemas.Resources{
				Requests: &schemas.ResourceItem{
					CPU:    "500m",
					Memory: "1G",
				},
				Limits: &schemas.ResourceItem{
					CPU:    "1000m",
					Memory: "2G",
				},
			},
			HPAConf: &schemas.DeploymentTargetHPAConf{
				CPU:         &defaultCPU,
				GPU:         &defaultGPU,
				MinReplicas: &defaultMinReplicas,
				MaxReplicas: &defaultMaxReplicas,
			},
		}
	}
	deploymentTarget := models.DeploymentTarget{
		CreatorAssociate: models.CreatorAssociate{
			UserId: opt.CreatorId,
		},
		DeploymentAssociate: models.DeploymentAssociate{
			DeploymentId: opt.DeploymentId,
		},
		DeploymentRevisionAssociate: models.DeploymentRevisionAssociate{
			DeploymentRevisionId: opt.DeploymentRevisionId,
		},
		DynamoNimVersionAssociate: models.DynamoNimVersionAssociate{
			DynamoNimVersionId:  opt.DynamoNimVersionId,
			DynamoNimVersionTag: opt.DynamoNimVersionTag,
		},
		Config: opt.Config,
	}
	err := s.getDB(ctx).Create(&deploymentTarget).Error
	if err != nil {
		return nil, err
	}
	return &deploymentTarget, err
}

func (s *deploymentTargetService) Get(ctx context.Context, id uint) (*models.DeploymentTarget, error) {
	var deploymentTarget models.DeploymentTarget
	err := s.getDB(ctx).Where("id = ?", id).First(&deploymentTarget).Error
	if err != nil {
		log.Error().Msgf("Failed to get deployment revision by id %d: %s", id, err.Error())
		return nil, err
	}
	if deploymentTarget.ID == 0 {
		return nil, consts.ErrNotFound
	}
	return &deploymentTarget, nil
}

func (s *deploymentTargetService) GetByUid(ctx context.Context, uid string) (*models.DeploymentTarget, error) {
	var deploymentTarget models.DeploymentTarget
	err := s.getDB(ctx).Where("uid = ?", uid).First(&deploymentTarget).Error
	if err != nil {
		log.Error().Msgf("Failed to get deployment revision by uid %s: %s", uid, err.Error())
		return nil, err
	}
	if deploymentTarget.ID == 0 {
		return nil, consts.ErrNotFound
	}
	return &deploymentTarget, nil
}

func (s *deploymentTargetService) List(ctx context.Context, opt ListDeploymentTargetOption) ([]*models.DeploymentTarget, uint, error) {
	query := s.getDB(ctx)
	if opt.DeploymentRevisionStatus != nil {
		query = query.Joins("INNER JOIN deployment_revision ON deployment_revision.id = deployment_target.deployment_revision_id and deployment_revision.status = ?", *opt.DeploymentRevisionStatus)
	}
	if opt.DeploymentId != nil {
		query = query.Where("deployment_target.deployment_id = ?", *opt.DeploymentId)
	}
	if opt.DeploymentRevisionId != nil {
		query = query.Where("deployment_target.deployment_revision_id = ?", *opt.DeploymentRevisionId)
	}
	if opt.DeploymentIds != nil {
		query = query.Where("deployment_target.deployment_id in (?)", *opt.DeploymentIds)
	}
	if opt.DeploymentRevisionIds != nil {
		query = query.Where("deployment_target.deployment_revision_id in (?)", *opt.DeploymentRevisionIds)
	}
	if opt.Type != nil {
		query = query.Where("deployment_target.type = ?", *opt.Type)
	}
	var total int64
	err := query.Count(&total).Error
	if err != nil {
		return nil, 0, err
	}
	deploymentTargets := make([]*models.DeploymentTarget, 0)
	query = opt.BindQueryWithLimit(query)
	err = query.Order("deployment_target.id ASC").Find(&deploymentTargets).Error
	if err != nil {
		return nil, 0, err
	}
	return deploymentTargets, uint(total), err
}

func (s *deploymentTargetService) Update(ctx context.Context, b *models.DeploymentTarget, opt UpdateDeploymentTargetOption) (*models.DeploymentTarget, error) {
	var err error
	updaters := make(map[string]interface{})

	if opt.Config != nil {
		updaters["config"] = *opt.Config
		defer func() {
			if err == nil {
				b.Config = *opt.Config
			}
		}()
	}

	if len(updaters) == 0 {
		return b, nil
	}

	log.Info().Msgf("Updating deployment target with updaters: %+v", updaters)

	err = s.getDB(ctx).Where("id = ?", b.ID).Updates(updaters).Error

	return b, err
}

func (s *deploymentTargetService) Deploy(ctx context.Context, deploymentTarget *models.DeploymentTarget, deployOption *models.DeployOption, ownership *schemas.OwnershipSchema) (*models.DeploymentTarget, error) {

	err := s.getDB(ctx).Where("id = ?", deploymentTarget.ID).Save(deploymentTarget).Error
	if err != nil {
		deleteErr := DeploymentManagementService.Delete(ctx, deploymentTarget)
		if deleteErr != nil {
			log.Error().Msg("Failed to clean up kube resources for erroneous deployment")
		}

		err = fmt.Errorf("failed to update deploymentTarget after creating kube resources: %s", err.Error())
		return nil, err
	}

	return deploymentTarget, nil
}

func (s *deploymentTargetService) Terminate(ctx context.Context, deploymentTarget *models.DeploymentTarget) (*models.DeploymentTarget, error) {
	err := DeploymentManagementService.Delete(ctx, deploymentTarget)
	if err != nil {
		log.Error().Msgf("Failed to terminate kube resources for deployment target %s\n", deploymentTarget.DynamoNimVersionTag)
		return nil, err
	}

	return deploymentTarget, nil
}

func (s *deploymentTargetService) getDB(ctx context.Context) *gorm.DB {
	db := database.DatabaseUtil.GetDBSession(ctx).Model(&models.DeploymentTarget{})
	return db
}
