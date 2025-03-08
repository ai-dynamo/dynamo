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

	"github.com/dynemo-ai/dynemo/deploy/dynamo/api-server/api/common/consts"
	"github.com/dynemo-ai/dynemo/deploy/dynamo/api-server/api/database"
	"github.com/dynemo-ai/dynemo/deploy/dynamo/api-server/api/models"
	"github.com/dynemo-ai/dynemo/deploy/dynamo/api-server/api/schemas"
	"github.com/rs/zerolog/log"
	"gorm.io/gorm"
)

type deploymentRevisionService struct{}

var DeploymentRevisionService = deploymentRevisionService{}

type CreateDeploymentRevisionOption struct {
	CreatorId    string
	DeploymentId uint
	Status       schemas.DeploymentRevisionStatus
}

type UpdateDeploymentRevisionOption struct {
	Status *schemas.DeploymentRevisionStatus
}

type ListDeploymentRevisionOption struct {
	BaseListOption
	DeploymentId  *uint
	DeploymentIds *[]uint
	Ids           *[]uint
	Status        *schemas.DeploymentRevisionStatus
}

func (s *deploymentRevisionService) Create(ctx context.Context, opt CreateDeploymentRevisionOption) (*models.DeploymentRevision, error) {
	deploymentRevision := models.DeploymentRevision{
		CreatorAssociate: models.CreatorAssociate{
			UserId: opt.CreatorId,
		},
		DeploymentAssociate: models.DeploymentAssociate{
			DeploymentId: opt.DeploymentId,
		},
		Status: opt.Status,
	}
	err := s.getDB(ctx).Create(&deploymentRevision).Error
	if err != nil {
		return nil, err
	}
	return &deploymentRevision, err
}

func (s *deploymentRevisionService) Update(ctx context.Context, deploymentRevision *models.DeploymentRevision, opt UpdateDeploymentRevisionOption) (*models.DeploymentRevision, error) {
	var err error
	updaters := make(map[string]interface{})
	if opt.Status != nil {
		updaters["status"] = *opt.Status
		defer func() {
			if err == nil {
				deploymentRevision.Status = *opt.Status
			}
		}()
	}

	if len(updaters) == 0 {
		return deploymentRevision, nil
	}

	log.Info().Msgf("Updating deployment revision with updaters: %+v", updaters)

	err = s.getDB(ctx).Where("id = ?", deploymentRevision.ID).Updates(updaters).Error
	if err != nil {
		log.Error().Msgf("Failed to update deployment revision: %s", err.Error())
		return nil, err
	}

	return deploymentRevision, err
}

func (s *deploymentRevisionService) Get(ctx context.Context, id uint) (*models.DeploymentRevision, error) {
	var deploymentRevision models.DeploymentRevision
	err := s.getDB(ctx).Where("id = ?", id).First(&deploymentRevision).Error
	if err != nil {
		log.Error().Msgf("Failed to get deployment revision by id %d: %s", id, err.Error())
		return nil, err
	}
	if deploymentRevision.ID == 0 {
		return nil, consts.ErrNotFound
	}
	return &deploymentRevision, nil
}

func (s *deploymentRevisionService) GetByUid(ctx context.Context, uid string) (*models.DeploymentRevision, error) {
	var deploymentRevision models.DeploymentRevision
	err := s.getDB(ctx).Where("uid = ?", uid).First(&deploymentRevision).Error
	if err != nil {
		log.Error().Msgf("Failed to get deployment revision by uid %s: %s", uid, err.Error())
		return nil, err
	}
	if deploymentRevision.ID == 0 {
		return nil, consts.ErrNotFound
	}
	return &deploymentRevision, nil
}

func (s *deploymentRevisionService) List(ctx context.Context, opt ListDeploymentRevisionOption) ([]*models.DeploymentRevision, uint, error) {
	query := s.getDB(ctx)
	if opt.DeploymentId != nil {
		query = query.Where("deployment_revision.deployment_id = ?", *opt.DeploymentId)
	}
	if opt.DeploymentIds != nil {
		query = query.Where("deployment_revision.deployment_id in (?)", *opt.DeploymentIds)
	}
	if opt.Status != nil {
		query = query.Where("deployment_revision.status = ?", *opt.Status)
	}
	if opt.Ids != nil {
		query = query.Where("deployment_revision.id in (?)", *opt.Ids)
	}
	query = query.Select("distinct(deployment_revision.*)")
	var total int64
	err := query.Count(&total).Error
	if err != nil {
		return nil, 0, err
	}
	deployments := make([]*models.DeploymentRevision, 0)
	query = opt.BindQueryWithLimit(query)
	err = query.Order("deployment_revision.id DESC").Find(&deployments).Error
	if err != nil {
		return nil, 0, err
	}
	return deployments, uint(total), err
}

func (s *deploymentRevisionService) GetDeployOption(ctx context.Context, deploymentRevision *models.DeploymentRevision, force bool) (*models.DeployOption, error) {
	deployOption := &models.DeployOption{
		Force: force,
	}
	return deployOption, nil
}

func (s *deploymentRevisionService) Terminate(ctx context.Context, deploymentRevision *models.DeploymentRevision) (err error) {
	deploymentTargets, _, err := DeploymentTargetService.List(ctx, ListDeploymentTargetOption{
		DeploymentRevisionId: &deploymentRevision.ID,
	})
	if err != nil {
		log.Error().Msgf("Failed to fetch deployment targets when terminating revision: %s", err.Error())
	}

	for _, target := range deploymentTargets {
		_, err := DeploymentTargetService.Terminate(ctx, target)
		if err != nil {
			log.Error().Msgf("Error occurred when terminating targets for revision: %s", err.Error())
			return err
		}
	}

	status := schemas.DeploymentRevisionStatusInactive
	_, err = s.Update(ctx, deploymentRevision, UpdateDeploymentRevisionOption{
		Status: &status,
	})
	if err != nil {
		log.Error().Msgf("Failed to set revision status to inactive: %s", err.Error())
		return err
	}
	return nil
}

func (s *deploymentRevisionService) Deploy(ctx context.Context, deploymentRevision *models.DeploymentRevision, deploymentTargets []*models.DeploymentTarget, ownership *schemas.OwnershipSchema, force bool) (err error) {
	_, err = DeploymentService.Get(ctx, deploymentRevision.DeploymentId)
	if err != nil {
		return
	}

	deployOption, err := s.GetDeployOption(ctx, deploymentRevision, force)
	if err != nil {
		return
	}

	if len(deploymentTargets) == 0 {
		deploymentTargets, _, err = DeploymentTargetService.List(ctx, ListDeploymentTargetOption{
			DeploymentRevisionId: &deploymentRevision.ID,
		})
		if err != nil {
			return
		}
	}

	// Can not use goroutine here because of pgx transaction bug
	for _, deploymentTarget := range deploymentTargets {
		_, err = DeploymentTargetService.Deploy(ctx, deploymentTarget, deployOption, ownership)
		if err != nil {
			return
		}
	}

	return nil
}

func (s *deploymentRevisionService) getDB(ctx context.Context) *gorm.DB {
	db := database.DatabaseUtil.GetDBSession(ctx).Model(&models.DeploymentRevision{})
	return db
}
