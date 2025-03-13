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
	"errors"
	"strings"

	"github.com/ai-dynamo/dynamo/deploy/dynamo/api-server/api/common/consts"
	"github.com/ai-dynamo/dynamo/deploy/dynamo/api-server/api/database"
	"github.com/ai-dynamo/dynamo/deploy/dynamo/api-server/api/models"
	"k8s.io/apimachinery/pkg/util/validation"

	"github.com/rs/zerolog/log"

	"gorm.io/gorm"
)

type clusterService struct{}

var ClusterService = clusterService{}

type CreateClusterOption struct {
	CreatorId      string
	OrganizationId string
	Name           string
	Description    string
	KubeConfig     string
}

type UpdateClusterOption struct {
	Description *string
	KubeConfig  *string
}

type ListClusterOption struct {
	BaseListOption
	OrganizationId *string
	Ids            *[]uint
	Names          *[]string
	CreatorIds     *[]uint
	Order          *string
}

func (s *clusterService) Create(ctx context.Context, opt CreateClusterOption) (*models.Cluster, error) {
	errs := validation.IsDNS1035Label(opt.Name)
	if len(errs) > 0 {
		return nil, errors.New(strings.Join(errs, ";"))
	}

	db := s.getDB(ctx)
	log.Info().Msg("Starting create cluster transaction")

	cluster := models.Cluster{
		Resource: models.Resource{
			Name: opt.Name,
		},
		OrganizationAssociate: models.OrganizationAssociate{
			OrganizationId: opt.OrganizationId,
		},
		CreatorAssociate: models.CreatorAssociate{
			UserId: opt.CreatorId,
		},
		Description: opt.Description,
		KubeConfig:  opt.KubeConfig,
	}

	if err := db.Create(&cluster).Error; err != nil {
		return nil, err
	}

	log.Info().Msg("Finished create cluster transaction")

	return &cluster, nil
}

func (s *clusterService) Update(ctx context.Context, c *models.Cluster, opt UpdateClusterOption) (*models.Cluster, error) {
	var err error
	updaters := make(map[string]interface{})

	if opt.Description != nil {
		updaters["description"] = *opt.Description
		defer func() {
			if err == nil {
				c.Description = *opt.Description
			}
		}()
	}
	if opt.KubeConfig != nil {
		updaters["kube_config"] = *opt.KubeConfig
		defer func() {
			if err == nil {
				c.KubeConfig = *opt.KubeConfig
			}
		}()
	}

	if len(updaters) == 0 {
		return c, nil
	}

	db := s.getDB(ctx)

	log.Info().Msgf("Updating cluster with updaters: %+v", updaters)

	err = db.Where("id = ?", c.ID).Updates(updaters).Error
	if err != nil {
		log.Error().Msgf("Failed to update cluster: %s", err.Error())

		return nil, err
	}

	return c, err
}

func (s *clusterService) Get(ctx context.Context, id uint) (*models.Cluster, error) {
	var cluster models.Cluster
	db := s.getDB(ctx)
	err := db.Where("id = ?", id).First(&cluster).Error
	if err != nil {
		log.Error().Msgf("Failed to get cluster by id %d: %s", id, err.Error())
		return nil, err
	}
	if cluster.ID == 0 {
		return nil, consts.ErrNotFound
	}
	return &cluster, nil
}

func (s *clusterService) GetByUid(ctx context.Context, uid string) (*models.Cluster, error) {
	var cluster models.Cluster
	db := s.getDB(ctx)
	err := db.Where("uid = ?", uid).First(&cluster).Error
	if err != nil {
		log.Error().Msgf("Failed to get cluster by uid %s: %s", uid, err.Error())
		return nil, err
	}
	if cluster.ID == 0 {
		return nil, consts.ErrNotFound
	}
	return &cluster, nil
}

func (s *clusterService) GetByName(ctx context.Context, organizationId string, name string) (*models.Cluster, error) {
	var cluster models.Cluster
	db := s.getDB(ctx)
	err := db.Where("organization_id = ?", organizationId).Where("name = ?", name).First(&cluster).Error

	if err != nil {
		log.Error().Msgf("Failed to get cluster by name %s: %s", name, err.Error())

		return nil, err
	}
	if cluster.ID == 0 {
		return nil, consts.ErrNotFound
	}
	return &cluster, nil
}

func (s *clusterService) GetIdByName(ctx context.Context, organizationId uint, name string) (uint, error) {
	var cluster models.Cluster
	db := s.getDB(ctx)
	err := db.Select("id").Where("organization_id = ?", organizationId).Where("name = ?", name).First(&cluster).Error
	return cluster.ID, err
}

func (s *clusterService) List(ctx context.Context, opt ListClusterOption) ([]*models.Cluster, uint, error) {
	clusters := make([]*models.Cluster, 0)

	query := s.getDB(ctx)

	if opt.Ids != nil {
		if len(*opt.Ids) == 0 {
			return clusters, 0, nil
		}
		query = query.Where("id in (?)", *opt.Ids)
	}
	if opt.Names != nil {
		if len(*opt.Names) == 0 {
			return clusters, 0, nil
		}
		query = query.Where("name in (?)", *opt.Names)
	}
	if opt.OrganizationId != nil {
		query = query.Where("organization_id = ?", *opt.OrganizationId)
	}
	var total int64
	err := query.Count(&total).Error
	if err != nil {
		return nil, 0, err
	}
	query = opt.BindQueryWithLimit(query)
	if opt.Ids == nil {
		if opt.Order == nil {
			query = query.Order("id DESC")
		} else {
			query = query.Order(*opt.Order)
		}
	}
	err = query.Find(&clusters).Error
	if err != nil {
		return nil, 0, err
	}
	return clusters, uint(total), err
}

func (s *clusterService) getDB(ctx context.Context) *gorm.DB {
	db := database.DatabaseUtil.GetDBSession(ctx).Model(&models.Cluster{})
	return db
}
