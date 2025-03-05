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
	"strings"
	"time"

	"github.com/dynemo-ai/dynemo/deploy/compoundai/api-server/api/common/consts"
	"github.com/dynemo-ai/dynemo/deploy/compoundai/api-server/api/database"
	"github.com/dynemo-ai/dynemo/deploy/compoundai/api-server/api/models"
	"github.com/dynemo-ai/dynemo/deploy/compoundai/api-server/api/schemas"
	"github.com/pkg/errors"
	"gorm.io/gorm"
	"gorm.io/gorm/clause"
	"k8s.io/apimachinery/pkg/util/validation"
)

type compoundComponentService struct{}

var CompoundComponentService = compoundComponentService{}

type CreateCompoundComponentOption struct {
	CreatorId      uint
	OrganizationId uint
	ClusterId      uint
	Name           string
	Description    string
	Version        string
	KubeNamespace  string
	Manifest       *schemas.CompoundComponentManifestSchema
}

type UpdateCompoundComponentOption struct {
	Description       *string
	Version           *string
	LatestInstalledAt **time.Time
	LatestHeartbeatAt **time.Time
	Manifest          **schemas.CompoundComponentManifestSchema
}

func (s *compoundComponentService) Create(ctx context.Context, opt CreateCompoundComponentOption) (*models.CompoundComponent, error) {
	errs := validation.IsDNS1035Label(opt.Name)
	if len(errs) > 0 {
		return nil, errors.New(strings.Join(errs, ";"))
	}

	errs = validation.IsDNS1035Label(opt.KubeNamespace)
	if len(errs) > 0 {
		return nil, errors.New(strings.Join(errs, ";"))
	}

	now := time.Now()

	compoundComponent := models.CompoundComponent{
		Resource: models.Resource{
			Name: opt.Name,
		},
		ClusterAssociate: models.ClusterAssociate{
			ClusterId: opt.ClusterId,
		},
		Description:       opt.Description,
		KubeNamespace:     opt.KubeNamespace,
		Manifest:          opt.Manifest,
		Version:           opt.Version,
		LatestInstalledAt: &now,
		LatestHeartbeatAt: &now,
	}
	err := s.getDB(ctx).Create(&compoundComponent).Error
	if err != nil {
		return nil, err
	}

	return &compoundComponent, err
}

func (s *compoundComponentService) Update(ctx context.Context, b *models.CompoundComponent, opt UpdateCompoundComponentOption) (*models.CompoundComponent, error) {
	var err error
	updaters := make(map[string]interface{})
	if opt.Description != nil {
		updaters["description"] = *opt.Description
		defer func() {
			if err == nil {
				b.Description = *opt.Description
			}
		}()
	}

	if opt.LatestHeartbeatAt != nil {
		updaters["latest_heartbeat_at"] = *opt.LatestHeartbeatAt
		defer func() {
			if err == nil {
				b.LatestHeartbeatAt = *opt.LatestHeartbeatAt
			}
		}()
	}

	if opt.LatestInstalledAt != nil {
		updaters["latest_installed_at"] = *opt.LatestInstalledAt
		defer func() {
			if err == nil {
				b.LatestInstalledAt = *opt.LatestInstalledAt
			}
		}()
	}

	if opt.Version != nil {
		updaters["version"] = *opt.Version
		defer func() {
			if err == nil {
				b.Version = *opt.Version
			}
		}()
	}

	if opt.Manifest != nil {
		updaters["manifest"] = *opt.Manifest
		defer func() {
			if err == nil {
				b.Manifest = *opt.Manifest
			}
		}()
	}

	if len(updaters) == 0 {
		return b, nil
	}

	err = s.getDB(ctx).Where("id = ?", b.ID).Updates(updaters).Error
	if err != nil {
		return nil, err
	}

	return b, err
}

func (s *compoundComponentService) Get(ctx context.Context, id uint) (*models.CompoundComponent, error) {
	var compoundComponent models.CompoundComponent
	err := s.getDB(ctx).Preload(clause.Associations).Where("id = ?", id).First(&compoundComponent).Error
	if err != nil {
		return nil, err
	}
	if compoundComponent.ID == 0 {
		return nil, consts.ErrNotFound
	}
	return &compoundComponent, nil
}

func (s *compoundComponentService) GetByUid(ctx context.Context, uid string) (*models.CompoundComponent, error) {
	var compoundComponent models.CompoundComponent
	err := s.getDB(ctx).Preload(clause.Associations).Where("uid = ?", uid).First(&compoundComponent).Error
	if err != nil {
		return nil, err
	}
	if compoundComponent.ID == 0 {
		return nil, consts.ErrNotFound
	}
	return &compoundComponent, nil
}

func (s *compoundComponentService) GetByName(ctx context.Context, clusterId uint, name string) (*models.CompoundComponent, error) {
	var compoundComponent models.CompoundComponent
	err := s.getDB(ctx).Where("cluster_id = ?", clusterId).Where("name = ?", name).First(&compoundComponent).Error
	if err != nil {
		return nil, errors.Wrapf(err, "get compoundComponent %s", name)
	}
	if compoundComponent.ID == 0 {
		return nil, consts.ErrNotFound
	}
	return &compoundComponent, nil
}

func (s *compoundComponentService) ListByUids(ctx context.Context, uids []string) ([]*models.CompoundComponent, error) {
	compoundComponents := make([]*models.CompoundComponent, 0, len(uids))
	if len(uids) == 0 {
		return compoundComponents, nil
	}
	err := s.getDB(ctx).Preload(clause.Associations).Where("uid in (?)", uids).Find(&compoundComponents).Error
	return compoundComponents, err
}

type ListCompoundComponentOption struct {
	Ids            *[]uint `json:"ids"`
	ClusterId      *uint   `json:"cluster_id"`
	ClusterIds     *[]uint `json:"cluster_ids"`
	OrganizationId *uint   `json:"organization_id"`
}

func (s *compoundComponentService) List(ctx context.Context, opt ListCompoundComponentOption) ([]*models.CompoundComponent, error) {
	query := s.getDB(ctx).Preload(clause.Associations)
	if opt.OrganizationId != nil {
		query = query.Where("organization_id = ?", *opt.OrganizationId)
	}
	if opt.ClusterIds != nil {
		query = query.Where("cluster_id in (?)", *opt.ClusterIds)
	}
	if opt.ClusterId != nil {
		query = query.Where("cluster_id = ?", *opt.ClusterId)
	}
	if opt.Ids != nil {
		query = query.Where("id in (?)", *opt.Ids)
	}

	compoundComponents := make([]*models.CompoundComponent, 0)
	err := query.Find(&compoundComponents).Error
	if err != nil {
		return nil, err
	}
	return compoundComponents, err
}

func (s *compoundComponentService) getDB(ctx context.Context) *gorm.DB {
	db := database.DatabaseUtil.GetDBSession(ctx).Model(&models.CompoundComponent{})
	return db
}
