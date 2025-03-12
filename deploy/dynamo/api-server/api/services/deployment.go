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
	"strings"
	"time"

	"github.com/ai-dynamo/dynamo/deploy/dynamo/api-server/api/common/consts"
	"github.com/ai-dynamo/dynamo/deploy/dynamo/api-server/api/database"
	"github.com/ai-dynamo/dynamo/deploy/dynamo/api-server/api/models"
	"github.com/ai-dynamo/dynamo/deploy/dynamo/api-server/api/schemas"
	"github.com/google/uuid"
	"github.com/pkg/errors"
	"github.com/rs/zerolog/log"
	"gorm.io/gorm"
	apiv1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/util/validation"
)

type deploymentService struct{}

var DeploymentService = deploymentService{}

type CreateDeploymentOption struct {
	CreatorId     string
	ClusterId     uint
	Name          string
	Description   string
	KubeNamespace string
}

type UpdateDeploymentOption struct {
	Description *string
	Status      *schemas.DeploymentStatus
}

type UpdateDeploymentStatusOption struct {
	Status    *schemas.DeploymentStatus
	SyncingAt **time.Time
	UpdatedAt **time.Time
}

type ListDeploymentOption struct {
	BaseListOption
	ClusterId           *uint
	CreatorId           *string
	LastUpdaterId       *uint
	OrganizationId      *string
	ClusterIds          *[]string
	CreatorIds          *[]uint
	LastUpdaterIds      *[]uint
	OrganizationIds     *[]string
	Ids                 *[]uint
	DynamoNimVersionIds *[]uint
	Statuses            *[]schemas.DeploymentStatus
	Order               *string
	DynamoNimName       *string
	DynamoNimTag        *string
}

func (s *deploymentService) Create(ctx context.Context, opt CreateDeploymentOption) (*models.Deployment, error) {
	errs := validation.IsDNS1035Label(opt.Name)
	if len(errs) > 0 {
		return nil, errors.New(strings.Join(errs, ";"))
	}

	errs = validation.IsDNS1035Label(opt.KubeNamespace)
	if len(errs) > 0 {
		return nil, errors.New(strings.Join(errs, ";"))
	}

	guid := uuid.New()

	deployment := models.Deployment{
		Resource: models.Resource{
			Name: opt.Name,
		},
		ClusterAssociate: models.ClusterAssociate{
			ClusterId: opt.ClusterId,
		},
		CreatorAssociate: models.CreatorAssociate{
			UserId: opt.CreatorId,
		},
		Description:     opt.Description,
		Status:          schemas.DeploymentStatusNonDeployed,
		KubeDeployToken: guid.String(),
		KubeNamespace:   opt.KubeNamespace,
	}

	db := s.getDB(ctx)

	err := db.Create(&deployment).Error
	if err != nil {
		log.Error().Msgf("Failed to create deployment %s", err.Error())
		return nil, err
	}

	return &deployment, err
}

func (s *deploymentService) Update(ctx context.Context, b *models.Deployment, opt UpdateDeploymentOption) (*models.Deployment, error) {
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

	if opt.Status != nil {
		updaters["status"] = *opt.Status
		defer func() {
			if err == nil {
				b.Status = *opt.Status
			}
		}()
	}

	if len(updaters) == 0 {
		return b, nil
	}

	log.Info().Msgf("Updating deployment with updaters %+v", updaters)
	err = s.getDB(ctx).Where("id = ?", b.ID).Updates(updaters).Error
	if err != nil {
		return nil, err
	}

	return b, err
}

func (s *deploymentService) Get(ctx context.Context, id uint) (*models.Deployment, error) {
	var deployment models.Deployment
	err := s.getDB(ctx).Where("id = ?", id).First(&deployment).Error
	if err != nil {
		log.Error().Msgf("Failed to get deployment by id %d: %s", id, err.Error())
		return nil, err
	}
	if deployment.ID == 0 {
		return nil, consts.ErrNotFound
	}
	return &deployment, nil
}

func (s *deploymentService) GetByUid(ctx context.Context, uid string) (*models.Deployment, error) {
	var deployment models.Deployment
	err := s.getDB(ctx).Where("uid = ?", uid).First(&deployment).Error
	if err != nil {
		log.Error().Msgf("Failed to get deployment by uid %s: %s", uid, err.Error())
		return nil, err
	}
	if deployment.ID == 0 {
		return nil, consts.ErrNotFound
	}
	return &deployment, nil
}

func (s *deploymentService) GetByName(ctx context.Context, clusterId uint, kubeNamespace, name string) (*models.Deployment, error) {
	var deployment models.Deployment
	err := s.getDB(ctx).Where("cluster_id = ?", clusterId).Where("kube_namespace = ?", kubeNamespace).Where("name = ?", name).First(&deployment).Error
	if err != nil {
		log.Error().Msgf("Failed to get deployment by name and creator %s: %s", name, err.Error())
		return nil, err
	}
	if deployment.ID == 0 {
		return nil, consts.ErrNotFound
	}
	return &deployment, nil
}

func (s *deploymentService) GetByNameAndCreator(ctx context.Context, clusterId uint, kubeNamespace, name string, creatorId string) (*models.Deployment, error) {
	var deployment models.Deployment
	err := s.getDB(ctx).Where("cluster_id = ?", clusterId).Where("kube_namespace = ?", kubeNamespace).Where("name = ?", name).Where("user_id = ?", creatorId).First(&deployment).Error
	if err != nil {
		log.Error().Msgf("Failed to get deployment by name %s: %s", name, err.Error())
		return nil, err
	}
	if deployment.ID == 0 {
		return nil, consts.ErrNotFound
	}
	return &deployment, nil
}

func (s *deploymentService) Delete(ctx context.Context, deployment *models.Deployment) (*models.Deployment, error) {
	if deployment.Status != schemas.DeploymentStatusTerminated && deployment.Status != schemas.DeploymentStatusTerminating {
		return nil, errors.New("deployment is not terminated")
	}
	return deployment, s.getDB(ctx).Unscoped().Delete(deployment).Error
}

func (s *deploymentService) Terminate(ctx context.Context, deployment *models.Deployment) (*models.Deployment, error) {
	deployment, err := s.UpdateStatus(ctx, deployment, UpdateDeploymentStatusOption{
		Status: schemas.DeploymentStatusTerminating.Ptr(),
	})
	if err != nil {
		return nil, err
	}

	start := uint(0)
	count := uint(1)

	deploymentRevisions, _, err := DeploymentRevisionService.List(ctx, ListDeploymentRevisionOption{
		BaseListOption: BaseListOption{
			Start: &start,
			Count: &count,
		},
		DeploymentId: &deployment.ID,
		Status:       schemas.DeploymentRevisionStatusActive.Ptr(),
	})
	if err != nil {
		return nil, err
	}

	log.Info().Msgf("Fetched %d active deployment revisions to terminate", len(deploymentRevisions))
	for _, deploymentRevision := range deploymentRevisions {
		err = DeploymentRevisionService.Terminate(ctx, deploymentRevision)
		if err != nil {
			return nil, err
		}
	}

	_, err = s.SyncStatus(ctx, deployment)
	return deployment, err
}

func (s *deploymentService) UpdateStatus(ctx context.Context, deployment *models.Deployment, opt UpdateDeploymentStatusOption) (*models.Deployment, error) {
	updater := map[string]interface{}{}
	if opt.Status != nil {
		deployment.Status = *opt.Status
		updater["status"] = *opt.Status
	}
	if opt.SyncingAt != nil {
		deployment.StatusSyncingAt = *opt.SyncingAt
		updater["status_syncing_at"] = *opt.SyncingAt
	}
	if opt.UpdatedAt != nil {
		deployment.StatusUpdatedAt = *opt.UpdatedAt
		updater["status_updated_at"] = *opt.UpdatedAt
	}
	log.Info().Msgf("Updating deployment with updaters %+v", updater)
	err := s.getDB(ctx).Where("id = ?", deployment.ID).Updates(updater).Error
	return deployment, err
}

func (s *deploymentService) SyncStatus(ctx context.Context, d *models.Deployment) (schemas.DeploymentStatus, error) {
	now := time.Now()
	nowPtr := &now
	_, err := s.UpdateStatus(ctx, d, UpdateDeploymentStatusOption{
		SyncingAt: &nowPtr,
	})
	if err != nil {
		log.Error().Msgf("Failed to update sync time for deployment %s: %s", d.Name, err.Error())
		return d.Status, err
	}

	currentStatus, err := s.getStatusFromK8s(ctx, d)
	if err != nil {
		log.Error().Msgf("Failed to get deployment status from k8s for deployment %s: %s", d.Name, err.Error())
		return currentStatus, err
	}

	now = time.Now()
	nowPtr = &now
	_, err = s.UpdateStatus(ctx, d, UpdateDeploymentStatusOption{
		Status:    &currentStatus,
		UpdatedAt: &nowPtr,
	})
	if err != nil {
		return currentStatus, err
	}
	return currentStatus, nil
}

func (s *deploymentService) List(ctx context.Context, opt ListDeploymentOption) ([]*models.Deployment, uint, error) {
	query := s.getDB(ctx)

	if opt.Ids != nil {
		query = query.Where("deployment.id in (?)", *opt.Ids)
	}

	query = query.Joins("LEFT JOIN deployment_revision ON deployment_revision.deployment_id = deployment.id AND deployment_revision.status = ?", schemas.DeploymentRevisionStatusActive)
	joinOnDeploymentTargets := query.Joins("LEFT JOIN deployment_target ON deployment_target.deployment_revision_id = deployment_revision.id")
	if opt.DynamoNimName != nil {
		query = joinOnDeploymentTargets.Where("deployment_target.dynamo_nim_version_tag LIKE ?", *opt.DynamoNimName+":%")
	}
	if opt.DynamoNimTag != nil {
		query = joinOnDeploymentTargets.Where("deployment_target.dynamo_nim_version_tag = ?", *opt.DynamoNimTag)
	}
	if opt.DynamoNimVersionIds != nil {
		query = joinOnDeploymentTargets.Where("deployment_target.dynamo_nim_version_id IN (?)", *opt.DynamoNimVersionIds)
	}
	if opt.ClusterId != nil {
		query = query.Where("deployment.cluster_id = ?", *opt.ClusterId)
	}
	if opt.ClusterIds != nil {
		query = query.Where("deployment.cluster_id IN (?)", *opt.ClusterIds)
	}
	if opt.Statuses != nil {
		query = query.Where("deployment.status IN (?)", *opt.Statuses)
	}
	if opt.OrganizationId != nil {
		query = query.Joins("LEFT JOIN cluster ON cluster.id = deployment.cluster_id")
		query = query.Where("cluster.organization_id = ?", *opt.OrganizationId)
	}
	if opt.CreatorId != nil {
		query = query.Where("deployment.user_id = ?", *opt.CreatorId)
	}
	query = opt.BindQueryWithKeywords(query, "deployment")
	query = query.Select("deployment_revision.*, deployment.*")
	var total int64
	err := query.Count(&total).Error

	if err != nil {
		return nil, 0, err
	}
	query = opt.BindQueryWithLimit(query)
	if opt.Order != nil {
		query = query.Order(*opt.Order)
	} else {
		query.Order("deployment.id DESC")
	}
	deployments := make([]*models.Deployment, 0)
	err = query.Find(&deployments).Error
	if err != nil {
		return nil, 0, err
	}
	return deployments, uint(total), err
}

func (s *deploymentService) getDB(ctx context.Context) *gorm.DB {
	db := database.DatabaseUtil.GetDBSession(ctx).Model(&models.Deployment{})
	return db
}

func (s *deploymentService) getStatusFromK8s(ctx context.Context, d *models.Deployment) (schemas.DeploymentStatus, error) {
	defaultStatus := schemas.DeploymentStatusUnknown

	cluster, err := ClusterService.Get(ctx, d.ClusterId)
	if err != nil {
		return defaultStatus, err
	}

	namespace := d.KubeNamespace

	_, podLister, err := GetPodInformer(ctx, cluster, namespace)
	if err != nil {
		return defaultStatus, err
	}

	imageBuilderPods := make([]*apiv1.Pod, 0)

	status_ := schemas.DeploymentRevisionStatusActive
	deploymentRevisions, _, err := DeploymentRevisionService.List(ctx, ListDeploymentRevisionOption{
		DeploymentId: &d.ID,
		Status:       &status_,
	})
	if err != nil {
		return defaultStatus, err
	}

	deploymentRevisionIds := make([]uint, 0, len(deploymentRevisions))
	for _, deploymentRevision := range deploymentRevisions {
		deploymentRevisionIds = append(deploymentRevisionIds, deploymentRevision.ID)
	}

	deploymentTargets, _, err := DeploymentTargetService.List(ctx, ListDeploymentTargetOption{
		DeploymentRevisionIds: &deploymentRevisionIds,
	})
	if err != nil {
		return defaultStatus, err
	}

	for _, deploymentTarget := range deploymentTargets {
		dynamoNimParts := strings.Split(deploymentTarget.DynamoNimVersionTag, ":")
		if len(dynamoNimParts) != 2 {
			return defaultStatus, errors.Errorf("Invalid format for DynamoNIM version tag %s. Expected 2 parts got %d", deploymentTarget.DynamoNimVersionTag, len(dynamoNimParts))
		}

		imageBuilderPodsSelector, err := labels.Parse(fmt.Sprintf("%s=%s,%s=%s", consts.KubeLabelDynamoNim, dynamoNimParts[0], consts.KubeLabelDynamoNimVersion, dynamoNimParts[1]))
		if err != nil {
			return defaultStatus, err
		}

		var pods_ []*apiv1.Pod
		pods_, err = K8sService.ListPodsBySelector(ctx, podLister, imageBuilderPodsSelector)
		if err != nil {
			return defaultStatus, err
		}
		imageBuilderPods = append(imageBuilderPods, pods_...)
	}

	log.Info().Msgf("Fetched %d image builder jobs", len(imageBuilderPods))
	if len(imageBuilderPods) != 0 {
		for _, imageBuilderPod := range imageBuilderPods {
			for _, container := range imageBuilderPod.Status.ContainerStatuses {
				if container.Name == consts.KubeImageBuilderMainContainer {
					if container.State.Waiting != nil || container.State.Running != nil {
						return schemas.DeploymentStatusImageBuilding, nil
					} else if container.State.Terminated != nil {
						if container.State.Terminated.ExitCode != 0 {
							return schemas.DeploymentStatusImageBuildFailed, nil
						}
					}
				}
			}
		}
	}

	pods, err := K8sService.ListPodsByDeployment(ctx, podLister, d)
	if err != nil {
		return defaultStatus, err
	}

	log.Info().Msgf("Fetched %d pods", len(pods))
	if len(pods) == 0 {
		if d.Status == schemas.DeploymentStatusTerminating || d.Status == schemas.DeploymentStatusTerminated {
			return schemas.DeploymentStatusTerminated, nil
		}
		if d.Status == schemas.DeploymentStatusDeploying {
			return schemas.DeploymentStatusDeploying, nil
		}
		return schemas.DeploymentStatusNonDeployed, nil
	}

	if d.Status == schemas.DeploymentStatusTerminated {
		return d.Status, nil
	}

	hasFailed := false
	hasRunning := false
	hasPending := false

	for _, p := range pods {
		log.Info().Msgf("pod %s has status %s", p.Name, p.Status.Phase)
		podStatus := p.Status
		if podStatus.Phase == apiv1.PodRunning {
			hasRunning = true
		}
		if podStatus.Phase == apiv1.PodFailed {
			hasFailed = true
		}
		if podStatus.Phase == apiv1.PodPending {
			hasPending = true
		}
	}

	var deploymentStatus schemas.DeploymentStatus
	if d.Status == schemas.DeploymentStatusTerminating {
		if !hasRunning {
			deploymentStatus = schemas.DeploymentStatusTerminated
		} else {
			deploymentStatus = schemas.DeploymentStatusTerminating
		}
	} else if hasFailed && hasRunning {
		if hasPending {
			deploymentStatus = schemas.DeploymentStatusDeploying
		} else {
			deploymentStatus = schemas.DeploymentStatusUnhealthy

		}
	} else if hasPending {
		deploymentStatus = schemas.DeploymentStatusDeploying
	} else if hasRunning {
		deploymentStatus = schemas.DeploymentStatusRunning
	}

	log.Info().Msgf("The current status of the deployment is %s", deploymentStatus)
	return deploymentStatus, nil
}
