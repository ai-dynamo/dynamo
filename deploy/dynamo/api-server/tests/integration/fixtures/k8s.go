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

package fixtures

import (
	"context"
	"fmt"

	"github.com/dynemo-ai/dynemo/deploy/dynamo/api-server/api/common/consts"
	"github.com/dynemo-ai/dynemo/deploy/dynamo/api-server/api/models"

	"github.com/rs/zerolog/log"
	apiv1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/fake"
	v1 "k8s.io/client-go/listers/core/v1"
)

type MockedK8sService struct{}

func (s *MockedK8sService) GetK8sClient(kubeConfig string) (kubernetes.Interface, error) {
	log.Info().Msgf("Using fake client.")
	return fake.NewClientset(), nil
}

func (s *MockedK8sService) ListPodsByDeployment(ctx context.Context, podLister v1.PodNamespaceLister, deployment *models.Deployment) ([]*apiv1.Pod, error) {
	log.Info().Msgf("Faking list by deployment")
	selector, err := labels.Parse(fmt.Sprintf("%s = %s", consts.KubeLabelCompoundNimVersionDeployment, deployment.Name))
	if err != nil {
		return nil, err
	}

	return s.ListPodsBySelector(ctx, podLister, selector)
}

func (s *MockedK8sService) ListPodsBySelector(ctx context.Context, podLister v1.PodNamespaceLister, selector labels.Selector) ([]*apiv1.Pod, error) {
	log.Info().Msgf("Faking list by selector")
	pods, err := podLister.List(selector)
	if err != nil {
		return nil, err
	}

	return pods, nil
}
