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
	"encoding/json"
	"fmt"

	"github.com/ai-dynamo/dynamo/deploy/dynamo/api-server/api/common/consts"
	"github.com/ai-dynamo/dynamo/deploy/dynamo/api-server/api/models"
	"github.com/ghodss/yaml"

	dynamov1alpha1 "github.com/ai-dynamo/dynamo/deploy/dynamo/operator/api/v1alpha1"
	apiv1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/kubernetes"
	v1 "k8s.io/client-go/listers/core/v1"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
	clientCmdApi "k8s.io/client-go/tools/clientcmd/api"
	clientCmdLatest "k8s.io/client-go/tools/clientcmd/api/latest"
	clientCmdApiV1 "k8s.io/client-go/tools/clientcmd/api/v1"
)

type k8sService struct{}

var K8sService IK8sService = &k8sService{}

func (s *k8sService) GetK8sRestConfig(kubeConfig string) (*rest.Config, error) {
	var restConfig *rest.Config
	var err error

	if kubeConfig == "" {
		restConfig, err = rest.InClusterConfig()
		if err != nil {
			kubeConfig :=
				clientcmd.NewDefaultClientConfigLoadingRules().GetDefaultFilename()
			restConfig, err = clientcmd.BuildConfigFromFlags("", kubeConfig)
			if err != nil {
				return nil, err
			}
		}
	} else {
		configV1 := clientCmdApiV1.Config{}
		var jsonBytes []byte
		jsonBytes, err := yaml.YAMLToJSON([]byte(kubeConfig))
		if err != nil {
			return nil, err
		}
		err = json.Unmarshal(jsonBytes, &configV1)
		if err != nil {
			return nil, err
		}

		var configObject runtime.Object
		configObject, err = clientCmdLatest.Scheme.ConvertToVersion(&configV1, clientCmdApi.SchemeGroupVersion)
		if err != nil {
			return nil, err
		}
		configInternal := configObject.(*clientCmdApi.Config)

		restConfig, err = clientcmd.NewDefaultClientConfig(*configInternal, &clientcmd.ConfigOverrides{
			ClusterDefaults: clientCmdApi.Cluster{Server: ""},
		}).ClientConfig()

		if err != nil {
			return nil, err
		}
	}

	return restConfig, nil
}

func (s *k8sService) GetK8sClient(kubeConfig string) (kubernetes.Interface, error) {
	restConfig, err := s.GetK8sRestConfig(kubeConfig)
	if err != nil {
		return nil, err
	}

	clientSet, err := kubernetes.NewForConfig(restConfig)
	if err != nil {
		return nil, err
	}

	return clientSet, nil
}

func (s *k8sService) GetDynamicClient(kubeConfig string) (dynamic.Interface, error) {
	restConfig, err := s.GetK8sRestConfig(kubeConfig)
	if err != nil {
		return nil, err
	}

	clientSet, err := dynamic.NewForConfig(restConfig)
	if err != nil {
		return nil, err
	}

	return clientSet, nil
}

func (s *k8sService) ListPodsByDeployment(ctx context.Context, podLister v1.PodNamespaceLister, deployment *models.Deployment) ([]*apiv1.Pod, error) {
	selector, err := labels.Parse(fmt.Sprintf("%s = %s", consts.KubeLabelDynamoNimVersionDeployment, deployment.Name))
	if err != nil {
		return nil, err
	}

	return s.ListPodsBySelector(ctx, podLister, selector)
}

func (s *k8sService) ListPodsBySelector(ctx context.Context, podLister v1.PodNamespaceLister, selector labels.Selector) ([]*apiv1.Pod, error) {
	pods, err := podLister.List(selector)
	if err != nil {
		return nil, err
	}

	return pods, nil
}

func (s *k8sService) CreateDynamoDeployment(ctx context.Context, dynamoDeployment *dynamov1alpha1.DynamoDeployment) error {
	k8sClient, err := s.GetDynamicClient("")
	if err != nil {
		return err
	}
	unstructuredData, err := runtime.DefaultUnstructuredConverter.ToUnstructured(dynamoDeployment)
	if err != nil {
		return err
	}
	unstructured := &unstructured.Unstructured{Object: unstructuredData}
	gvr := schema.GroupVersionResource{
		Group:    dynamov1alpha1.GroupVersion.Group,
		Version:  dynamov1alpha1.GroupVersion.Version,
		Resource: "dynamodeployments",
	}
	_, err = k8sClient.Resource(gvr).Namespace(dynamoDeployment.Namespace).Create(ctx, unstructured, metav1.CreateOptions{})
	if err != nil {
		return err
	}
	return nil
}

type IK8sService interface {
	GetK8sClient(string) (kubernetes.Interface, error)
	CreateDynamoDeployment(context.Context, *dynamov1alpha1.DynamoDeployment) error
	ListPodsByDeployment(context.Context, v1.PodNamespaceLister, *models.Deployment) ([]*apiv1.Pod, error)
	ListPodsBySelector(context.Context, v1.PodNamespaceLister, labels.Selector) ([]*apiv1.Pod, error)
}
