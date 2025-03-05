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
	"fmt"
	"sync"
	"time"

	"github.com/dynemo-ai/dynemo/deploy/compoundai/api-server/api/models"
	"k8s.io/client-go/informers"
	informerAppsV1 "k8s.io/client-go/informers/apps/v1"
	informerCoreV1 "k8s.io/client-go/informers/core/v1"
	informerNetworkingV1 "k8s.io/client-go/informers/networking/v1"
	listerAppsV1 "k8s.io/client-go/listers/apps/v1"
	listerCoreV1 "k8s.io/client-go/listers/core/v1"
	listerNetworkingV1 "k8s.io/client-go/listers/networking/v1"
	"k8s.io/client-go/tools/cache"
)

type CacheKey string

var (
	informerSyncTimeout = 30 * time.Second

	informerFactoryCache   = make(map[CacheKey]informers.SharedInformerFactory)
	informerFactoryCacheRW = &sync.RWMutex{}
)

type getSharedInformerFactoryOption struct {
	cluster   *models.Cluster
	namespace *string
}

func getSharedInformerFactory(option *getSharedInformerFactoryOption) (informers.SharedInformerFactory, error) {

	var cacheKey CacheKey
	if option.namespace != nil {
		cacheKey = CacheKey(fmt.Sprintf("%s:%s", option.cluster.Name, *option.namespace))
	} else {
		cacheKey = CacheKey(option.cluster.Name)
	}

	informerFactoryCacheRW.Lock()
	defer informerFactoryCacheRW.Unlock()

	factory, ok := informerFactoryCache[cacheKey]
	if !ok {
		clientset, err := K8sService.GetK8sClient(option.cluster.KubeConfig)
		if err != nil {
			return nil, err
		}
		informerOptions := make([]informers.SharedInformerOption, 0)
		if option.namespace != nil {
			informerOptions = append(informerOptions, informers.WithNamespace(*option.namespace))
		}
		factory = informers.NewSharedInformerFactoryWithOptions(clientset, 0, informerOptions...)
	}

	return factory, nil
}

func startAndSyncInformer(ctx context.Context, informer cache.SharedIndexInformer) (err error) {
	go informer.Run(ctx.Done())

	ctx_, cancel := context.WithTimeout(ctx, informerSyncTimeout)
	defer cancel()

	if !cache.WaitForCacheSync(ctx_.Done(), informer.HasSynced) {
		err = errors.New("timed out waiting for caches to sync informer")
		return err
	}

	return nil
}

func GetPodInformer(ctx context.Context, cluster *models.Cluster, namespace string) (informerCoreV1.PodInformer, listerCoreV1.PodNamespaceLister, error) {
	factory, err := getSharedInformerFactory(&getSharedInformerFactoryOption{
		cluster:   cluster,
		namespace: &namespace,
	})
	if err != nil {
		return nil, nil, err
	}
	podInformer := factory.Core().V1().Pods()
	err = startAndSyncInformer(ctx, podInformer.Informer())
	if err != nil {
		return nil, nil, err
	}
	return podInformer, podInformer.Lister().Pods(namespace), nil
}

func GetDeploymentInformer(ctx context.Context, kubeCluster *models.Cluster, namespace string) (informerAppsV1.DeploymentInformer, listerAppsV1.DeploymentNamespaceLister, error) {
	factory, err := getSharedInformerFactory(&getSharedInformerFactoryOption{
		cluster:   kubeCluster,
		namespace: &namespace,
	})
	if err != nil {
		return nil, nil, err
	}
	deploymentInformer := factory.Apps().V1().Deployments()
	err = startAndSyncInformer(ctx, deploymentInformer.Informer())
	if err != nil {
		return nil, nil, err
	}
	return deploymentInformer, deploymentInformer.Lister().Deployments(namespace), nil
}

func GetIngressInformer(ctx context.Context, kubeCluster *models.Cluster, namespace string) (informerNetworkingV1.IngressInformer, listerNetworkingV1.IngressNamespaceLister, error) {
	factory, err := getSharedInformerFactory(&getSharedInformerFactoryOption{
		cluster:   kubeCluster,
		namespace: &namespace,
	})
	if err != nil {
		return nil, nil, err
	}
	ingressInformer := factory.Networking().V1().Ingresses()
	err = startAndSyncInformer(ctx, ingressInformer.Informer())
	if err != nil {
		return nil, nil, err
	}
	return ingressInformer, ingressInformer.Lister().Ingresses(namespace), nil
}

func GetEventInformer(ctx context.Context, cluster *models.Cluster, namespace string) (informerCoreV1.EventInformer, listerCoreV1.EventNamespaceLister, error) {
	factory, err := getSharedInformerFactory(&getSharedInformerFactoryOption{
		cluster:   cluster,
		namespace: &namespace,
	})
	if err != nil {
		return nil, nil, err
	}
	eventInformer := factory.Core().V1().Events()
	err = startAndSyncInformer(ctx, eventInformer.Informer())
	if err != nil {
		return nil, nil, err
	}
	return eventInformer, eventInformer.Lister().Events(namespace), nil
}
