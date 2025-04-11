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

package nim

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"net/http"
	"strings"

	"emperror.dev/errors"
	compounaiCommon "github.com/ai-dynamo/dynamo/deploy/dynamo/operator/api/dynamo/common"
	"github.com/ai-dynamo/dynamo/deploy/dynamo/operator/api/dynamo/schemas"
	yataiclient "github.com/ai-dynamo/dynamo/deploy/dynamo/operator/api/dynamo/yatai-client"
	"github.com/ai-dynamo/dynamo/deploy/dynamo/operator/api/v1alpha1"
	commonconfig "github.com/ai-dynamo/dynamo/deploy/dynamo/operator/internal/config"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/dynamo/operator/internal/consts"
	"github.com/huandu/xstrings"
	corev1 "k8s.io/api/core/v1"
	k8serrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/runtime"
	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/ai-dynamo/dynamo/deploy/dynamo/operator/internal/archive"
	"gopkg.in/yaml.v2"
)

// ServiceConfig represents the YAML configuration structure for a service
type DynamoConfig struct {
	Enabled   bool   `yaml:"enabled"`
	Namespace string `yaml:"namespace"`
	Name      string `yaml:"name"`
}

type Resources struct {
	CPU    string            `yaml:"cpu,omitempty"`
	Memory string            `yaml:"memory,omitempty"`
	GPU    string            `yaml:"gpu,omitempty"`
	Custom map[string]string `yaml:"custom,omitempty"`
}

type Traffic struct {
	Timeout int `yaml:"timeout"`
}

type Autoscaling struct {
	MinReplicas int `yaml:"min_replicas"`
	MaxReplicas int `yaml:"max_replicas"`
}

type Config struct {
	Dynamo      *DynamoConfig `yaml:"dynamo,omitempty"`
	Resources   *Resources    `yaml:"resources,omitempty"`
	Traffic     *Traffic      `yaml:"traffic,omitempty"`
	Autoscaling *Autoscaling  `yaml:"autoscaling,omitempty"`
}

type ServiceConfig struct {
	Name         string              `yaml:"name"`
	Dependencies []map[string]string `yaml:"dependencies,omitempty"`
	Config       Config              `yaml:"config"`
}

func RetrieveDynamoNimDownloadURL(ctx context.Context, dynamoDeployment *v1alpha1.DynamoDeployment, recorder EventRecorder) (*string, *string, error) {
	dynamoNimDownloadURL := ""
	dynamoNimApiToken := ""
	var dynamoNim *schemas.DynamoNIM
	dynamoNimRepositoryName, _, dynamoNimVersion := xstrings.Partition(dynamoDeployment.Spec.DynamoNim, ":")

	var err error
	var yataiClient_ **yataiclient.YataiClient
	var yataiConf_ **commonconfig.YataiConfig

	yataiClient_, yataiConf_, err = GetYataiClient(ctx)
	if err != nil {
		err = errors.Wrap(err, "get yatai client")
		return nil, nil, err
	}

	if yataiClient_ == nil || yataiConf_ == nil {
		err = errors.New("can't get yatai client, please check yatai configuration")
		return nil, nil, err
	}

	yataiClient := *yataiClient_
	yataiConf := *yataiConf_

	recorder.Eventf(dynamoDeployment, corev1.EventTypeNormal, "GenerateImageBuilderPod", "Getting dynamoNim %s from yatai service", dynamoDeployment.Spec.DynamoNim)
	dynamoNim, err = yataiClient.GetBento(ctx, dynamoNimRepositoryName, dynamoNimVersion)
	if err != nil {
		err = errors.Wrap(err, "get dynamoNim")
		return nil, nil, err
	}
	recorder.Eventf(dynamoDeployment, corev1.EventTypeNormal, "GenerateImageBuilderPod", "Got dynamoNim %s from yatai service", dynamoDeployment.Spec.DynamoNim)

	if dynamoNim.TransmissionStrategy != nil && *dynamoNim.TransmissionStrategy == schemas.TransmissionStrategyPresignedURL {
		var dynamoNim_ *schemas.DynamoNIM
		recorder.Eventf(dynamoDeployment, corev1.EventTypeNormal, "GenerateImageBuilderPod", "Getting presigned url for dynamoNim %s from yatai service", dynamoDeployment.Spec.DynamoNim)
		dynamoNim_, err = yataiClient.PresignBentoDownloadURL(ctx, dynamoNimRepositoryName, dynamoNimVersion)
		if err != nil {
			err = errors.Wrap(err, "presign dynamoNim download url")
			return nil, nil, err
		}
		recorder.Eventf(dynamoDeployment, corev1.EventTypeNormal, "GenerateImageBuilderPod", "Got presigned url for dynamoNim %s from yatai service", dynamoDeployment.Spec.DynamoNim)
		dynamoNimDownloadURL = dynamoNim_.PresignedDownloadUrl
	} else {
		dynamoNimDownloadURL = fmt.Sprintf("%s/api/v1/dynamo_nims/%s/versions/%s/download", yataiConf.Endpoint, dynamoNimRepositoryName, dynamoNimVersion)
		dynamoNimApiToken = fmt.Sprintf("%s:%s:$%s", commonconsts.YataiImageBuilderComponentName, yataiConf.ClusterName, commonconsts.EnvYataiApiToken)
	}

	return &dynamoNimDownloadURL, &dynamoNimApiToken, nil
}

// ServicesConfig represents the top-level YAML structure of a dynamoNim yaml file stored in a dynamoNim tar file
type DynamoNIMConfig struct {
	DynamoTag    string          `yaml:"service"`
	Services     []ServiceConfig `yaml:"services"`
	EntryService string          `yaml:"entry_service"`
}

type EventRecorder interface {
	Eventf(obj runtime.Object, eventtype string, reason string, message string, args ...interface{})
}

func RetrieveDynamoNIMConfigurationFile(ctx context.Context, url string, yataiApiToken string) (*bytes.Buffer, error) {
	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set(commonconsts.YataiApiTokenHeaderName, yataiApiToken)

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer func() {
		if err := resp.Body.Close(); err != nil {
			logger := log.FromContext(ctx)
			logger.Error(err, "error closing response body")
		}
	}()

	// Read the tar file into memory
	tarData, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	// Extract the YAML file
	yamlFileName := "bento.yaml"
	yamlContent, err := archive.ExtractFileFromTar(tarData, yamlFileName)
	if err != nil {
		return nil, err
	}

	return yamlContent, nil
}

func GetYataiClient(ctx context.Context) (yataiClient **yataiclient.YataiClient, yataiConf **commonconfig.YataiConfig, err error) {
	yataiConf_, err := commonconfig.GetYataiConfig(ctx)
	isNotFound := k8serrors.IsNotFound(err)
	if err != nil && !isNotFound {
		err = errors.Wrap(err, "get yatai config")
		return
	}

	if isNotFound {
		return
	}

	if yataiConf_.Endpoint == "" {
		return
	}

	if yataiConf_.ClusterName == "" {
		yataiConf_.ClusterName = "default"
	}

	yataiClient_ := yataiclient.NewYataiClient(yataiConf_.Endpoint, fmt.Sprintf("%s:%s:%s", commonconsts.YataiImageBuilderComponentName, yataiConf_.ClusterName, yataiConf_.ApiToken))

	yataiClient = &yataiClient_
	yataiConf = &yataiConf_
	return
}

func ParseDynamoNIMConfig(ctx context.Context, yamlContent *bytes.Buffer) (*DynamoNIMConfig, error) {
	var config DynamoNIMConfig
	logger := log.FromContext(ctx)
	logger.Info("trying to parse dynamoNim config", "yamlContent", yamlContent.String())
	err := yaml.Unmarshal(yamlContent.Bytes(), &config)
	return &config, err
}

func GetDynamoNIMConfig(ctx context.Context, dynamoDeployment *v1alpha1.DynamoDeployment, recorder EventRecorder) (*DynamoNIMConfig, error) {
	dynamoNimDownloadURL, dynamoNimApiToken, err := RetrieveDynamoNimDownloadURL(ctx, dynamoDeployment, recorder)
	if err != nil {
		return nil, err
	}
	yamlContent, err := RetrieveDynamoNIMConfigurationFile(ctx, *dynamoNimDownloadURL, *dynamoNimApiToken)
	if err != nil {
		return nil, err
	}
	return ParseDynamoNIMConfig(ctx, yamlContent)
}

// generate DynamoNIMDeployment from config
func GenerateDynamoNIMDeployments(parentDynamoDeployment *v1alpha1.DynamoDeployment, config *DynamoNIMConfig) (map[string]*v1alpha1.DynamoNimDeployment, error) {
	dynamoServices := make(map[string]string)
	deployments := make(map[string]*v1alpha1.DynamoNimDeployment)
	for _, service := range config.Services {
		deployment := &v1alpha1.DynamoNimDeployment{}
		deployment.Name = fmt.Sprintf("%s-%s", parentDynamoDeployment.Name, strings.ToLower(service.Name))
		deployment.Namespace = parentDynamoDeployment.Namespace
		deployment.Spec.DynamoTag = config.DynamoTag
		deployment.Spec.DynamoNim = strings.ReplaceAll(parentDynamoDeployment.Spec.DynamoNim, ":", "--")
		deployment.Spec.ServiceName = service.Name
		if service.Config.Dynamo != nil && service.Config.Dynamo.Enabled {
			dynamoServices[service.Name] = fmt.Sprintf("%s/%s", service.Config.Dynamo.Name, service.Config.Dynamo.Namespace)
		} else {
			// dynamo is not enabled
			if config.EntryService == service.Name {
				// enable virtual service for the entry service
				deployment.Spec.Ingress.Enabled = true
				deployment.Spec.Ingress.UseVirtualService = &deployment.Spec.Ingress.Enabled
			}
		}
		if service.Config.Resources != nil {
			deployment.Spec.Resources = &compounaiCommon.Resources{
				Requests: &compounaiCommon.ResourceItem{
					CPU:    service.Config.Resources.CPU,
					Memory: service.Config.Resources.Memory,
					GPU:    service.Config.Resources.GPU,
					Custom: service.Config.Resources.Custom,
				},
				Limits: &compounaiCommon.ResourceItem{
					CPU:    service.Config.Resources.CPU,
					Memory: service.Config.Resources.Memory,
					GPU:    service.Config.Resources.GPU,
					Custom: service.Config.Resources.Custom,
				},
			}
		}
		deployment.Spec.Autoscaling = &v1alpha1.Autoscaling{
			Enabled: false,
		}
		if service.Config.Autoscaling != nil {
			deployment.Spec.Autoscaling.Enabled = true
			deployment.Spec.Autoscaling.MinReplicas = service.Config.Autoscaling.MinReplicas
			deployment.Spec.Autoscaling.MaxReplicas = service.Config.Autoscaling.MaxReplicas
		}
		deployments[service.Name] = deployment
	}
	for _, service := range config.Services {
		deployment := deployments[service.Name]
		// generate external services
		for _, dependency := range service.Dependencies {
			dependentServiceName := dependency["service"]
			if deployment.Spec.ExternalServices == nil {
				deployment.Spec.ExternalServices = make(map[string]v1alpha1.ExternalService)
			}
			dependencyDeployment := deployments[dependentServiceName]
			if dependencyDeployment == nil {
				return nil, fmt.Errorf("dependency %s not found", dependentServiceName)
			}
			if dynamoService, ok := dynamoServices[dependentServiceName]; ok {
				deployment.Spec.ExternalServices[dependentServiceName] = v1alpha1.ExternalService{
					DeploymentSelectorKey:   "dynamo",
					DeploymentSelectorValue: dynamoService,
				}
			} else {
				deployment.Spec.ExternalServices[dependentServiceName] = v1alpha1.ExternalService{
					DeploymentSelectorKey:   "name",
					DeploymentSelectorValue: dependentServiceName,
				}
			}
		}
	}
	return deployments, nil
}
