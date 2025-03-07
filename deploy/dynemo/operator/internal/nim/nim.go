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
	compounaiCommon "github.com/dynemo-ai/dynemo/deploy/compoundai/operator/api/compoundai/common"
	"github.com/dynemo-ai/dynemo/deploy/compoundai/operator/api/compoundai/modelschemas"
	"github.com/dynemo-ai/dynemo/deploy/compoundai/operator/api/compoundai/schemasv1"
	yataiclient "github.com/dynemo-ai/dynemo/deploy/compoundai/operator/api/compoundai/yatai-client"
	"github.com/dynemo-ai/dynemo/deploy/compoundai/operator/api/v1alpha1"
	commonconfig "github.com/dynemo-ai/dynemo/deploy/compoundai/operator/pkg/compoundai/config"
	commonconsts "github.com/dynemo-ai/dynemo/deploy/compoundai/operator/pkg/compoundai/consts"
	"github.com/huandu/xstrings"
	corev1 "k8s.io/api/core/v1"
	k8serrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/runtime"
	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/dynemo-ai/dynemo/deploy/compoundai/operator/internal/archive"
	"gopkg.in/yaml.v2"
)

// ServiceConfig represents the YAML configuration structure for a service
type NovaConfig struct {
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
	Nova        *NovaConfig  `yaml:"nova,omitempty"`
	Resources   *Resources   `yaml:"resources,omitempty"`
	Traffic     *Traffic     `yaml:"traffic,omitempty"`
	Autoscaling *Autoscaling `yaml:"autoscaling,omitempty"`
}

type ServiceConfig struct {
	Name         string              `yaml:"name"`
	Dependencies []map[string]string `yaml:"dependencies,omitempty"`
	Config       Config              `yaml:"config"`
}

func RetrieveCompoundAINimDownloadURL(ctx context.Context, compoundAIDeployment *v1alpha1.CompoundAIDeployment, secretGetter SecretGetter, recorder EventRecorder) (*string, *string, error) {
	compoundAINimDownloadURL := ""
	compoundAINimApiToken := ""
	var compoundAINim *schemasv1.BentoFullSchema
	compoundAINimRepositoryName, _, compoundAINimVersion := xstrings.Partition(compoundAIDeployment.Spec.CompoundAINim, ":")

	var err error
	var yataiClient_ **yataiclient.YataiClient
	var yataiConf_ **commonconfig.YataiConfig

	yataiClient_, yataiConf_, err = GetYataiClient(ctx, secretGetter)
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

	recorder.Eventf(compoundAIDeployment, corev1.EventTypeNormal, "GenerateImageBuilderPod", "Getting compoundAINim %s from yatai service", compoundAIDeployment.Spec.CompoundAINim)
	compoundAINim, err = yataiClient.GetBento(ctx, compoundAINimRepositoryName, compoundAINimVersion)
	if err != nil {
		err = errors.Wrap(err, "get compoundAINim")
		return nil, nil, err
	}
	recorder.Eventf(compoundAIDeployment, corev1.EventTypeNormal, "GenerateImageBuilderPod", "Got compoundAINim %s from yatai service", compoundAIDeployment.Spec.CompoundAINim)

	if compoundAINim.TransmissionStrategy != nil && *compoundAINim.TransmissionStrategy == modelschemas.TransmissionStrategyPresignedURL {
		var compoundAINim_ *schemasv1.BentoSchema
		recorder.Eventf(compoundAIDeployment, corev1.EventTypeNormal, "GenerateImageBuilderPod", "Getting presigned url for compoundAINim %s from yatai service", compoundAIDeployment.Spec.CompoundAINim)
		compoundAINim_, err = yataiClient.PresignBentoDownloadURL(ctx, compoundAINimRepositoryName, compoundAINimVersion)
		if err != nil {
			err = errors.Wrap(err, "presign compoundAINim download url")
			return nil, nil, err
		}
		recorder.Eventf(compoundAIDeployment, corev1.EventTypeNormal, "GenerateImageBuilderPod", "Got presigned url for compoundAINim %s from yatai service", compoundAIDeployment.Spec.CompoundAINim)
		compoundAINimDownloadURL = compoundAINim_.PresignedDownloadUrl
	} else {
		compoundAINimDownloadURL = fmt.Sprintf("%s/api/v1/bento_repositories/%s/bentos/%s/download", yataiConf.Endpoint, compoundAINimRepositoryName, compoundAINimVersion)
		compoundAINimApiToken = fmt.Sprintf("%s:%s:$%s", commonconsts.YataiImageBuilderComponentName, yataiConf.ClusterName, commonconsts.EnvYataiApiToken)
	}

	return &compoundAINimDownloadURL, &compoundAINimApiToken, nil
}

// ServicesConfig represents the top-level YAML structure of a compoundAINim yaml file stored in a compoundAINim tar file
type CompoundAINIMConfig struct {
	Services []ServiceConfig `yaml:"services"`
}

type EventRecorder interface {
	Eventf(obj runtime.Object, eventtype string, reason string, message string, args ...interface{})
}

func RetrieveCompoundAINIMConfigurationFile(ctx context.Context, url string, yataiApiToken string) (*bytes.Buffer, error) {
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

func GetYataiClientWithAuth(ctx context.Context, compoundAINimRequest *v1alpha1.CompoundAINimRequest, secretGetter SecretGetter) (**yataiclient.YataiClient, **commonconfig.YataiConfig, error) {
	orgId, ok := compoundAINimRequest.Labels[commonconsts.NgcOrganizationHeaderName]
	if !ok {
		orgId = commonconsts.DefaultOrgId
	}

	userId, ok := compoundAINimRequest.Labels[commonconsts.NgcUserHeaderName]
	if !ok {
		userId = commonconsts.DefaultUserId
	}

	auth := yataiclient.CompoundAIAuthHeaders{
		OrgId:  orgId,
		UserId: userId,
	}

	client, yataiConf, err := GetYataiClient(ctx, secretGetter)
	if err != nil {
		return nil, nil, err
	}

	(*client).SetAuth(auth)
	return client, yataiConf, err
}

type SecretGetter func(ctx context.Context, namespace, name string) (*corev1.Secret, error)

func GetYataiClient(ctx context.Context, secretGetter SecretGetter) (yataiClient **yataiclient.YataiClient, yataiConf **commonconfig.YataiConfig, err error) {
	yataiConf_, err := commonconfig.GetYataiConfig(ctx, secretGetter, commonconsts.YataiImageBuilderComponentName, false)
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

func ParseCompoundAINIMConfig(ctx context.Context, yamlContent *bytes.Buffer) (*CompoundAINIMConfig, error) {
	var config CompoundAINIMConfig
	logger := log.FromContext(ctx)
	logger.Info("trying to parse compoundAINim config", "yamlContent", yamlContent.String())
	err := yaml.Unmarshal(yamlContent.Bytes(), &config)
	return &config, err
}

func GetCompoundAINIMConfig(ctx context.Context, compoundAIDeployment *v1alpha1.CompoundAIDeployment, secretGetter SecretGetter, recorder EventRecorder) (*CompoundAINIMConfig, error) {
	compoundAINimDownloadURL, compoundAINimApiToken, err := RetrieveCompoundAINimDownloadURL(ctx, compoundAIDeployment, secretGetter, recorder)
	if err != nil {
		return nil, err
	}
	yamlContent, err := RetrieveCompoundAINIMConfigurationFile(ctx, *compoundAINimDownloadURL, *compoundAINimApiToken)
	if err != nil {
		return nil, err
	}
	return ParseCompoundAINIMConfig(ctx, yamlContent)
}

// generate CompoundAINIMDeployment from config
func GenerateCompoundAINIMDeployments(parentCompoundAIDeployment *v1alpha1.CompoundAIDeployment, config *CompoundAINIMConfig) (map[string]*v1alpha1.CompoundAINimDeployment, error) {
	novaServices := make(map[string]string)
	deployments := make(map[string]*v1alpha1.CompoundAINimDeployment)
	for _, service := range config.Services {
		deployment := &v1alpha1.CompoundAINimDeployment{}
		deployment.Name = fmt.Sprintf("%s-%s", parentCompoundAIDeployment.Name, strings.ToLower(service.Name))
		deployment.Namespace = parentCompoundAIDeployment.Namespace
		deployment.Spec.CompoundAINim = strings.Split(parentCompoundAIDeployment.Spec.CompoundAINim, ":")[0]
		deployment.Spec.ServiceName = service.Name
		if service.Config.Nova != nil && service.Config.Nova.Enabled {
			novaServices[service.Name] = fmt.Sprintf("%s/%s", service.Config.Nova.Name, service.Config.Nova.Namespace)
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
		if service.Config.Autoscaling != nil {
			deployment.Spec.Autoscaling = &v1alpha1.Autoscaling{
				MinReplicas: service.Config.Autoscaling.MinReplicas,
				MaxReplicas: service.Config.Autoscaling.MaxReplicas,
			}
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
			if novaService, ok := novaServices[dependentServiceName]; ok {
				deployment.Spec.ExternalServices[dependentServiceName] = v1alpha1.ExternalService{
					DeploymentSelectorKey:   "nova",
					DeploymentSelectorValue: novaService,
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
