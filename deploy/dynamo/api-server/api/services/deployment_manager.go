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
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"

	"github.com/dynemo-ai/dynemo/deploy/dynamo/api-server/api/common/consts"
	"github.com/dynemo-ai/dynemo/deploy/dynamo/api-server/api/common/utils"
	"github.com/dynemo-ai/dynemo/deploy/dynamo/api-server/api/crds"
	"github.com/dynemo-ai/dynemo/deploy/dynamo/api-server/api/models"
	"github.com/dynemo-ai/dynemo/deploy/dynamo/api-server/api/schemas"
	"github.com/rs/zerolog/log"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
)

type deploymentManagementService struct{}

var DeploymentManagementService = deploymentManagementService{}

type DMSConfiguration struct {
	Version string      `json:"version"`
	Data    interface{} `json:"data"`
}

type DMSCreateRequest struct {
	Name          string                  `json:"name"`
	Namespace     string                  `json:"namespace"`
	ResourceType  crds.CustomResourceType `json:"type"`
	Configuration interface{}             `json:"configuration"`
	Labels        map[string]string       `json:"labels"`
}

type DMSResponseStatus struct {
	Status  string `json:"status"`
	Message string `json:"message"`
}

type DMSCreateResponse struct {
	Id            string            `json:"id"`
	Status        DMSResponseStatus `json:"status"`
	Configuration interface{}       `json:"configuration"`
}

func (s *deploymentManagementService) Create(ctx context.Context, deploymentTarget *models.DeploymentTarget, deployOption *models.DeployOption, ownership *schemas.OwnershipSchema) (*models.DeploymentTarget, error) {
	dmsHost, dmsPort, err := getDMSPortAndHost()
	if err != nil {
		log.Error().Msg(err.Error())
		return nil, err
	}

	url := fmt.Sprintf("http://%s:%s/v1/deployments", dmsHost, dmsPort)
	deployment, err := DeploymentService.Get(ctx, deploymentTarget.DeploymentId)

	if err != nil {
		log.Info().Msg("Could not find associated deployment")
		return nil, err
	}

	defer func() {
		if err != nil {
			s.Delete(ctx, deploymentTarget)
		}
	}()

	compoundNimDeployment, compoundNimRequest := s.transformToDMSRequestsV1alpha1(deployment, deploymentTarget, ownership)

	body, err := sendRequest(compoundNimDeployment, url, http.MethodPost)
	if err != nil {
		return nil, err
	}
	var result DMSCreateResponse
	err = json.Unmarshal(body, &result)
	if err != nil {
		fmt.Println("Error unmarshaling:", err)
		return nil, err
	}
	deploymentTarget.KubeDeploymentId = result.Id

	body, err = sendRequest(compoundNimRequest, url, http.MethodPost)
	if err != nil {
		return nil, err
	}
	err = json.Unmarshal(body, &result)
	if err != nil {
		fmt.Println("Error unmarshaling:", err)
		return nil, err
	}

	deploymentTarget.KubeRequestId = result.Id
	return deploymentTarget, nil
}

func (s *deploymentManagementService) Delete(ctx context.Context, deploymentTarget *models.DeploymentTarget) error {
	dmsHost, dmsPort, err := getDMSPortAndHost()
	if err != nil {
		log.Error().Msg(err.Error())
		return err
	}

	if deploymentTarget.KubeDeploymentId != "" {
		urlDeployment := fmt.Sprintf("http://%s:%s/v1/deployments/%s", dmsHost, dmsPort, deploymentTarget.KubeDeploymentId)
		_, err := sendRequest(nil, urlDeployment, http.MethodDelete)
		if err != nil {
			return err
		}
	}

	if deploymentTarget.KubeRequestId != "" {
		urlRequest := fmt.Sprintf("http://%s:%s/v1/deployments/%s", os.Getenv("DMS_HOST"), os.Getenv("DMS_PORT"), deploymentTarget.KubeRequestId)
		_, err := sendRequest(nil, urlRequest, http.MethodDelete)
		if err != nil {
			return err
		}
	}

	return nil
}

func (s *deploymentManagementService) transformToDMSRequestsV1alpha1(deployment *models.Deployment, deploymentTarget *models.DeploymentTarget, ownership *schemas.OwnershipSchema) (compoundNimDeployment DMSCreateRequest, compoundNimRequest DMSCreateRequest) {
	translatedTag := s.translateCompoundNimVersionTagToRFC1123(deploymentTarget.CompoundNimVersionTag)

	livenessProbe, readinessProbe := createProbeSpecs(deploymentTarget.Config.DeploymentOverrides)

	compoundNimDeployment = DMSCreateRequest{
		Name:         deployment.Name,
		Namespace:    deployment.KubeNamespace,
		ResourceType: crds.CompoundNimDeployment,
		Configuration: crds.CompoundNimDeploymentConfigurationV1Alpha1{
			Data: crds.CompoundNimDeploymentData{
				CompoundNimVersion: translatedTag,
				Resources:          *deploymentTarget.Config.Resources,
				ExternalServices:   deploymentTarget.Config.ExternalServices,
				LivenessProbe:      livenessProbe,
				ReadinessProbe:     readinessProbe,
			},
			Version: crds.ApiVersion,
		},
		Labels: map[string]string{
			consts.NgcOrganizationHeaderName: ownership.OrganizationId,
			consts.NgcUserHeaderName:         ownership.UserId,
		},
	}

	compoundNimRequest = DMSCreateRequest{
		Name:         translatedTag,
		Namespace:    deployment.KubeNamespace,
		ResourceType: crds.CompoundNimRequest,
		Configuration: crds.CompoundNimRequestConfigurationV1Alpha1{
			Data: crds.CompoundNimRequestData{
				CompoundNimVersionTag: deploymentTarget.CompoundNimVersionTag,
			},
			Version: crds.ApiVersion,
		},
		Labels: map[string]string{
			consts.NgcOrganizationHeaderName: ownership.OrganizationId,
			consts.NgcUserHeaderName:         ownership.UserId,
		},
	}
	return
}

func createProbeSpecs(deploymentOverrides *schemas.DeploymentOverrides) (livenessProbe *corev1.Probe, readinessProbe *corev1.Probe) {
	if deploymentOverrides != nil && deploymentOverrides.ColdStartTimeout != nil {
		livenessProbe = &corev1.Probe{
			InitialDelaySeconds: *deploymentOverrides.ColdStartTimeout,
			TimeoutSeconds:      20,
			FailureThreshold:    6,
			ProbeHandler: corev1.ProbeHandler{
				HTTPGet: &corev1.HTTPGetAction{
					Path: "/livez",
					Port: intstr.FromString(consts.CompoundNimContainerPortName),
				},
			},
		}

		readinessProbe = &corev1.Probe{
			InitialDelaySeconds: *deploymentOverrides.ColdStartTimeout,
			TimeoutSeconds:      5,
			FailureThreshold:    12,
			ProbeHandler: corev1.ProbeHandler{
				HTTPGet: &corev1.HTTPGetAction{
					Path: "/readyz",
					Port: intstr.FromString(consts.CompoundNimContainerPortName),
				},
			},
		}
	}

	return
}

func getDMSPortAndHost() (string, string, error) {
	dmsHost, err := utils.MustGetEnv("DMS_HOST")
	if err != nil {
		return "", "", err
	}

	dmsPort, err := utils.MustGetEnv("DMS_PORT")
	if err != nil {
		return "", "", err
	}

	return dmsHost, dmsPort, nil
}

/**
 * Translates a Compound NIM Version tag to a valid RFC 1123 DNS label.
 *
 * This function makes the following modifications to the input string:
 * 1. Replaces all ":" characters with "--" because colons are not permitted in DNS labels.
 * 2. If the resulting string exceeds the 63-character limit imposed by RFC 1123, it truncates
 *    the string to 63 characters.
 *
 * @param {string} tag - The original Dynamo Nim tag that needs to be converted.
 * @returns {string} - A string that complies with the RFC 1123 DNS label format.
 *
 * Example:
 *   Input: "nim:latest"
 *   Output: "nim--latest"
 */
func (s *deploymentManagementService) translateCompoundNimVersionTagToRFC1123(tag string) string {
	translated := strings.ReplaceAll(tag, ":", "--")

	// If the length exceeds 63 characters, truncate it
	if len(translated) > 63 {
		translated = translated[:63]
	}

	return translated
}

func sendRequest(payload interface{}, url string, method string) ([]byte, error) {
	jsonData, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %v", err)
	}

	req, err := http.NewRequest(method, url, bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %v", err)
	}

	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to send request: %v", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("received non-OK response: %v, %s", resp.Status, body)
	}

	return body, nil
}
