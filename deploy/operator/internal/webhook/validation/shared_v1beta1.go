/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

package validation

import (
	"errors"
	"fmt"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
)

// V1Beta1SharedSpecValidator validates v1beta1-only shared spec invariants.
type V1Beta1SharedSpecValidator struct {
	spec      *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec
	fieldPath string
}

// NewV1Beta1SharedSpecValidator creates a v1beta1 shared spec validator.
func NewV1Beta1SharedSpecValidator(spec *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec, fieldPath string) *V1Beta1SharedSpecValidator {
	return &V1Beta1SharedSpecValidator{
		spec:      spec,
		fieldPath: fieldPath,
	}
}

// Validate checks invariants that cannot be validated after conversion to v1alpha1.
func (v *V1Beta1SharedSpecValidator) Validate() error {
	return errors.Join(
		v.validatePodTemplateContainers(),
		v.validateFrontendSidecarReference(),
	)
}

func (v *V1Beta1SharedSpecValidator) validatePodTemplateContainers() error {
	if v.spec.PodTemplate == nil {
		return nil
	}

	var errs []error
	containersPath := v.podTemplateContainersPath()
	for i, c := range v.spec.PodTemplate.Spec.Containers {
		if c.Name == nvidiacomv1beta1.MainContainerName {
			continue
		}
		if c.Image == "" {
			errs = append(errs, fmt.Errorf(
				"%s[%d].image is required for non-main container %q",
				containersPath, i, c.Name))
		}
	}

	return errors.Join(errs...)
}

func (v *V1Beta1SharedSpecValidator) validateFrontendSidecarReference() error {
	if v.spec.FrontendSidecar == nil {
		return nil
	}

	containersPath := v.podTemplateContainersPath()
	if v.spec.PodTemplate != nil {
		for _, c := range v.spec.PodTemplate.Spec.Containers {
			if c.Name == *v.spec.FrontendSidecar {
				return nil
			}
		}
	}

	return fmt.Errorf(
		"%s.frontendSidecar %q must match a container name in %s",
		v.fieldPath, *v.spec.FrontendSidecar, containersPath)
}

func (v *V1Beta1SharedSpecValidator) podTemplateContainersPath() string {
	return v.fieldPath + ".podTemplate.spec.containers"
}
