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
	"context"
	"strings"
	"testing"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"sigs.k8s.io/yaml"
)

func TestBugDCD_BetaPodTemplateSidecarWithoutImageNeedsBetaValidation(t *testing.T) {
	dcd := decodeBetaDCDYAML(t, `
apiVersion: nvidia.com/v1beta1
kind: DynamoComponentDeployment
metadata:
  name: invalid-sidecar
  namespace: default
spec:
  backendFramework: vllm
  name: worker
  podTemplate:
    spec:
      containers:
      - name: main
      - name: metrics
`)

	// The existing equivalent v1alpha1 webhook sees this after conversion and admits it.
	assertAlphaDCDValidationAdmitsBeta(t, dcd)
	assertBetaDCDValidationRejects(t, dcd, `spec.podTemplate.spec.containers[1].image is required for non-main container "metrics"`)
}

func TestBugDCD_BetaFrontendSidecarMustReferencePodTemplateContainer(t *testing.T) {
	dcd := decodeBetaDCDYAML(t, `
apiVersion: nvidia.com/v1beta1
kind: DynamoComponentDeployment
metadata:
  name: invalid-frontend-ref
  namespace: default
spec:
  backendFramework: vllm
  name: worker
  frontendSidecar: sidecar-frontend
  podTemplate:
    spec:
      containers:
      - name: main
`)

	// The existing equivalent v1alpha1 webhook sees this after conversion and admits it.
	assertAlphaDCDValidationAdmitsBeta(t, dcd)
	assertBetaDCDValidationRejects(t, dcd, `spec.frontendSidecar "sidecar-frontend" must match a container name in spec.podTemplate.spec.containers`)
}

func TestBugDGD_BetaPodTemplateSidecarWithoutImageNeedsBetaValidation(t *testing.T) {
	dgd := decodeBetaDGDYAML(t, `
apiVersion: nvidia.com/v1beta1
kind: DynamoGraphDeployment
metadata:
  name: invalid-sidecar
  namespace: default
spec:
  backendFramework: vllm
  components:
  - name: worker
    podTemplate:
      spec:
        containers:
        - name: main
        - name: metrics
`)

	// The existing equivalent v1alpha1 webhook sees this after conversion and admits it.
	assertAlphaDGDValidationAdmitsBeta(t, dgd)
	assertBetaDGDValidationRejects(t, dgd, `spec.components[worker].podTemplate.spec.containers[1].image is required for non-main container "metrics"`)
}

func TestBugDGD_BetaFrontendSidecarMustReferencePodTemplateContainer(t *testing.T) {
	dgd := decodeBetaDGDYAML(t, `
apiVersion: nvidia.com/v1beta1
kind: DynamoGraphDeployment
metadata:
  name: invalid-frontend-ref
  namespace: default
spec:
  backendFramework: vllm
  components:
  - name: frontend
    frontendSidecar: sidecar-frontend
    podTemplate:
      spec:
        containers:
        - name: main
`)

	// The existing equivalent v1alpha1 webhook sees this after conversion and admits it.
	assertAlphaDGDValidationAdmitsBeta(t, dgd)
	assertBetaDGDValidationRejects(t, dgd, `spec.components[frontend].frontendSidecar "sidecar-frontend" must match a container name in spec.components[frontend].podTemplate.spec.containers`)
}

func decodeBetaDCDYAML(t *testing.T, data string) *nvidiacomv1beta1.DynamoComponentDeployment {
	t.Helper()

	var dcd nvidiacomv1beta1.DynamoComponentDeployment
	if err := yaml.Unmarshal([]byte(data), &dcd); err != nil {
		t.Fatalf("decode v1beta1 DCD YAML: %v", err)
	}
	return &dcd
}

func decodeBetaDGDYAML(t *testing.T, data string) *nvidiacomv1beta1.DynamoGraphDeployment {
	t.Helper()

	var dgd nvidiacomv1beta1.DynamoGraphDeployment
	if err := yaml.Unmarshal([]byte(data), &dgd); err != nil {
		t.Fatalf("decode v1beta1 DGD YAML: %v", err)
	}
	return &dgd
}

func assertAlphaDCDValidationAdmitsBeta(t *testing.T, src *nvidiacomv1beta1.DynamoComponentDeployment) {
	t.Helper()

	alpha := &nvidiacomv1alpha1.DynamoComponentDeployment{}
	if err := alpha.ConvertFrom(src); err != nil {
		t.Fatalf("convert beta DCD to alpha: %v", err)
	}
	if _, err := NewDynamoComponentDeploymentValidator(alpha).Validate(context.Background()); err != nil {
		t.Fatalf("alpha-compatible DCD validation rejected beta object: %v", err)
	}
}

func assertAlphaDGDValidationAdmitsBeta(t *testing.T, src *nvidiacomv1beta1.DynamoGraphDeployment) {
	t.Helper()

	alpha := &nvidiacomv1alpha1.DynamoGraphDeployment{}
	if err := alpha.ConvertFrom(src); err != nil {
		t.Fatalf("convert beta DGD to alpha: %v", err)
	}
	if _, err := NewDynamoGraphDeploymentValidator(alpha, false).Validate(context.Background()); err != nil {
		t.Fatalf("alpha-compatible DGD validation rejected beta object: %v", err)
	}
}

func assertBetaDCDValidationRejects(t *testing.T, dcd *nvidiacomv1beta1.DynamoComponentDeployment, want string) {
	t.Helper()

	_, err := NewDynamoComponentDeploymentV1Beta1Validator(dcd).Validate(context.Background())
	if err == nil {
		t.Fatalf("expected v1beta1 DCD validation to reject object")
	}
	if !strings.Contains(err.Error(), want) {
		t.Fatalf("v1beta1 DCD validation error = %q, want to contain %q", err.Error(), want)
	}
}

func assertBetaDGDValidationRejects(t *testing.T, dgd *nvidiacomv1beta1.DynamoGraphDeployment, want string) {
	t.Helper()

	_, err := NewDynamoGraphDeploymentV1Beta1Validator(dgd).Validate(context.Background())
	if err == nil {
		t.Fatalf("expected v1beta1 DGD validation to reject object")
	}
	if !strings.Contains(err.Error(), want) {
		t.Fatalf("v1beta1 DGD validation error = %q, want to contain %q", err.Error(), want)
	}
}
