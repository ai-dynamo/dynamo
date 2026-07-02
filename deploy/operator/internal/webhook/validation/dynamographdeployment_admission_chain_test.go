/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
	"encoding/json"
	"fmt"
	"path/filepath"
	goruntime "runtime"
	"strings"
	"testing"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	admissionv1 "k8s.io/api/admission/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apiextensions-apiserver/pkg/apis/apiextensions"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	crdvalidation "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/validation"
	apiextensionsvalidation "k8s.io/apiextensions-apiserver/pkg/apiserver/validation"
	apitest "k8s.io/apiextensions-apiserver/pkg/test"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	k8sptr "k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/webhook/admission"
)

type dgdAdmissionResource string

const (
	dgdAdmissionResourceDGD     dgdAdmissionResource = "dgd"
	dgdAdmissionResourceDCD     dgdAdmissionResource = "dcd"
	dgdAdmissionWorkerName                           = "worker"
	dgdAdmissionUpperWorkerName                      = "WORKER"
)

type dgdRequestValidator struct {
	schemaValidator apiextensionsvalidation.SchemaValidator
	celValidator    apitest.CELValidateFunc
}

func TestDynamoGraphDeploymentAdmissionChain(t *testing.T) {
	requestValidators := dgdAdmissionRequestValidators(t)

	tests := []struct {
		name      string
		resource  dgdAdmissionResource
		version   string
		operation admissionv1.Operation
		current   map[string]any
		old       map[string]any

		wantRequestErr string
		wantWebhookErr string
	}{
		{
			name:      "v1beta1 empty component name is rejected before create webhook",
			resource:  dgdAdmissionResourceDGD,
			version:   "v1beta1",
			operation: admissionv1.Create,
			current: betaDGDAdmissionRequest(t, func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Spec.Components[1].ComponentName = ""
			}),
			wantRequestErr: `spec.components[1].name: Invalid value: "": spec.components[1].name in body should be at least 1 chars long`,
		},
		{
			name:      "v1beta1 case-insensitive component names are rejected before create webhook",
			resource:  dgdAdmissionResourceDGD,
			version:   "v1beta1",
			operation: admissionv1.Create,
			current: betaDGDAdmissionRequest(t, func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Spec.Components[0].ComponentName = dgdAdmissionWorkerName
				dgd.Spec.Components[1].ComponentName = dgdAdmissionUpperWorkerName
			}),
			wantRequestErr: "spec.components: Invalid value: component names must be unique case-insensitively",
		},
		{
			name:      "v1alpha1 empty service key reaches create webhook",
			resource:  dgdAdmissionResourceDGD,
			version:   "v1alpha1",
			operation: admissionv1.Create,
			current:   alphaDGDAdmissionRequestWithServiceNames(t, ""),
		},
		{
			name:      "v1alpha1 case-insensitive service keys reach create webhook",
			resource:  dgdAdmissionResourceDGD,
			version:   "v1alpha1",
			operation: admissionv1.Create,
			current:   alphaDGDAdmissionRequestWithServiceNames(t, dgdAdmissionWorkerName, dgdAdmissionUpperWorkerName),
		},
		{
			name:      "v1beta1 empty compilation cache PVC name is rejected before webhook",
			resource:  dgdAdmissionResourceDGD,
			version:   "v1beta1",
			operation: admissionv1.Create,
			current: betaDGDAdmissionRequest(t, func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				betaWorkerComponent(dgd).CompilationCache = &nvidiacomv1beta1.CompilationCacheConfig{}
			}),
			wantRequestErr: `spec.components[1].compilationCache.pvcName: Invalid value: "": spec.components[1].compilationCache.pvcName in body should be at least 1 chars long`,
		},
		{
			name:      "v1alpha1 compilation cache mount with explicit empty name reaches webhook",
			resource:  dgdAdmissionResourceDGD,
			version:   "v1alpha1",
			operation: admissionv1.Create,
			current: alphaDGDAdmissionRequestWithRawVolumeMount(t, map[string]any{
				"name":                  "",
				"useAsCompilationCache": true,
			}),
		},
		{
			name:      "v1alpha1 regular volume mount with explicit empty name retains historical behavior",
			resource:  dgdAdmissionResourceDGD,
			version:   "v1alpha1",
			operation: admissionv1.Create,
			current: alphaDGDAdmissionRequestWithRawVolumeMount(t, map[string]any{
				"name":       "",
				"mountPoint": "/data",
			}),
		},
		{
			name:      "v1beta1 sidecar without image is rejected by CEL before webhook",
			resource:  dgdAdmissionResourceDGD,
			version:   "v1beta1",
			operation: admissionv1.Create,
			current: betaDGDAdmissionRequest(t, func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				betaWorkerComponent(dgd).PodTemplate = &corev1.PodTemplateSpec{Spec: corev1.PodSpec{
					Containers: []corev1.Container{{Name: consts.MainContainerName}, {Name: "metrics"}},
				}}
			}),
			wantRequestErr: "spec.components[1].podTemplate.spec.containers[1]: Invalid value: sidecar containers must specify a non-empty image",
		},
		{
			name:      "v1beta1 init container without image is rejected by CEL before webhook",
			resource:  dgdAdmissionResourceDGD,
			version:   "v1beta1",
			operation: admissionv1.Create,
			current: betaDGDAdmissionRequest(t, func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				betaWorkerComponent(dgd).PodTemplate = &corev1.PodTemplateSpec{Spec: corev1.PodSpec{
					Containers:     []corev1.Container{{Name: consts.MainContainerName}},
					InitContainers: []corev1.Container{{Name: "prepare"}},
				}}
			}),
			wantRequestErr: "spec.components[1].podTemplate.spec.initContainers[0]: Invalid value: init containers must specify a non-empty image",
		},
		{
			name:      "v1alpha1 sidecar without image reaches webhook after conversion",
			resource:  dgdAdmissionResourceDGD,
			version:   "v1alpha1",
			operation: admissionv1.Create,
			current: alphaDGDAdmissionRequest(t, func(dgd *nvidiacomv1alpha1.DynamoGraphDeployment) {
				dgd.Spec.Services[dgdAdmissionWorkerName].ExtraPodSpec = &nvidiacomv1alpha1.ExtraPodSpec{PodSpec: &corev1.PodSpec{
					Containers: []corev1.Container{{Name: "metrics"}},
				}}
			}),
		},
		{
			name:      "v1alpha1 init container without image reaches webhook after conversion",
			resource:  dgdAdmissionResourceDGD,
			version:   "v1alpha1",
			operation: admissionv1.Create,
			current: alphaDGDAdmissionRequest(t, func(dgd *nvidiacomv1alpha1.DynamoGraphDeployment) {
				dgd.Spec.Services[dgdAdmissionWorkerName].ExtraPodSpec = &nvidiacomv1alpha1.ExtraPodSpec{PodSpec: &corev1.PodSpec{
					InitContainers: []corev1.Container{{Name: "prepare"}},
				}}
			}),
		},
		{
			name:      "v1alpha1 frontend sidecar with empty image reaches webhook after conversion",
			resource:  dgdAdmissionResourceDGD,
			version:   "v1alpha1",
			operation: admissionv1.Create,
			current: alphaDGDAdmissionRequest(t, func(dgd *nvidiacomv1alpha1.DynamoGraphDeployment) {
				dgd.Spec.Services[dgdAdmissionWorkerName].FrontendSidecar = &nvidiacomv1alpha1.FrontendSidecarSpec{}
			}),
		},
		{
			name:      "invalid v1beta1 pod template backend annotation is rejected by CEL before webhook",
			resource:  dgdAdmissionResourceDGD,
			version:   "v1beta1",
			operation: admissionv1.Create,
			current: betaDGDAdmissionRequest(t, func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				betaWorkerComponent(dgd).PodTemplate = &corev1.PodTemplateSpec{
					ObjectMeta: metav1.ObjectMeta{Annotations: map[string]string{
						consts.KubeAnnotationVLLMDistributedExecutorBackend: "invalid",
					}},
					Spec: corev1.PodSpec{Containers: []corev1.Container{{Name: consts.MainContainerName}}},
				}
			}),
			wantRequestErr: "spec.components[1].podTemplate: Invalid value: podTemplate backend annotation must be mp or ray, case-insensitively",
		},
		{
			name:      "invalid v1alpha1 extra pod metadata annotation reaches webhook after conversion",
			resource:  dgdAdmissionResourceDGD,
			version:   "v1alpha1",
			operation: admissionv1.Create,
			current: alphaDGDAdmissionRequest(t, func(dgd *nvidiacomv1alpha1.DynamoGraphDeployment) {
				dgd.Spec.Services[dgdAdmissionWorkerName].ExtraPodMetadata = &nvidiacomv1alpha1.ExtraPodMetadata{
					Annotations: map[string]string{consts.KubeAnnotationVLLMDistributedExecutorBackend: "invalid"},
				}
			}),
		},
		{
			name:      "invalid v1alpha1 service annotation remains rejected by webhook",
			resource:  dgdAdmissionResourceDGD,
			version:   "v1alpha1",
			operation: admissionv1.Create,
			current: alphaDGDAdmissionRequest(t, func(dgd *nvidiacomv1alpha1.DynamoGraphDeployment) {
				dgd.Spec.Services[dgdAdmissionWorkerName].Annotations = map[string]string{
					consts.KubeAnnotationVLLMDistributedExecutorBackend: "invalid",
				}
			}),
			wantWebhookErr: `spec.services[worker].annotations[nvidia.com/vllm-distributed-executor-backend] has invalid value "invalid": must be "mp" or "ray"`,
		},
		{
			name:      "v1beta1 frontend sidecar must still reference an existing container",
			resource:  dgdAdmissionResourceDGD,
			version:   "v1beta1",
			operation: admissionv1.Create,
			current: betaDGDAdmissionRequest(t, func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				worker := betaWorkerComponent(dgd)
				worker.FrontendSidecar = k8sptr.To("missing")
				worker.PodTemplate = &corev1.PodTemplateSpec{Spec: corev1.PodSpec{
					Containers: []corev1.Container{{Name: consts.MainContainerName}},
				}}
			}),
			wantWebhookErr: `spec.components[worker].frontendSidecar "missing" does not match any podTemplate.spec.containers name`,
		},
		{
			name:      "valid v1alpha1 create reaches webhook",
			resource:  dgdAdmissionResourceDGD,
			version:   "v1alpha1",
			operation: admissionv1.Create,
			current:   alphaDGDAdmissionRequest(t, nil),
		},
		{
			name:      "valid v1beta1 create reaches webhook",
			resource:  dgdAdmissionResourceDGD,
			version:   "v1beta1",
			operation: admissionv1.Create,
			current: betaDGDAdmissionRequest(t, func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				betaWorkerComponent(dgd).PodTemplate = validBetaPodTemplate()
			}),
		},
		{
			name:      "v1beta1 pod template container counts are not artificially bounded",
			resource:  dgdAdmissionResourceDGD,
			version:   "v1beta1",
			operation: admissionv1.Create,
			current: betaDGDAdmissionRequest(t, func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				podTemplate := validBetaPodTemplate()
				for i := range 32 {
					podTemplate.Spec.Containers = append(podTemplate.Spec.Containers, corev1.Container{
						Name:  fmt.Sprintf("sidecar-%d", i),
						Image: "sidecar:latest",
					})
					podTemplate.Spec.InitContainers = append(podTemplate.Spec.InitContainers, corev1.Container{
						Name:  fmt.Sprintf("init-%d", i),
						Image: "init:latest",
					})
				}
				betaWorkerComponent(dgd).PodTemplate = podTemplate
			}),
		},
		{
			name:      "v1beta1 case-insensitive component names are rejected before update webhook",
			resource:  dgdAdmissionResourceDGD,
			version:   "v1beta1",
			operation: admissionv1.Update,
			old: betaDGDAdmissionRequest(t, func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Spec.Components[0].ComponentName = dgdAdmissionWorkerName
				dgd.Spec.Components[1].ComponentName = dgdAdmissionUpperWorkerName
			}),
			current: betaDGDAdmissionRequest(t, func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Spec.Components[0].ComponentName = dgdAdmissionWorkerName
				dgd.Spec.Components[1].ComponentName = dgdAdmissionUpperWorkerName
			}),
			wantRequestErr: "spec.components: Invalid value: component names must be unique case-insensitively",
		},
		{
			name:      "unrelated update of v1alpha1 object with empty service key reaches webhook",
			resource:  dgdAdmissionResourceDGD,
			version:   "v1alpha1",
			operation: admissionv1.Update,
			old:       alphaDGDAdmissionRequestWithServiceNames(t, ""),
			current: dgdAdmissionRequestWithLabel(
				t,
				alphaDGDAdmissionRequestWithServiceNames(t, ""),
			),
		},
		{
			name:      "unrelated update of v1alpha1 object with case-insensitive service keys reaches webhook",
			resource:  dgdAdmissionResourceDGD,
			version:   "v1alpha1",
			operation: admissionv1.Update,
			old:       alphaDGDAdmissionRequestWithServiceNames(t, dgdAdmissionWorkerName, dgdAdmissionUpperWorkerName),
			current: dgdAdmissionRequestWithLabel(
				t,
				alphaDGDAdmissionRequestWithServiceNames(t, dgdAdmissionWorkerName, dgdAdmissionUpperWorkerName),
			),
		},
		{
			name:      "unrelated update of v1alpha1 object with historical converted shapes reaches webhook",
			resource:  dgdAdmissionResourceDGD,
			version:   "v1alpha1",
			operation: admissionv1.Update,
			old:       alphaDGDAdmissionHistoricalShapes(t),
			current: dgdAdmissionRequestWithLabel(
				t,
				alphaDGDAdmissionHistoricalShapes(t),
			),
		},
		{
			name:      "valid v1beta1 update reaches webhook",
			resource:  dgdAdmissionResourceDGD,
			version:   "v1beta1",
			operation: admissionv1.Update,
			old:       betaDGDAdmissionRequest(t, nil),
			current: dgdAdmissionRequestWithLabel(
				t,
				betaDGDAdmissionRequest(t, nil),
			),
		},
		{
			name:      "v1beta1 DCD sidecar without image is rejected by shared CEL",
			resource:  dgdAdmissionResourceDCD,
			version:   "v1beta1",
			operation: admissionv1.Create,
			current: betaDCDAdmissionRequest(t, func(dcd *nvidiacomv1beta1.DynamoComponentDeployment) {
				dcd.Spec.PodTemplate = &corev1.PodTemplateSpec{Spec: corev1.PodSpec{
					Containers: []corev1.Container{{Name: consts.MainContainerName}, {Name: "metrics"}},
				}}
			}),
			wantRequestErr: "spec.podTemplate.spec.containers[1]: Invalid value: sidecar containers must specify a non-empty image",
		},
		{
			name:      "v1beta1 DCD init container without image is rejected by shared CEL",
			resource:  dgdAdmissionResourceDCD,
			version:   "v1beta1",
			operation: admissionv1.Create,
			current: betaDCDAdmissionRequest(t, func(dcd *nvidiacomv1beta1.DynamoComponentDeployment) {
				dcd.Spec.PodTemplate = &corev1.PodTemplateSpec{Spec: corev1.PodSpec{
					Containers:     []corev1.Container{{Name: consts.MainContainerName}},
					InitContainers: []corev1.Container{{Name: "prepare"}},
				}}
			}),
			wantRequestErr: "spec.podTemplate.spec.initContainers[0]: Invalid value: init containers must specify a non-empty image",
		},
		{
			name:      "v1beta1 DCD invalid backend annotation is rejected by shared CEL",
			resource:  dgdAdmissionResourceDCD,
			version:   "v1beta1",
			operation: admissionv1.Create,
			current: betaDCDAdmissionRequest(t, func(dcd *nvidiacomv1beta1.DynamoComponentDeployment) {
				dcd.Spec.PodTemplate = &corev1.PodTemplateSpec{
					ObjectMeta: metav1.ObjectMeta{Annotations: map[string]string{
						consts.KubeAnnotationVLLMDistributedExecutorBackend: "invalid",
					}},
					Spec: corev1.PodSpec{Containers: []corev1.Container{{Name: consts.MainContainerName}}},
				}
			}),
			wantRequestErr: "spec.podTemplate: Invalid value: podTemplate backend annotation must be mp or ray, case-insensitively",
		},
		{
			name:      "valid v1beta1 DCD reaches converted alpha webhook",
			resource:  dgdAdmissionResourceDCD,
			version:   "v1beta1",
			operation: admissionv1.Create,
			current: betaDCDAdmissionRequest(t, func(dcd *nvidiacomv1beta1.DynamoComponentDeployment) {
				dcd.Spec.PodTemplate = validBetaPodTemplate()
			}),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			validator := requestValidators[tt.resource][tt.version]
			requestErrs := validator.validate(tt.current, tt.old)
			if tt.wantRequestErr != "" {
				assertDGDAdmissionRequestError(t, requestErrs, tt.wantRequestErr)
				return
			}
			if len(requestErrs) != 0 {
				t.Fatalf("request-version validation errors = %v, want none", requestErrs)
			}

			_, err := invokeDGDAdmissionWebhook(t, tt.resource, tt.version, tt.operation, tt.old, tt.current)
			if tt.wantWebhookErr == "" {
				if err != nil {
					t.Fatalf("webhook error = %v, want none", err)
				}
			} else if err == nil || !strings.Contains(err.Error(), tt.wantWebhookErr) {
				t.Fatalf("webhook error = %v, want one containing %q", err, tt.wantWebhookErr)
			}
		})
	}
}

func dgdAdmissionRequestValidators(t *testing.T) map[dgdAdmissionResource]map[string]*dgdRequestValidator {
	t.Helper()
	_, thisFile, _, ok := goruntime.Caller(0)
	if !ok {
		t.Fatal("runtime.Caller(0) failed")
	}
	crdDir := filepath.Join(filepath.Dir(thisFile), "../../../config/crd/bases")
	return map[dgdAdmissionResource]map[string]*dgdRequestValidator{
		dgdAdmissionResourceDGD: loadDGDRequestValidators(t, filepath.Join(crdDir, "nvidia.com_dynamographdeployments.yaml")),
		dgdAdmissionResourceDCD: loadDGDRequestValidators(t, filepath.Join(crdDir, "nvidia.com_dynamocomponentdeployments.yaml")),
	}
}

func loadDGDRequestValidators(t *testing.T, crdPath string) map[string]*dgdRequestValidator {
	t.Helper()
	crd := apitest.MustLoadManifest[apiextensionsv1.CustomResourceDefinition](t, crdPath)
	internalCRD := &apiextensions.CustomResourceDefinition{}
	if err := apiextensionsv1.Convert_v1_CustomResourceDefinition_To_apiextensions_CustomResourceDefinition(crd, internalCRD, nil); err != nil {
		t.Fatalf("convert CRD %s: %v", crdPath, err)
	}

	// Match API server initialization before exercising its static CEL cost budgets.
	internalCRD.Spec.Conversion.WebhookClientConfig.Service.Port = 443
	for _, version := range internalCRD.Spec.Versions {
		if version.Storage {
			internalCRD.Status.StoredVersions = append(internalCRD.Status.StoredVersions, version.Name)
		}
	}
	if errs := crdvalidation.ValidateCustomResourceDefinition(t.Context(), internalCRD); len(errs) != 0 {
		t.Fatalf("validate CRD %s: %v", crdPath, errs)
	}

	celValidators := apitest.VersionValidatorsFromFile(t, crdPath)
	validators := make(map[string]*dgdRequestValidator, len(crd.Spec.Versions))
	for _, version := range crd.Spec.Versions {
		var internalSchema apiextensions.JSONSchemaProps
		if err := apiextensionsv1.Convert_v1_JSONSchemaProps_To_apiextensions_JSONSchemaProps(
			version.Schema.OpenAPIV3Schema,
			&internalSchema,
			nil,
		); err != nil {
			t.Fatalf("convert schema for %s in %s: %v", version.Name, crdPath, err)
		}
		schemaValidator, _, err := apiextensionsvalidation.NewSchemaValidator(&internalSchema)
		if err != nil {
			t.Fatalf("compile schema validator for %s in %s: %v", version.Name, crdPath, err)
		}
		validators[version.Name] = &dgdRequestValidator{
			schemaValidator: schemaValidator,
			celValidator:    celValidators[version.Name],
		}
	}
	return validators
}

func (v *dgdRequestValidator) validate(current, old map[string]any) field.ErrorList {
	var errs field.ErrorList
	if old == nil {
		errs = apiextensionsvalidation.ValidateCustomResource(nil, current, v.schemaValidator)
	} else {
		errs = apiextensionsvalidation.ValidateCustomResourceUpdate(nil, current, old, v.schemaValidator)
	}
	if len(errs) != 0 {
		return errs
	}
	return v.celValidator(current, old)
}

func invokeDGDAdmissionWebhook(
	t *testing.T,
	resource dgdAdmissionResource,
	version string,
	operation admissionv1.Operation,
	oldRequest, currentRequest map[string]any,
) (admission.Warnings, error) {
	t.Helper()
	switch resource {
	case dgdAdmissionResourceDGD:
		handler := NewDynamoGraphDeploymentHandler(nil, "", false)
		ctx := dgdAdmissionContext(operation, nvidiacomv1beta1.DynamoGraphDeploymentGVK)
		current := dgdAdmissionBetaDGD(t, version, currentRequest)
		if operation == admissionv1.Create {
			return handler.ValidateCreate(ctx, current)
		}
		return handler.ValidateUpdate(ctx, dgdAdmissionBetaDGD(t, version, oldRequest), current)
	case dgdAdmissionResourceDCD:
		handler := NewDynamoComponentDeploymentHandler()
		ctx := dgdAdmissionContext(operation, nvidiacomv1alpha1.DynamoComponentDeploymentGVK)
		current := dgdAdmissionAlphaDCD(t, version, currentRequest)
		if operation == admissionv1.Create {
			return handler.ValidateCreate(ctx, current)
		}
		return handler.ValidateUpdate(ctx, dgdAdmissionAlphaDCD(t, version, oldRequest), current)
	default:
		t.Fatalf("unsupported admission resource %q", resource)
		return nil, nil
	}
}

func dgdAdmissionBetaDGD(t *testing.T, version string, request map[string]any) *nvidiacomv1beta1.DynamoGraphDeployment {
	t.Helper()
	if version == "v1beta1" {
		out := &nvidiacomv1beta1.DynamoGraphDeployment{}
		decodeDGDAdmissionRequest(t, request, out)
		return out
	}
	alpha := &nvidiacomv1alpha1.DynamoGraphDeployment{}
	decodeDGDAdmissionRequest(t, request, alpha)
	out := &nvidiacomv1beta1.DynamoGraphDeployment{}
	if err := alpha.ConvertTo(out); err != nil {
		t.Fatalf("convert v1alpha1 DGD to v1beta1: %v", err)
	}
	return out
}

func dgdAdmissionAlphaDCD(t *testing.T, version string, request map[string]any) *nvidiacomv1alpha1.DynamoComponentDeployment {
	t.Helper()
	if version == "v1alpha1" {
		out := &nvidiacomv1alpha1.DynamoComponentDeployment{}
		decodeDGDAdmissionRequest(t, request, out)
		return out
	}
	beta := &nvidiacomv1beta1.DynamoComponentDeployment{}
	decodeDGDAdmissionRequest(t, request, beta)
	out := &nvidiacomv1alpha1.DynamoComponentDeployment{}
	if err := out.ConvertFrom(beta); err != nil {
		t.Fatalf("convert v1beta1 DCD to v1alpha1: %v", err)
	}
	return out
}

func betaDGDAdmissionRequest(
	t *testing.T,
	mutate func(*nvidiacomv1beta1.DynamoGraphDeployment),
) map[string]any {
	t.Helper()
	dgd := newBetaDGDForValidation()
	dgd.TypeMeta = metav1.TypeMeta{APIVersion: nvidiacomv1beta1.GroupVersion.String(), Kind: "DynamoGraphDeployment"}
	if mutate != nil {
		mutate(dgd)
	}
	return dgdAdmissionRequestMap(t, dgd)
}

func alphaDGDAdmissionRequest(
	t *testing.T,
	mutate func(*nvidiacomv1alpha1.DynamoGraphDeployment),
) map[string]any {
	t.Helper()
	dgd := newAlphaDGDForCompatibilityValidation()
	dgd.TypeMeta = metav1.TypeMeta{APIVersion: nvidiacomv1alpha1.GroupVersion.String(), Kind: "DynamoGraphDeployment"}
	if mutate != nil {
		mutate(dgd)
	}
	return dgdAdmissionRequestMap(t, dgd)
}

func alphaDGDAdmissionRequestWithServiceNames(t *testing.T, names ...string) map[string]any {
	t.Helper()
	return alphaDGDAdmissionRequest(t, func(dgd *nvidiacomv1alpha1.DynamoGraphDeployment) {
		service := dgd.Spec.Services[dgdAdmissionWorkerName]
		dgd.Spec.Services = make(map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec, len(names))
		for _, name := range names {
			dgd.Spec.Services[name] = service.DeepCopy()
		}
	})
}

func alphaDGDAdmissionRequestWithRawVolumeMount(t *testing.T, mount map[string]any) map[string]any {
	t.Helper()
	request := alphaDGDAdmissionRequest(t, nil)
	dgdAdmissionAlphaService(t, request, dgdAdmissionWorkerName)["volumeMounts"] = []any{mount}
	return request
}

func alphaDGDAdmissionHistoricalShapes(t *testing.T) map[string]any {
	t.Helper()
	request := alphaDGDAdmissionRequest(t, func(dgd *nvidiacomv1alpha1.DynamoGraphDeployment) {
		service := dgd.Spec.Services[dgdAdmissionWorkerName]
		service.ExtraPodMetadata = &nvidiacomv1alpha1.ExtraPodMetadata{Annotations: map[string]string{
			consts.KubeAnnotationVLLMDistributedExecutorBackend: "invalid",
		}}
		service.ExtraPodSpec = &nvidiacomv1alpha1.ExtraPodSpec{PodSpec: &corev1.PodSpec{
			Containers:     []corev1.Container{{Name: "metrics"}},
			InitContainers: []corev1.Container{{Name: "prepare"}},
		}}
		service.FrontendSidecar = &nvidiacomv1alpha1.FrontendSidecarSpec{}
	})
	dgdAdmissionAlphaService(t, request, dgdAdmissionWorkerName)["volumeMounts"] = []any{map[string]any{
		"name":                  "",
		"useAsCompilationCache": true,
	}}
	return request
}

func betaDCDAdmissionRequest(
	t *testing.T,
	mutate func(*nvidiacomv1beta1.DynamoComponentDeployment),
) map[string]any {
	t.Helper()
	dcd := &nvidiacomv1beta1.DynamoComponentDeployment{
		TypeMeta: metav1.TypeMeta{APIVersion: nvidiacomv1beta1.GroupVersion.String(), Kind: "DynamoComponentDeployment"},
		ObjectMeta: metav1.ObjectMeta{
			Name:      dgdAdmissionWorkerName,
			Namespace: "default",
		},
		Spec: nvidiacomv1beta1.DynamoComponentDeploymentSpec{
			BackendFramework: "vllm",
			DynamoComponentDeploymentSharedSpec: nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{
				ComponentName: dgdAdmissionWorkerName,
				ComponentType: nvidiacomv1beta1.ComponentTypeWorker,
			},
		},
	}
	if mutate != nil {
		mutate(dcd)
	}
	return dgdAdmissionRequestMap(t, dcd)
}

func validBetaPodTemplate() *corev1.PodTemplateSpec {
	return &corev1.PodTemplateSpec{
		ObjectMeta: metav1.ObjectMeta{Annotations: map[string]string{
			consts.KubeAnnotationVLLMDistributedExecutorBackend: "RaY",
		}},
		Spec: corev1.PodSpec{
			Containers: []corev1.Container{
				{Name: consts.MainContainerName},
				{Name: "metrics", Image: "metrics:latest"},
			},
			InitContainers: []corev1.Container{{Name: "prepare", Image: "prepare:latest"}},
		},
	}
}

func dgdAdmissionRequestMap(t *testing.T, obj runtime.Object) map[string]any {
	t.Helper()
	data, err := json.Marshal(obj)
	if err != nil {
		t.Fatalf("marshal %T: %v", obj, err)
	}
	var request map[string]any
	if err := json.Unmarshal(data, &request); err != nil {
		t.Fatalf("unmarshal %T request: %v", obj, err)
	}
	delete(request, "status")
	return request
}

func decodeDGDAdmissionRequest(t *testing.T, request map[string]any, into runtime.Object) {
	t.Helper()
	data, err := json.Marshal(request)
	if err != nil {
		t.Fatalf("marshal request: %v", err)
	}
	if err := json.Unmarshal(data, into); err != nil {
		t.Fatalf("decode request into %T: %v", into, err)
	}
}

func dgdAdmissionAlphaService(t *testing.T, request map[string]any, name string) map[string]any {
	t.Helper()
	spec, ok := request["spec"].(map[string]any)
	if !ok {
		t.Fatal("request spec is missing or not an object")
	}
	services, ok := spec["services"].(map[string]any)
	if !ok {
		t.Fatal("request spec.services is missing or not an object")
	}
	service, ok := services[name].(map[string]any)
	if !ok {
		t.Fatalf("request spec.services[%q] is missing or not an object", name)
	}
	return service
}

func dgdAdmissionRequestWithLabel(t *testing.T, request map[string]any) map[string]any {
	t.Helper()
	metadata, ok := request["metadata"].(map[string]any)
	if !ok {
		t.Fatal("request metadata is missing or not an object")
	}
	metadata["labels"] = map[string]any{"updated": "true"}
	return request
}

func assertDGDAdmissionRequestError(t *testing.T, got field.ErrorList, want string) {
	t.Helper()
	if len(got) != 1 {
		t.Fatalf("request-version validation errors = %v, want exactly %q", got, want)
	}
	if got[0].Error() != want {
		t.Fatalf("request-version validation error = %q, want %q", got[0], want)
	}
}
