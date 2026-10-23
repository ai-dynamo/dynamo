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

package validation_test

import (
	"encoding/json"
	"errors"
	"reflect"
	"strings"
	"testing"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	nvidiacomv1beta2 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta2"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	webhookdefaulting "github.com/ai-dynamo/dynamo/deploy/operator/internal/webhook/defaulting"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/utils/ptr"
)

const alternateAdmissionModel = "Qwen/Qwen3-8B"

func TestDynamoGraphDeploymentRequestValidator_Validate(t *testing.T) {
	const deprecatedV1beta1ComponentNames = `{
		"apiVersion":"nvidia.com/v1beta1",
		"kind":"DynamoGraphDeployment",
		"spec":{"components":[
			{"name":"VllmWorker"},
			{"name":"TRTLLMPrefillWorker"},
			{"name":"SGLangDecodeWorker"}
		]}
	}`
	const deprecatedV1alpha1ServiceNames = `{
		"apiVersion":"nvidia.com/v1alpha1",
		"kind":"DynamoGraphDeployment",
		"spec":{"services":{"VllmDecodeWorker":{},"TRTLLMDecodeWorker":{}}}
	}`
	const deprecatedUpdateComponentName = `{
		"apiVersion":"nvidia.com/v1beta1",
		"kind":"DynamoGraphDeployment",
		"spec":{"components":[{"name":"VllmPrefillWorker"}]}
	}`

	discoveredH200 := webhookdefaulting.DiscoveredHardware{SKU: "h200_sxm", GPUs: 8}

	tests := []struct {
		name               string
		request            runtime.Object
		oldRequest         runtime.Object
		gpuDiscovery       bool
		seedWithoutWebhook bool
		// discovered and discoveryErr are what cluster GPU discovery reports while the case
		// runs; beforeUpdate may change them between the seed and the update.
		discovered   webhookdefaulting.DiscoveredHardware
		discoveryErr error
		beforeUpdate func(*testing.T)
		// check inspects the object the API server returned.
		check         func(*testing.T, *unstructured.Unstructured)
		wantImage     string
		wantSchemaErr string
		wantCELErr    string
		wantStatusErr string
		wantWebhook   []string
		wantWarnings  []string
	}{
		// Source-version schema, CEL, and conversion boundaries.
		{
			name: "valid v1beta1 request is defaulted",
			request: betaDGDRForAdmission(func(request *nvidiacomv1beta1.DynamoGraphDeploymentRequest) {
				request.Spec.Image = ""
				request.Spec.RuntimeVersionOverride = ""
			}),
			gpuDiscovery: true,
			wantImage:    "nvcr.io/nvidia/ai-dynamo/dynamo-planner:1.6.0",
		},
		{
			name: "custom image requires runtime version override",
			request: betaDGDRForAdmission(func(request *nvidiacomv1beta1.DynamoGraphDeploymentRequest) {
				request.Spec.RuntimeVersionOverride = ""
			}),
			gpuDiscovery: true,
			wantWebhook: []string{
				"spec.runtimeVersionOverride: Required value: is required when spec.image has no parseable semantic-version tag",
			},
		},
		{
			name: "runtime version override must be canonical semver core",
			request: betaDGDRForAdmission(func(request *nvidiacomv1beta1.DynamoGraphDeploymentRequest) {
				request.Spec.RuntimeVersionOverride = "1.2"
			}),
			gpuDiscovery:  true,
			wantSchemaErr: `spec.runtimeVersionOverride: Invalid value: "1.2": spec.runtimeVersionOverride in body should match '^(0|[1-9][0-9]{0,3})\.(0|[1-9][0-9]{0,3})\.(0|[1-9][0-9]{0,3})$'`,
		},
		{
			name:         "valid v1alpha1 request converts through the production path",
			request:      alphaDGDRForAdmission(nil),
			gpuDiscovery: true,
		},
		{
			name:    "v1beta2 create bypasses legacy v1beta1 validation",
			request: beta2DGDRForAdmission(setBeta2AdmissionHardware),
		},
		{
			name:       "v1beta2 update bypasses legacy v1beta1 validation",
			oldRequest: beta2DGDRForAdmission(setBeta2AdmissionHardware),
			request: beta2DGDRForAdmission(func(request *nvidiacomv1beta2.DynamoGraphDeploymentRequest) {
				setBeta2AdmissionHardware(request)
				request.Labels = map[string]string{"updated": "true"}
			}),
		},
		{
			name: "v1beta1 empty native representation is rejected by envelope validation",
			request: betaDGDRForAdmission(func(request *nvidiacomv1beta1.DynamoGraphDeploymentRequest) {
				request.Spec.Model = ""
			}),
			gpuDiscovery: true,
			wantCELErr:   `spec: Invalid value: exactly one of native v1beta1 fields or v1beta2 must be specified`,
		},
		{
			name: "v1beta2 envelope rejects runtime version override",
			request: betaDGDRForAdmission(func(request *nvidiacomv1beta1.DynamoGraphDeploymentRequest) {
				request.Spec = nvidiacomv1beta1.DynamoGraphDeploymentRequestSpec{
					RuntimeVersionOverride: "1.1.0",
					V1Beta2: &apiextensionsv1.JSON{
						Raw: []byte(`{"model":{"name":"Qwen/Qwen3-0.6B"}}`),
					},
				}
			}),
			gpuDiscovery: true,
			wantCELErr:   `spec: Invalid value: v1beta2 is mutually exclusive with native v1beta1 fields`,
		},
		{
			name: "v1beta1 SLA optimization enum is rejected by source schema",
			request: betaDGDRForAdmission(func(request *nvidiacomv1beta1.DynamoGraphDeploymentRequest) {
				optimization := nvidiacomv1beta1.OptimizationType("cost")
				request.Spec.SLA = &nvidiacomv1beta1.SLASpec{OptimizationType: &optimization}
			}),
			gpuDiscovery:  true,
			wantSchemaErr: `spec.sla.optimizationType: Unsupported value: "cost": supported values: "latency", "throughput"`,
		},
		{
			name: "v1alpha1 backend enum is rejected by source schema before conversion",
			request: alphaDGDRForAdmission(func(request *nvidiacomv1alpha1.DynamoGraphDeploymentRequest) {
				request.Spec.Backend = "unknown"
			}),
			gpuDiscovery:  true,
			wantSchemaErr: `spec.backend: Unsupported value: "unknown": supported values: "auto", "vllm", "sglang", "trtllm"`,
		},

		// Structural create rules.
		{
			name: "DGD-only metadata annotations are ignored",
			request: betaDGDRForAdmission(func(request *nvidiacomv1beta1.DynamoGraphDeploymentRequest) {
				request.Annotations = map[string]string{consts.KubeAnnotationDynamoOperatorOriginVersion: "not-semver"}
			}),
			gpuDiscovery: true,
		},
		{
			name: "thorough search requires a concrete backend",
			request: betaDGDRForAdmission(func(request *nvidiacomv1beta1.DynamoGraphDeploymentRequest) {
				request.Spec.Backend = nvidiacomv1beta1.BackendTypeAuto
				request.Spec.SearchStrategy = nvidiacomv1beta1.SearchStrategyThorough
			}),
			gpuDiscovery: true,
			wantWebhook: []string{
				`spec.searchStrategy: Invalid value: "thorough": is incompatible with spec.backend "auto"; set spec.backend to a specific backend (sglang, trtllm, or vllm)`,
			},
		},
		{
			name:         "GPU discovery permits omitted hardware",
			request:      betaDGDRForAdmission(nil),
			gpuDiscovery: true,
		},
		{
			name:    "disabled GPU discovery requires manual hardware",
			request: betaDGDRForAdmission(nil),
			wantWebhook: []string{
				"spec.hardware: Required value: GPU hardware configuration is required when GPU discovery is disabled",
			},
		},
		{
			name: "manual hardware permits disabled GPU discovery",
			request: betaDGDRForAdmission(func(request *nvidiacomv1beta1.DynamoGraphDeploymentRequest) {
				request.Spec.Hardware = &nvidiacomv1beta1.HardwareSpec{GPUSKU: nvidiacomv1beta1.GPUSKUTypeH100SXM}
			}),
		},
		{
			name: "B300 hardware is admitted",
			request: betaDGDRForAdmission(func(request *nvidiacomv1beta1.DynamoGraphDeploymentRequest) {
				request.Spec.Hardware = &nvidiacomv1beta1.HardwareSpec{GPUSKU: nvidiacomv1beta1.GPUSKUTypeB300SXM}
			}),
		},
		{
			name: "Vera Rubin NVL72 hardware is admitted",
			request: betaDGDRForAdmission(func(request *nvidiacomv1beta1.DynamoGraphDeploymentRequest) {
				request.Spec.Hardware = &nvidiacomv1beta1.HardwareSpec{GPUSKU: nvidiacomv1beta1.GPUSKUTypeVRNVL72}
			}),
		},
		{
			name: "canonical GB200 hardware is admitted",
			request: betaDGDRForAdmission(func(request *nvidiacomv1beta1.DynamoGraphDeploymentRequest) {
				request.Spec.Hardware = &nvidiacomv1beta1.HardwareSpec{GPUSKU: nvidiacomv1beta1.GPUSKUTypeGB200}
			}),
		},
		{
			name: "independent create failures aggregate in API declaration order",
			request: betaDGDRForAdmission(func(request *nvidiacomv1beta1.DynamoGraphDeploymentRequest) {
				request.Spec.Backend = nvidiacomv1beta1.BackendTypeAuto
				request.Spec.SearchStrategy = nvidiacomv1beta1.SearchStrategyThorough
			}),
			wantWebhook: []string{
				"spec.hardware: Required value: GPU hardware configuration is required when GPU discovery is disabled",
				`spec.searchStrategy: Invalid value: "thorough": is incompatible with spec.backend "auto"; set spec.backend to a specific backend (sglang, trtllm, or vllm)`,
			},
		},

		// Compatibility warnings.
		{
			name: "deprecated GB200 SXM SKU warns",
			request: betaDGDRForAdmission(func(request *nvidiacomv1beta1.DynamoGraphDeploymentRequest) {
				//nolint:staticcheck // SA1019: Exercise admission of the deprecated API value.
				request.Spec.Hardware = &nvidiacomv1beta1.HardwareSpec{GPUSKU: nvidiacomv1beta1.GPUSKUTypeGB200SXM}
			}),
			gpuDiscovery: true,
			wantWarnings: []string{
				`spec.hardware.gpuSku: "gb200_sxm" is deprecated; use "gb200". The legacy value will be removed in a future release`,
			},
		},
		{
			name: "deprecated v1beta1 DGD override component names warn",
			request: betaDGDRForAdmission(func(request *nvidiacomv1beta1.DynamoGraphDeploymentRequest) {
				request.Spec.Overrides = &nvidiacomv1beta1.OverridesSpec{
					DGD: &runtime.RawExtension{Raw: []byte(deprecatedV1beta1ComponentNames)},
				}
			}),
			gpuDiscovery: true,
			wantWarnings: []string{
				`spec.overrides.dgd.spec.components[name=SGLangDecodeWorker]: override target "SGLangDecodeWorker" is deprecated; use worker (aggregate) or decode (disaggregated). Legacy-name translation will be removed in a future release`,
				`spec.overrides.dgd.spec.components[name=TRTLLMPrefillWorker]: override target "TRTLLMPrefillWorker" is deprecated; use prefill. Legacy-name translation will be removed in a future release`,
				`spec.overrides.dgd.spec.components[name=VllmWorker]: override target "VllmWorker" is deprecated; use worker. Legacy-name translation will be removed in a future release`,
			},
		},
		{
			name: "deprecated v1alpha1 DGD override service names warn",
			request: betaDGDRForAdmission(func(request *nvidiacomv1beta1.DynamoGraphDeploymentRequest) {
				request.Spec.Overrides = &nvidiacomv1beta1.OverridesSpec{
					DGD: &runtime.RawExtension{Raw: []byte(deprecatedV1alpha1ServiceNames)},
				}
			}),
			gpuDiscovery: true,
			wantWarnings: []string{
				`spec.overrides.dgd.spec.services.TRTLLMDecodeWorker: override target "TRTLLMDecodeWorker" is deprecated; use decode. Legacy-name translation will be removed in a future release`,
				`spec.overrides.dgd.spec.services.VllmDecodeWorker: override target "VllmDecodeWorker" is deprecated; use worker (aggregate) or decode (disaggregated). Legacy-name translation will be removed in a future release`,
			},
		},

		// Structural update rules.
		{
			name:       "deprecated DGD override component name warns on update",
			oldRequest: betaDGDRForAdmission(nil),
			request: betaDGDRForAdmission(func(request *nvidiacomv1beta1.DynamoGraphDeploymentRequest) {
				request.Spec.Overrides = &nvidiacomv1beta1.OverridesSpec{
					DGD: &runtime.RawExtension{Raw: []byte(deprecatedUpdateComponentName)},
				}
			}),
			gpuDiscovery: true,
			wantWarnings: []string{
				`spec.overrides.dgd.spec.components[name=VllmPrefillWorker]: override target "VllmPrefillWorker" is deprecated; use prefill. Legacy-name translation will be removed in a future release`,
			},
		},
		{
			name: "unchanged spec is accepted during profiling",
			oldRequest: betaDGDRForAdmission(func(request *nvidiacomv1beta1.DynamoGraphDeploymentRequest) {
				request.Status.Phase = nvidiacomv1beta1.DGDRPhaseProfiling
			}),
			request:      betaDGDRForAdmission(nil),
			gpuDiscovery: true,
		},
		{
			name: "spec update is rejected during profiling",
			oldRequest: betaDGDRForAdmission(func(request *nvidiacomv1beta1.DynamoGraphDeploymentRequest) {
				request.Status.Phase = nvidiacomv1beta1.DGDRPhaseProfiling
			}),
			request: betaDGDRForAdmission(func(request *nvidiacomv1beta1.DynamoGraphDeploymentRequest) {
				request.Spec.Model = alternateAdmissionModel
			}),
			gpuDiscovery: true,
			wantWebhook: []string{
				`spec: Forbidden: updates are forbidden while the resource is in phase "Profiling"; delete and recreate the resource to change its spec`,
			},
		},
		{
			name: "auto apply can be enabled after reviewing a ready request",
			oldRequest: betaDGDRForAdmission(func(request *nvidiacomv1beta1.DynamoGraphDeploymentRequest) {
				request.Spec.AutoApply = ptr.To(false)
				request.Status.Phase = nvidiacomv1beta1.DGDRPhaseReady
			}),
			request: betaDGDRForAdmission(func(request *nvidiacomv1beta1.DynamoGraphDeploymentRequest) {
				request.Spec.AutoApply = ptr.To(true)
				request.Status.Phase = nvidiacomv1beta1.DGDRPhaseReady
			}),
			gpuDiscovery: true,
		},
		{
			name:               "auto apply activation can add a missing runtime version override",
			seedWithoutWebhook: true,
			oldRequest: betaDGDRForAdmission(func(request *nvidiacomv1beta1.DynamoGraphDeploymentRequest) {
				request.Spec.AutoApply = ptr.To(false)
				request.Spec.RuntimeVersionOverride = ""
				request.Status.Phase = nvidiacomv1beta1.DGDRPhaseReady
			}),
			request: betaDGDRForAdmission(func(request *nvidiacomv1beta1.DynamoGraphDeploymentRequest) {
				request.Spec.AutoApply = ptr.To(true)
				request.Status.Phase = nvidiacomv1beta1.DGDRPhaseReady
			}),
			gpuDiscovery: true,
		},
		{
			name:               "auto apply activation requires a missing runtime version override",
			seedWithoutWebhook: true,
			oldRequest: betaDGDRForAdmission(func(request *nvidiacomv1beta1.DynamoGraphDeploymentRequest) {
				request.Spec.AutoApply = ptr.To(false)
				request.Spec.RuntimeVersionOverride = ""
				request.Status.Phase = nvidiacomv1beta1.DGDRPhaseReady
			}),
			request: betaDGDRForAdmission(func(request *nvidiacomv1beta1.DynamoGraphDeploymentRequest) {
				request.Spec.AutoApply = ptr.To(true)
				request.Spec.RuntimeVersionOverride = ""
				request.Status.Phase = nvidiacomv1beta1.DGDRPhaseReady
			}),
			gpuDiscovery: true,
			wantWebhook: []string{
				"spec.runtimeVersionOverride: Required value: is required when spec.image has no parseable semantic-version tag",
			},
		},
		{
			name:               "runtime version override can be added while ready with auto apply disabled",
			seedWithoutWebhook: true,
			oldRequest: betaDGDRForAdmission(func(request *nvidiacomv1beta1.DynamoGraphDeploymentRequest) {
				request.Spec.AutoApply = ptr.To(false)
				request.Spec.RuntimeVersionOverride = ""
				request.Status.Phase = nvidiacomv1beta1.DGDRPhaseReady
			}),
			request: betaDGDRForAdmission(func(request *nvidiacomv1beta1.DynamoGraphDeploymentRequest) {
				request.Spec.AutoApply = ptr.To(false)
				request.Status.Phase = nvidiacomv1beta1.DGDRPhaseReady
			}),
			gpuDiscovery: true,
		},
		{
			name: "runtime version override can change while ready with auto apply disabled",
			oldRequest: betaDGDRForAdmission(func(request *nvidiacomv1beta1.DynamoGraphDeploymentRequest) {
				request.Spec.AutoApply = ptr.To(false)
				request.Status.Phase = nvidiacomv1beta1.DGDRPhaseReady
			}),
			request: betaDGDRForAdmission(func(request *nvidiacomv1beta1.DynamoGraphDeploymentRequest) {
				request.Spec.AutoApply = ptr.To(false)
				request.Spec.RuntimeVersionOverride = "1.2.0"
				request.Status.Phase = nvidiacomv1beta1.DGDRPhaseReady
			}),
			gpuDiscovery: true,
		},
		{
			name: "runtime version override can change while profiling with auto apply disabled",
			oldRequest: betaDGDRForAdmission(func(request *nvidiacomv1beta1.DynamoGraphDeploymentRequest) {
				request.Spec.AutoApply = ptr.To(false)
				request.Status.Phase = nvidiacomv1beta1.DGDRPhaseProfiling
			}),
			request: betaDGDRForAdmission(func(request *nvidiacomv1beta1.DynamoGraphDeploymentRequest) {
				request.Spec.AutoApply = ptr.To(false)
				request.Spec.RuntimeVersionOverride = "1.2.0"
				request.Status.Phase = nvidiacomv1beta1.DGDRPhaseProfiling
			}),
			gpuDiscovery: true,
		},
		{
			name: "other spec updates remain forbidden while ready with auto apply disabled",
			oldRequest: betaDGDRForAdmission(func(request *nvidiacomv1beta1.DynamoGraphDeploymentRequest) {
				request.Spec.AutoApply = ptr.To(false)
				request.Status.Phase = nvidiacomv1beta1.DGDRPhaseReady
			}),
			request: betaDGDRForAdmission(func(request *nvidiacomv1beta1.DynamoGraphDeploymentRequest) {
				request.Spec.AutoApply = ptr.To(false)
				request.Spec.Model = alternateAdmissionModel
				request.Status.Phase = nvidiacomv1beta1.DGDRPhaseReady
			}),
			gpuDiscovery: true,
			wantWebhook: []string{
				`spec: Forbidden: updates are forbidden while the resource is in phase "Ready"; delete and recreate the resource to change its spec`,
			},
		},
		{
			name: "spec update is rejected during deploying",
			oldRequest: betaDGDRForAdmission(func(request *nvidiacomv1beta1.DynamoGraphDeploymentRequest) {
				request.Status.Phase = nvidiacomv1beta1.DGDRPhaseDeploying
			}),
			request: betaDGDRForAdmission(func(request *nvidiacomv1beta1.DynamoGraphDeploymentRequest) {
				request.Spec.Model = alternateAdmissionModel
			}),
			gpuDiscovery: true,
			wantWebhook: []string{
				`spec: Forbidden: updates are forbidden while the resource is in phase "Deploying"; delete and recreate the resource to change its spec`,
			},
		},
		{
			name: "spec update is rejected during deployed",
			oldRequest: betaDGDRForAdmission(func(request *nvidiacomv1beta1.DynamoGraphDeploymentRequest) {
				request.Status.Phase = nvidiacomv1beta1.DGDRPhaseDeployed
			}),
			request: betaDGDRForAdmission(func(request *nvidiacomv1beta1.DynamoGraphDeploymentRequest) {
				request.Spec.Model = alternateAdmissionModel
			}),
			gpuDiscovery: true,
			wantWebhook: []string{
				`spec: Forbidden: updates are forbidden while the resource is in phase "Deployed"; delete and recreate the resource to change its spec`,
			},
		},
		{
			name: "spec update is accepted during failed phase",
			oldRequest: betaDGDRForAdmission(func(request *nvidiacomv1beta1.DynamoGraphDeploymentRequest) {
				request.Status.Phase = nvidiacomv1beta1.DGDRPhaseFailed
			}),
			request: betaDGDRForAdmission(func(request *nvidiacomv1beta1.DynamoGraphDeploymentRequest) {
				request.Spec.Model = alternateAdmissionModel
			}),
			gpuDiscovery: true,
		},
		{
			name:               "unchanged thorough and auto violation is ratcheted on update",
			seedWithoutWebhook: true,
			oldRequest: betaDGDRForAdmission(func(request *nvidiacomv1beta1.DynamoGraphDeploymentRequest) {
				request.Spec.Backend = nvidiacomv1beta1.BackendTypeAuto
				request.Spec.SearchStrategy = nvidiacomv1beta1.SearchStrategyThorough
			}),
			request: betaDGDRForAdmission(func(request *nvidiacomv1beta1.DynamoGraphDeploymentRequest) {
				request.Spec.Backend = nvidiacomv1beta1.BackendTypeAuto
				request.Spec.SearchStrategy = nvidiacomv1beta1.SearchStrategyThorough
				request.Labels = map[string]string{"updated": "true"}
			}),
			gpuDiscovery: true,
		},
		{
			name:               "unchanged legacy custom image without override is ratcheted on update",
			seedWithoutWebhook: true,
			oldRequest: betaDGDRForAdmission(func(request *nvidiacomv1beta1.DynamoGraphDeploymentRequest) {
				request.Spec.RuntimeVersionOverride = ""
				request.Status.Phase = nvidiacomv1beta1.DGDRPhaseDeployed
			}),
			request: betaDGDRForAdmission(func(request *nvidiacomv1beta1.DynamoGraphDeploymentRequest) {
				request.Spec.RuntimeVersionOverride = ""
				request.Status.Phase = nvidiacomv1beta1.DGDRPhaseDeployed
				request.Labels = map[string]string{"updated": "true"}
			}),
			gpuDiscovery: true,
		},
		{
			name:               "adding runtime version override is rejected after deployment",
			seedWithoutWebhook: true,
			oldRequest: betaDGDRForAdmission(func(request *nvidiacomv1beta1.DynamoGraphDeploymentRequest) {
				request.Spec.RuntimeVersionOverride = ""
				request.Status.Phase = nvidiacomv1beta1.DGDRPhaseDeployed
			}),
			request: betaDGDRForAdmission(func(request *nvidiacomv1beta1.DynamoGraphDeploymentRequest) {
				request.Status.Phase = nvidiacomv1beta1.DGDRPhaseDeployed
				request.Labels = map[string]string{"updated": "true"}
			}),
			gpuDiscovery: true,
			wantWebhook: []string{
				`spec: Forbidden: updates are forbidden while the resource is in phase "Deployed"; delete and recreate the resource to change its spec`,
			},
		},
		{
			name: "newly introduced custom image without override is rejected on update",
			oldRequest: betaDGDRForAdmission(func(request *nvidiacomv1beta1.DynamoGraphDeploymentRequest) {
				request.Spec.Image = "test-profiler:1.1.0"
				request.Spec.RuntimeVersionOverride = ""
			}),
			request: betaDGDRForAdmission(func(request *nvidiacomv1beta1.DynamoGraphDeploymentRequest) {
				request.Spec.RuntimeVersionOverride = ""
			}),
			gpuDiscovery: true,
			wantWebhook: []string{
				"spec.runtimeVersionOverride: Required value: is required when spec.image has no parseable semantic-version tag",
			},
		},
		{
			name:               "changing a legacy custom image without override is rejected on update",
			seedWithoutWebhook: true,
			oldRequest: betaDGDRForAdmission(func(request *nvidiacomv1beta1.DynamoGraphDeploymentRequest) {
				request.Spec.RuntimeVersionOverride = ""
			}),
			request: betaDGDRForAdmission(func(request *nvidiacomv1beta1.DynamoGraphDeploymentRequest) {
				request.Spec.Image = "test-profiler:other-custom"
				request.Spec.RuntimeVersionOverride = ""
			}),
			gpuDiscovery: true,
			wantWebhook: []string{
				"spec.runtimeVersionOverride: Required value: is required when spec.image has no parseable semantic-version tag",
			},
		},
		{
			name:       "missing hardware is ratcheted when GPU discovery becomes disabled",
			oldRequest: betaDGDRForAdmission(nil),
			request: betaDGDRForAdmission(func(request *nvidiacomv1beta1.DynamoGraphDeploymentRequest) {
				request.Labels = map[string]string{"updated": "true"}
			}),
		},
		{
			name: "removing manual hardware is rejected while GPU discovery is disabled",
			oldRequest: betaDGDRForAdmission(func(request *nvidiacomv1beta1.DynamoGraphDeploymentRequest) {
				request.Spec.Hardware = &nvidiacomv1beta1.HardwareSpec{GPUSKU: nvidiacomv1beta1.GPUSKUTypeH100SXM}
			}),
			request: betaDGDRForAdmission(nil),
			wantWebhook: []string{
				"spec.hardware: Required value: GPU hardware configuration is required when GPU discovery is disabled",
			},
		},
		{
			name:       "newly introduced search violation is rejected on update",
			oldRequest: betaDGDRForAdmission(nil),
			request: betaDGDRForAdmission(func(request *nvidiacomv1beta1.DynamoGraphDeploymentRequest) {
				request.Spec.Backend = nvidiacomv1beta1.BackendTypeAuto
				request.Spec.SearchStrategy = nvidiacomv1beta1.SearchStrategyThorough
			}),
			gpuDiscovery: true,
			wantWebhook: []string{
				`spec.searchStrategy: Invalid value: "thorough": is incompatible with spec.backend "auto"; set spec.backend to a specific backend (sglang, trtllm, or vllm)`,
			},
		},

		// v1beta2: the CRD owns the unconditional defaults, admission the environment ones.
		{
			name:         "v1beta2 model-only request gets schema and admission defaults",
			request:      v1beta2DGDRForAdmission(nil, nil),
			gpuDiscovery: true,
			discovered:   discoveredH200,
			check: func(t *testing.T, result *unstructured.Unstructured) {
				expectV1beta2SchemaDefaults(t, result)
				expectV1beta2AdmissionDefaults(t, result)
			},
		},
		{
			name: "v1beta2 explicit hardware is kept",
			request: v1beta2DGDRForAdmission(func(spec map[string]any) {
				spec["hardware"] = map[string]any{"gpu": map[string]any{"sku": "h100_sxm", "budget": int64(4)}}
			}, nil),
			gpuDiscovery: true,
			discovered:   discoveredH200,
			check: func(t *testing.T, result *unstructured.Unstructured) {
				if got := v1beta2String(t, result, "spec", "hardware", "gpu", "sku"); got != "h100_sxm" {
					t.Fatalf("hardware.gpu.sku = %q, want h100_sxm", got)
				}
				if got := v1beta2Number(t, result, "spec", "hardware", "gpu", "budget"); got != 4 {
					t.Fatalf("hardware.gpu.budget = %v, want 4", got)
				}
			},
		},
		{
			name:         "v1beta2 failed discovery leaves the hardware unresolved and is admitted",
			request:      v1beta2DGDRForAdmission(nil, nil),
			gpuDiscovery: true,
			discoveryErr: errors.New("no GPUs found"),
			check: func(t *testing.T, result *unstructured.Unstructured) {
				v1beta2Absent(t, result, "spec", "hardware")
			},
		},
		{
			name: "v1beta2 parameters are stored untouched and never decide hardware defaulting",
			request: v1beta2DGDRForAdmission(func(spec map[string]any) {
				setV1beta2SearchSpace(spec, map[string]any{"hardware_sku": "h100_sxm", "gpu_budget": int64(4)})
			}, nil),
			gpuDiscovery: true,
			discoveryErr: errors.New("no GPUs found"),
			check: func(t *testing.T, result *unstructured.Unstructured) {
				v1beta2Absent(t, result, "spec", "hardware")
				if got := v1beta2String(t, result, "spec", "aisimulate", "parameters", "search_space", "hardware_sku"); got != "h100_sxm" {
					t.Fatalf("parameters.search_space.hardware_sku = %q, want h100_sxm", got)
				}
			},
		},
		{
			name: "v1beta2 parameters null does not suppress hardware discovery",
			request: v1beta2DGDRForAdmission(func(spec map[string]any) {
				setV1beta2SearchSpace(spec, map[string]any{"hardware_sku": nil})
			}, nil),
			gpuDiscovery: true,
			discovered:   discoveredH200,
			check: func(t *testing.T, result *unstructured.Unstructured) {
				if got := v1beta2String(t, result, "spec", "hardware", "gpu", "sku"); got == "" {
					t.Fatal("hardware.gpu.sku is empty: discovery must read the typed spec only")
				}
			},
		},
		{
			name: "v1beta2 metadata-only update does not rediscover hardware after discovery recovers",
			oldRequest: v1beta2DGDRForAdmission(func(spec map[string]any) {
				setV1beta2SearchSpace(spec, map[string]any{"hardware_sku": "h100_sxm", "gpu_budget": int64(8)})
			}, nil),
			request: v1beta2DGDRForAdmission(func(spec map[string]any) {
				setV1beta2SearchSpace(spec, map[string]any{"hardware_sku": "h100_sxm", "gpu_budget": int64(8)})
			}, map[string]any{"updated": "true"}),
			gpuDiscovery: true,
			discoveryErr: errors.New("no GPUs found"),
			beforeUpdate: func(*testing.T) { admissionHardware.set(discoveredH200, nil) },
			check: func(t *testing.T, result *unstructured.Unstructured) {
				v1beta2Absent(t, result, "spec", "hardware")
				if got := result.GetGeneration(); got != 1 {
					t.Fatalf("generation = %d, want 1: the spec must not change", got)
				}
			},
		},
		{
			name: "v1beta2 removing only the budget rediscovers only the budget",
			oldRequest: v1beta2DGDRForAdmission(func(spec map[string]any) {
				spec["hardware"] = map[string]any{"gpu": map[string]any{"sku": "h100_sxm", "budget": int64(4)}}
			}, nil),
			request: v1beta2DGDRForAdmission(func(spec map[string]any) {
				spec["hardware"] = map[string]any{"gpu": map[string]any{"sku": "h100_sxm"}}
			}, nil),
			gpuDiscovery: true,
			discovered:   discoveredH200,
			check: func(t *testing.T, result *unstructured.Unstructured) {
				expectV1beta2GPU(t, result, "h100_sxm", 8)
			},
		},
		{
			name: "v1beta2 removing only the SKU rediscovers only the SKU",
			oldRequest: v1beta2DGDRForAdmission(func(spec map[string]any) {
				spec["hardware"] = map[string]any{"gpu": map[string]any{"sku": "h100_sxm", "budget": int64(4)}}
			}, nil),
			request: v1beta2DGDRForAdmission(func(spec map[string]any) {
				spec["hardware"] = map[string]any{"gpu": map[string]any{"budget": int64(4)}}
			}, nil),
			gpuDiscovery: true,
			discovered:   discoveredH200,
			check: func(t *testing.T, result *unstructured.Unstructured) {
				expectV1beta2GPU(t, result, "h200_sxm", 4)
			},
		},
		{
			name: "v1beta2 removing all typed hardware rediscovers the SKU and the budget",
			oldRequest: v1beta2DGDRForAdmission(func(spec map[string]any) {
				spec["hardware"] = map[string]any{"gpu": map[string]any{"sku": "h100_sxm", "budget": int64(4)}}
			}, nil),
			request:      v1beta2DGDRForAdmission(nil, nil),
			gpuDiscovery: true,
			discovered:   discoveredH200,
			check: func(t *testing.T, result *unstructured.Unstructured) {
				expectV1beta2GPU(t, result, "h200_sxm", 8)
			},
		},
		{
			name: "v1beta2 empty workload receives the static model-only default",
			request: v1beta2DGDRForAdmission(func(spec map[string]any) {
				spec["workload"] = map[string]any{}
			}, nil),
			gpuDiscovery: true,
			discovered:   discoveredH200,
			check: func(t *testing.T, result *unstructured.Unstructured) {
				for key, want := range map[string]float64{"isl": 1024, "osl": 128, "concurrency": 10} {
					if got := v1beta2Number(t, result, "spec", "workload", "static", key); got != want {
						t.Fatalf("workload.static.%s = %v, want %v", key, got, want)
					}
				}
				v1beta2Absent(t, result, "spec", "workload", "static", "kvLoadRatio")
			},
		},
		{
			name: "v1beta2 empty workload static is an opt-out and stays empty",
			request: v1beta2DGDRForAdmission(func(spec map[string]any) {
				spec["workload"] = map[string]any{"static": map[string]any{}}
			}, nil),
			gpuDiscovery: true,
			discovered:   discoveredH200,
			check: func(t *testing.T, result *unstructured.Unstructured) {
				v1beta2Empty(t, result, "spec", "workload", "static")
			},
		},
		{
			name: "v1beta2 empty workload static stays empty beside native parameters",
			request: v1beta2DGDRForAdmission(func(spec map[string]any) {
				spec["workload"] = map[string]any{"static": map[string]any{}}
				spec["aisimulate"] = map[string]any{"parameters": mustJSONObject(t, `{"workload":{"isl":512,"osl":128,"concurrency":8,"request_count":100}}`)}
			}, nil),
			gpuDiscovery: true,
			discovered:   discoveredH200,
			check: func(t *testing.T, result *unstructured.Unstructured) {
				v1beta2Empty(t, result, "spec", "workload", "static")
				v1beta2Absent(t, result, "spec", "objective", "sla")
				expectV1beta2AdmissionDefaults(t, result)
			},
		},
		{
			name: "v1beta2 empty sla is an opt-out and is not refilled",
			request: v1beta2DGDRForAdmission(func(spec map[string]any) {
				spec["objective"] = map[string]any{"sla": map[string]any{}}
			}, nil),
			gpuDiscovery: true,
			discovered:   discoveredH200,
			check: func(t *testing.T, result *unstructured.Unstructured) {
				v1beta2Empty(t, result, "spec", "objective", "sla")
				if got := v1beta2Number(t, result, "spec", "workload", "static", "isl"); got != 1024 {
					t.Fatalf("workload.static.isl = %v, want the schema default 1024", got)
				}
			},
		},
		{
			name: "v1beta2 native parameters win without rewriting the typed spec",
			request: v1beta2DGDRForAdmission(func(spec map[string]any) {
				spec["aisimulate"] = map[string]any{"parameters": mustJSONObject(t, `{"workload":{"isl":256}}`)}
			}, nil),
			gpuDiscovery: true,
			discovered:   discoveredH200,
			check: func(t *testing.T, result *unstructured.Unstructured) {
				if got := v1beta2Number(t, result, "spec", "workload", "static", "isl"); got != 1024 {
					t.Fatalf("typed workload.static.isl = %v, want 1024", got)
				}
				if got := v1beta2Number(t, result, "spec", "aisimulate", "parameters", "workload", "isl"); got != 256 {
					t.Fatalf("parameters.workload.isl = %v, want 256", got)
				}
			},
		},
		{
			name: "v1beta2 stored invalid spec is rejected again on a metadata-only update",
			oldRequest: v1beta2DGDRForAdmission(func(spec map[string]any) {
				spec["backends"] = []any{"vllm", "vllm"}
			}, nil),
			request: v1beta2DGDRForAdmission(func(spec map[string]any) {
				spec["backends"] = []any{"vllm", "vllm"}
			}, map[string]any{"updated": "true"}),
			seedWithoutWebhook: true,
			gpuDiscovery:       true,
			discovered:         discoveredH200,
			wantWebhook:        []string{`spec.backends[1]: Duplicate value: "vllm"`},
		},
		{
			name: "v1beta2 duplicate backends are rejected on create",
			request: v1beta2DGDRForAdmission(func(spec map[string]any) {
				spec["backends"] = []any{"vllm", "vllm"}
			}, nil),
			gpuDiscovery: true,
			discovered:   discoveredH200,
			wantWebhook:  []string{`spec.backends[1]: Duplicate value: "vllm"`},
		},
		{
			name: "v1beta2 versionOverride latest is rejected by the schema",
			request: v1beta2DGDRForAdmission(func(spec map[string]any) {
				spec["aisimulate"] = map[string]any{"versionOverride": "latest"}
			}, nil),
			gpuDiscovery:  true,
			discovered:    discoveredH200,
			wantSchemaErr: `spec.aisimulate.versionOverride: Invalid value: "latest": spec.aisimulate.versionOverride in body should match '^(0|[1-9][0-9]{0,3})\.(0|[1-9][0-9]{0,3})\.(0|[1-9][0-9]{0,3})$'`,
		},
		{
			name: "v1beta2 versionOverride with a v prefix is rejected by the schema",
			request: v1beta2DGDRForAdmission(func(spec map[string]any) {
				spec["aisimulate"] = map[string]any{"versionOverride": "v0.13.0"}
			}, nil),
			gpuDiscovery:  true,
			discovered:    discoveredH200,
			wantSchemaErr: `spec.aisimulate.versionOverride: Invalid value: "v0.13.0": spec.aisimulate.versionOverride in body should match '^(0|[1-9][0-9]{0,3})\.(0|[1-9][0-9]{0,3})\.(0|[1-9][0-9]{0,3})$'`,
		},
		{
			name: "v1beta2 versionOverride 0.13.0 is accepted",
			request: v1beta2DGDRForAdmission(func(spec map[string]any) {
				spec["aisimulate"] = map[string]any{"versionOverride": "0.13.0"}
			}, nil),
			gpuDiscovery: true,
			discovered:   discoveredH200,
			check: func(t *testing.T, result *unstructured.Unstructured) {
				if got := v1beta2String(t, result, "spec", "aisimulate", "versionOverride"); got != "0.13.0" {
					t.Fatalf("versionOverride = %q, want 0.13.0", got)
				}
			},
		},

		// v1beta2 UPDATE: defaults stay idempotent, hardware is never rediscovered, and any
		// spec change is fully validated.
		{
			name:         "v1beta2 update without a spec change keeps the spec and generation",
			oldRequest:   v1beta2DGDRForAdmission(nil, nil),
			request:      v1beta2DGDRForAdmission(nil, map[string]any{"updated": "true"}),
			gpuDiscovery: true,
			discovered:   discoveredH200,
			check: func(t *testing.T, result *unstructured.Unstructured) {
				if result.GetGeneration() != 1 {
					t.Fatalf("generation = %d, want 1: defaulting must not change an unchanged spec", result.GetGeneration())
				}
				expectV1beta2SchemaDefaults(t, result)
				expectV1beta2AdmissionDefaults(t, result)
			},
		},
		{
			name:               "v1beta2 metadata-only update does not pick up hardware once discovery recovers",
			oldRequest:         v1beta2DGDRForAdmission(nil, nil),
			seedWithoutWebhook: true, // stored before hardware was validated
			request:            v1beta2DGDRForAdmission(nil, map[string]any{"updated": "true"}),
			gpuDiscovery:       true,
			discoveryErr:       errors.New("no GPUs found"),
			beforeUpdate:       func(*testing.T) { admissionHardware.set(discoveredH200, nil) },
			check: func(t *testing.T, result *unstructured.Unstructured) {
				if result.GetGeneration() != 1 {
					t.Fatalf("generation = %d, want 1: an update must not rewrite the spec", result.GetGeneration())
				}
				v1beta2Absent(t, result, "spec", "hardware")
			},
		},
		{
			name:         "v1beta2 native-only hardware survives a metadata-only update after discovery recovers",
			oldRequest:   v1beta2NativeHardwareDGDR(nil),
			request:      v1beta2NativeHardwareDGDR(map[string]any{"updated": "true"}),
			gpuDiscovery: true,
			discoveryErr: errors.New("no GPUs found"),
			beforeUpdate: func(*testing.T) { admissionHardware.set(discoveredH200, nil) },
			check: func(t *testing.T, result *unstructured.Unstructured) {
				if result.GetGeneration() != 1 {
					t.Fatalf("generation = %d, want 1", result.GetGeneration())
				}
				v1beta2Absent(t, result, "spec", "hardware")
				if got := v1beta2String(t, result, "spec", "aisimulate", "parameters", "search_space", "hardware_sku"); got != "h100_sxm" {
					t.Fatalf("parameters.search_space.hardware_sku = %q, want h100_sxm", got)
				}
			},
		},
		{
			name:       "v1beta2 spec change is accepted and bumps the generation",
			oldRequest: v1beta2DGDRForAdmission(nil, nil),
			request: v1beta2DGDRForAdmission(func(spec map[string]any) {
				spec["model"] = map[string]any{"name": alternateAdmissionModel}
			}, nil),
			gpuDiscovery: true,
			discovered:   discoveredH200,
			check: func(t *testing.T, result *unstructured.Unstructured) {
				if result.GetGeneration() != 2 {
					t.Fatalf("generation = %d, want 2", result.GetGeneration())
				}
				if got := v1beta2String(t, result, "spec", "model", "name"); got != alternateAdmissionModel {
					t.Fatalf("model.name = %q, want %q", got, alternateAdmissionModel)
				}
			},
		},
		{
			name:       "v1beta2 update to duplicate backends is rejected",
			oldRequest: v1beta2DGDRForAdmission(nil, nil),
			request: v1beta2DGDRForAdmission(func(spec map[string]any) {
				spec["backends"] = []any{"sglang", "sglang"}
			}, nil),
			gpuDiscovery: true,
			discovered:   discoveredH200,
			wantWebhook:  []string{`spec.backends[1]: Duplicate value: "sglang"`},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			admissionHardware.set(tt.discovered, tt.discoveryErr)
			t.Cleanup(func() { admissionHardware.set(webhookdefaulting.DiscoveredHardware{}, nil) })

			gates := features.Gates{GPUDiscovery: tt.gpuDiscovery}
			test := admissionTestCase{
				object:             tt.request,
				oldObject:          tt.oldRequest,
				gates:              gates,
				seedWithoutWebhook: tt.seedWithoutWebhook,
				wantSchemaError:    tt.wantSchemaErr,
				wantCELError:       tt.wantCELErr,
				wantStatusError:    tt.wantStatusErr,
				wantWebhookErrors:  tt.wantWebhook,
				wantWarnings:       tt.wantWarnings,
				beforeUpdate:       tt.beforeUpdate,
			}

			if tt.oldRequest != nil && !tt.gpuDiscovery {
				seedGates := gates
				seedGates.GPUDiscovery = true
				test.seedGates = &seedGates
			}
			actual := runAdmissionTest(t, test)
			if tt.check != nil {
				tt.check(t, actual)
			}
			if tt.wantImage != "" {
				image, found, err := unstructured.NestedString(actual.Object, "spec", "image")
				if err != nil || !found {
					t.Fatalf("read defaulted spec.image: found=%v, err=%v", found, err)
				}
				if image != tt.wantImage {
					t.Fatalf("defaulted spec.image = %q, want %q", image, tt.wantImage)
				}
			}
		})
	}
}

func TestDynamoGraphDeploymentRequestStatusRepresentationMatchesSpec(t *testing.T) {
	tests := []struct {
		name         string
		createObject runtime.Object
		statusObject runtime.Object
		status       map[string]any
	}{
		{
			name:         "v1alpha1 native spec rejects v1beta2 status",
			createObject: alphaDGDRForAdmission(nil),
			statusObject: alphaDGDRForAdmission(nil),
			status:       map[string]any{"v1beta2": map[string]any{"observedGeneration": int64(1)}},
		},
		{
			name:         "v1alpha1 v1beta2 spec rejects native status",
			createObject: beta2DGDRForAdmission(setBeta2AdmissionHardware),
			statusObject: alphaDGDRForAdmission(nil),
			status:       map[string]any{"state": string(nvidiacomv1alpha1.DGDRStatePending)},
		},
		{
			name:         "v1beta1 native spec rejects v1beta2 status",
			createObject: betaDGDRForAdmission(nil),
			statusObject: betaDGDRForAdmission(nil),
			status:       map[string]any{"v1beta2": map[string]any{"observedGeneration": int64(1)}},
		},
		{
			name:         "v1beta1 v1beta2 spec rejects native status",
			createObject: beta2DGDRForAdmission(setBeta2AdmissionHardware),
			statusObject: betaDGDRForAdmission(nil),
			status:       map[string]any{"phase": string(nvidiacomv1beta1.DGDRPhasePending)},
		},
		{
			name:         "v1beta2 native spec rejects v1beta1 status",
			createObject: beta2DGDRForAdmission(setBeta2AdmissionHardware),
			statusObject: beta2DGDRForAdmission(nil),
			status:       map[string]any{"v1beta1": map[string]any{"phase": string(nvidiacomv1beta1.DGDRPhasePending)}},
		},
		{
			name:         "v1beta2 v1beta1 spec rejects native status",
			createObject: betaDGDRForAdmission(nil),
			statusObject: beta2DGDRForAdmission(nil),
			status:       map[string]any{"observedGeneration": int64(1)},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Create a request through the production API and conversion path")
			env := admissionEnv.ForTest(t)
			admissionGate.set(features.Gates{GPUDiscovery: true})
			warnings := &warningRecorder{}
			createClient := newAdmissionResourceClient(t, env, test.createObject, "", warnings)
			object, _ := admissionObject(t, test.createObject, env.Namespace(), nil)
			created, err := createClient.Create(t.Context(), object, metav1.CreateOptions{})
			if err != nil {
				t.Fatalf("create request: %v", err)
			}

			t.Log("Read the request through the API version used for the status update")
			statusClient := newAdmissionResourceClient(t, env, test.statusObject, "", warnings)
			converted, err := statusClient.Get(t.Context(), created.GetName(), metav1.GetOptions{})
			if err != nil {
				t.Fatalf("get request for status update: %v", err)
			}

			t.Log("Replace status with the representation opposite to spec")
			if err := unstructured.SetNestedMap(converted.Object, test.status, "status"); err != nil {
				t.Fatalf("set mismatched status: %v", err)
			}
			_, err = statusClient.UpdateStatus(t.Context(), converted, metav1.UpdateOptions{})
			if err == nil || !strings.Contains(err.Error(), "nonempty status must use the representation selected by spec") {
				t.Fatalf("status update error = %v, want representation mismatch", err)
			}
		})
	}
}

func TestDynamoGraphDeploymentRequestV1Beta2EmptyStatusSurvivesV1Alpha1(t *testing.T) {
	t.Log("Create a native v1beta2 request with empty status")
	env := admissionEnv.ForTest(t)
	admissionGate.set(features.Gates{GPUDiscovery: true})
	warnings := &warningRecorder{}
	beta2Fixture := beta2DGDRForAdmission(func(request *nvidiacomv1beta2.DynamoGraphDeploymentRequest) {
		request.Name = "empty-status-alpha-bridge"
		setBeta2AdmissionHardware(request)
	})
	beta2Client := newAdmissionResourceClient(t, env, beta2Fixture, "", warnings)
	object, _ := admissionObject(t, beta2Fixture, env.Namespace(), nil)
	created, err := beta2Client.Create(t.Context(), object, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("create v1beta2 request: %v", err)
	}

	t.Log("Read and write the empty status through v1alpha1")
	alphaClient := newAdmissionResourceClient(t, env, alphaDGDRForAdmission(nil), "", warnings)
	alpha, err := alphaClient.Get(t.Context(), created.GetName(), metav1.GetOptions{})
	if err != nil {
		t.Fatalf("get request as v1alpha1: %v", err)
	}
	assertEmptyDGDRStatus(t, alpha)
	alpha, err = alphaClient.UpdateStatus(t.Context(), alpha, metav1.UpdateOptions{})
	if err != nil {
		t.Fatalf("write empty v1alpha1 status: %v", err)
	}

	t.Log("Read the request as v1beta2 and prove native status can still be written")
	converted, err := beta2Client.Get(t.Context(), alpha.GetName(), metav1.GetOptions{})
	if err != nil {
		t.Fatalf("get request as v1beta2: %v", err)
	}
	assertEmptyDGDRStatus(t, converted)
	if err := unstructured.SetNestedField(converted.Object, int64(1), "status", "observedGeneration"); err != nil {
		t.Fatalf("set native v1beta2 status: %v", err)
	}
	if _, err := beta2Client.UpdateStatus(t.Context(), converted, metav1.UpdateOptions{}); err != nil {
		t.Fatalf("write native v1beta2 status after v1alpha1 round-trip: %v", err)
	}
}

func assertEmptyDGDRStatus(t *testing.T, object *unstructured.Unstructured) {
	t.Helper()
	status, found, err := unstructured.NestedMap(object.Object, "status")
	if err != nil {
		t.Fatalf("read status: %v", err)
	}
	if found && len(status) != 0 {
		t.Fatalf("status = %v, want empty", status)
	}
}

func betaDGDRForAdmission(
	mutate func(*nvidiacomv1beta1.DynamoGraphDeploymentRequest),
) *nvidiacomv1beta1.DynamoGraphDeploymentRequest {
	request := &nvidiacomv1beta1.DynamoGraphDeploymentRequest{
		TypeMeta: metav1.TypeMeta{
			APIVersion: nvidiacomv1beta1.GroupVersion.String(),
			Kind:       "DynamoGraphDeploymentRequest",
		},
		ObjectMeta: metav1.ObjectMeta{Name: "test-dgdr", Namespace: "default"},
		Spec: nvidiacomv1beta1.DynamoGraphDeploymentRequestSpec{
			Model:                  "Qwen/Qwen3-0.6B",
			Backend:                nvidiacomv1beta1.BackendTypeVllm,
			Image:                  "profiler:latest",
			RuntimeVersionOverride: "1.1.0",
			SearchStrategy:         nvidiacomv1beta1.SearchStrategyRapid,
		},
	}
	if mutate != nil {
		mutate(request)
	}
	return request
}

func alphaDGDRForAdmission(
	mutate func(*nvidiacomv1alpha1.DynamoGraphDeploymentRequest),
) *nvidiacomv1alpha1.DynamoGraphDeploymentRequest {
	request := &nvidiacomv1alpha1.DynamoGraphDeploymentRequest{
		TypeMeta: metav1.TypeMeta{
			APIVersion: nvidiacomv1alpha1.GroupVersion.String(),
			Kind:       "DynamoGraphDeploymentRequest",
		},
		ObjectMeta: metav1.ObjectMeta{Name: "test-dgdr", Namespace: "default"},
		Spec: nvidiacomv1alpha1.DynamoGraphDeploymentRequestSpec{
			Model:                  "Qwen/Qwen3-0.6B",
			Backend:                "vllm",
			RuntimeVersionOverride: "1.1.0",
			ProfilingConfig: &nvidiacomv1alpha1.ProfilingConfigSpec{
				ProfilerImage: "profiler:latest",
			},
		},
	}
	if mutate != nil {
		mutate(request)
	}
	return request
}

func beta2DGDRForAdmission(
	mutate func(*nvidiacomv1beta2.DynamoGraphDeploymentRequest),
) *nvidiacomv1beta2.DynamoGraphDeploymentRequest {
	request := &nvidiacomv1beta2.DynamoGraphDeploymentRequest{
		TypeMeta: metav1.TypeMeta{
			APIVersion: nvidiacomv1beta2.GroupVersion.String(),
			Kind:       "DynamoGraphDeploymentRequest",
		},
		ObjectMeta: metav1.ObjectMeta{Name: "test-dgdr", Namespace: "default"},
		Spec: nvidiacomv1beta2.DynamoGraphDeploymentRequestSpec{
			NativeSearchInput: nvidiacomv1beta2.NativeSearchInput{
				Model: &nvidiacomv1beta2.ModelSpec{Name: "Qwen/Qwen3-0.6B"},
			},
		},
	}
	if mutate != nil {
		mutate(request)
	}
	return request
}

const v1beta2AdmissionModel = "Qwen/Qwen3-0.6B"

// v1beta2NativeHardwareDGDR supplies hardware only through native AISimulate parameters.
func v1beta2NativeHardwareDGDR(labels map[string]any) *unstructured.Unstructured {
	return v1beta2DGDRForAdmission(func(spec map[string]any) {
		spec["aisimulate"] = map[string]any{
			"image": admissionAISimulateImage,
			"parameters": map[string]any{
				"search_space": map[string]any{"hardware_sku": "h100_sxm", "gpu_budget": int64(8)},
			},
		}
	}, labels)
}

// v1beta2DGDRForAdmission builds a v1beta2 Request as a client sends it, before the CRD
// defaults. It is unstructured on purpose: an omitted field,
// an explicit empty object and a null are different requests, and typed Go cannot tell them
// apart.
func v1beta2DGDRForAdmission(mutate func(spec map[string]any), labels map[string]any) *unstructured.Unstructured {
	spec := map[string]any{"model": map[string]any{"name": v1beta2AdmissionModel}}
	if mutate != nil {
		mutate(spec)
	}
	metadata := map[string]any{"name": "test-v1beta2-dgdr", "namespace": "default"}
	if labels != nil {
		metadata["labels"] = labels
	}
	return &unstructured.Unstructured{Object: map[string]any{
		"apiVersion": "nvidia.com/v1beta2",
		"kind":       "DynamoGraphDeploymentRequest",
		"metadata":   metadata,
		"spec":       spec,
	}}
}

// mustJSONObject decodes a native AISimulate parameters document.
func mustJSONObject(t *testing.T, document string) map[string]any {
	t.Helper()
	var value map[string]any
	if err := json.Unmarshal([]byte(document), &value); err != nil {
		t.Fatalf("decode %q: %v", document, err)
	}
	return value
}

func v1beta2Field(t *testing.T, obj *unstructured.Unstructured, path ...string) (any, bool) {
	t.Helper()
	value, found, err := unstructured.NestedFieldNoCopy(obj.Object, path...)
	if err != nil {
		t.Fatalf("read %v: %v", path, err)
	}
	return value, found
}

func v1beta2Number(t *testing.T, obj *unstructured.Unstructured, path ...string) float64 {
	t.Helper()
	value, found := v1beta2Field(t, obj, path...)
	if !found {
		t.Fatalf("%v is not set", path)
	}
	switch n := value.(type) {
	case int64:
		return float64(n)
	case float64:
		return n
	default:
		t.Fatalf("%v = %#v, want a number", path, value)
		return 0
	}
}

func v1beta2String(t *testing.T, obj *unstructured.Unstructured, path ...string) string {
	t.Helper()
	value, found := v1beta2Field(t, obj, path...)
	if !found {
		t.Fatalf("%v is not set", path)
	}
	text, ok := value.(string)
	if !ok {
		t.Fatalf("%v = %#v, want a string", path, value)
	}
	return text
}

func v1beta2Absent(t *testing.T, obj *unstructured.Unstructured, path ...string) {
	t.Helper()
	if value, found := v1beta2Field(t, obj, path...); found {
		t.Fatalf("%v = %#v, want it absent", path, value)
	}
}

func v1beta2Empty(t *testing.T, obj *unstructured.Unstructured, path ...string) {
	t.Helper()
	value, found := v1beta2Field(t, obj, path...)
	if !found || !reflect.DeepEqual(value, map[string]any{}) {
		t.Fatalf("%v = %#v (found=%v), want an empty object", path, value, found)
	}
}

// expectV1beta2SchemaDefaults checks what the CRD supplies when workload and sla are omitted.
func expectV1beta2SchemaDefaults(t *testing.T, obj *unstructured.Unstructured) {
	t.Helper()
	for key, want := range map[string]float64{"isl": 1024, "osl": 128, "concurrency": 10} {
		if got := v1beta2Number(t, obj, "spec", "workload", "static", key); got != want {
			t.Fatalf("workload.static.%s = %v, want %v", key, got, want)
		}
	}
	v1beta2Absent(t, obj, "spec", "workload", "static", "kvLoadRatio")
	v1beta2Absent(t, obj, "spec", "objective", "sla")
	if got := v1beta2String(t, obj, "spec", "objective", "mode"); got != "Optimize" {
		t.Fatalf("objective.mode = %q, want Optimize", got)
	}
}

// expectV1beta2AdmissionDefaults checks what admission supplies from the environment.
func expectV1beta2AdmissionDefaults(t *testing.T, obj *unstructured.Unstructured) {
	t.Helper()
	if got := v1beta2String(t, obj, "spec", "objective", "metric"); got != "Throughput" {
		t.Fatalf("objective.metric = %q, want Throughput", got)
	}
	if got := v1beta2String(t, obj, "spec", "aisimulate", "image"); got != admissionAISimulateImage {
		t.Fatalf("aisimulate.image = %q, want %q", got, admissionAISimulateImage)
	}
	if got := v1beta2String(t, obj, "spec", "hardware", "gpu", "sku"); got != "h200_sxm" {
		t.Fatalf("hardware.gpu.sku = %q, want h200_sxm", got)
	}
	if got := v1beta2Number(t, obj, "spec", "hardware", "gpu", "budget"); got != 8 {
		t.Fatalf("hardware.gpu.budget = %v, want 8", got)
	}
}

// setV1beta2SearchSpace sets spec.aisimulate.parameters.search_space to searchSpace, replacing
// any parameters the fixture carried.
func setV1beta2SearchSpace(spec map[string]any, searchSpace map[string]any) {
	aisimulate, _ := spec["aisimulate"].(map[string]any)
	if aisimulate == nil {
		aisimulate = map[string]any{}
		spec["aisimulate"] = aisimulate
	}
	aisimulate["parameters"] = map[string]any{"search_space": searchSpace}
}

// setBeta2AdmissionHardware names the hardware, so fixtures do not depend on GPU discovery.
func setBeta2AdmissionHardware(request *nvidiacomv1beta2.DynamoGraphDeploymentRequest) {
	sku, budget := "h100_sxm", int32(4)
	request.Spec.Hardware = &nvidiacomv1beta2.HardwareSpec{
		GPU: &nvidiacomv1beta2.GPUHardwareSpec{SKU: &sku, Budget: &budget},
	}
}

// expectV1beta2GPU asserts the typed GPU hardware the API server stored.
func expectV1beta2GPU(t *testing.T, result *unstructured.Unstructured, sku string, budget float64) {
	t.Helper()
	if got := v1beta2String(t, result, "spec", "hardware", "gpu", "sku"); got != sku {
		t.Fatalf("hardware.gpu.sku = %q, want %q", got, sku)
	}
	if got := v1beta2Number(t, result, "spec", "hardware", "gpu", "budget"); got != budget {
		t.Fatalf("hardware.gpu.budget = %v, want %v", got, budget)
	}
}
