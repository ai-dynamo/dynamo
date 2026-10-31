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
	"errors"
	"strings"
	"testing"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	nvidiacomv1beta2 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta2"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/gpu"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
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

	discoveredH200 := gpu.GPUInfo{System: "h200_sxm", GPUsPerNode: 8, NodesWithGPUs: 1, TotalGPUs: 8, SelectedShapeGPUs: 8}
	discoveredH100 := gpu.GPUInfo{System: "h100_sxm", GPUsPerNode: 8, NodesWithGPUs: 1, TotalGPUs: 8, SelectedShapeGPUs: 8}

	tests := []struct {
		name               string
		request            runtime.Object
		oldRequest         runtime.Object
		gpuDiscovery       bool
		seedWithoutWebhook bool
		// discovered and discoveryErr are what cluster GPU discovery reports while the case
		// runs; beforeUpdate may change them between the seed and the update.
		discovered   gpu.GPUInfo
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
			name:       "v1beta2 create bypasses legacy v1beta1 validation",
			request:    beta2DGDRForAdmission(setBeta2AdmissionHardware),
			discovered: discoveredH100,
		},
		{
			name:       "v1beta2 update bypasses legacy v1beta1 validation",
			oldRequest: beta2DGDRForAdmission(setBeta2AdmissionHardware),
			request: beta2DGDRForAdmission(func(request *nvidiacomv1beta2.DynamoGraphDeploymentRequest) {
				setBeta2AdmissionHardware(request)
				request.Labels = map[string]string{"updated": "true"}
			}),
			discovered: discoveredH100,
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
			name: "RTX PRO 6000 hardware is admitted",
			request: betaDGDRForAdmission(func(request *nvidiacomv1beta1.DynamoGraphDeploymentRequest) {
				request.Spec.Hardware = &nvidiacomv1beta1.HardwareSpec{GPUSKU: nvidiacomv1beta1.GPUSKUTypeRTXPRO6000Server}
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
				request.Spec.AutoApply = new(false)
				request.Status.Phase = nvidiacomv1beta1.DGDRPhaseReady
			}),
			request: betaDGDRForAdmission(func(request *nvidiacomv1beta1.DynamoGraphDeploymentRequest) {
				request.Spec.AutoApply = new(true)
				request.Status.Phase = nvidiacomv1beta1.DGDRPhaseReady
			}),
			gpuDiscovery: true,
		},
		{
			name:               "auto apply activation can add a missing runtime version override",
			seedWithoutWebhook: true,
			oldRequest: betaDGDRForAdmission(func(request *nvidiacomv1beta1.DynamoGraphDeploymentRequest) {
				request.Spec.AutoApply = new(false)
				request.Spec.RuntimeVersionOverride = ""
				request.Status.Phase = nvidiacomv1beta1.DGDRPhaseReady
			}),
			request: betaDGDRForAdmission(func(request *nvidiacomv1beta1.DynamoGraphDeploymentRequest) {
				request.Spec.AutoApply = new(true)
				request.Status.Phase = nvidiacomv1beta1.DGDRPhaseReady
			}),
			gpuDiscovery: true,
		},
		{
			name:               "auto apply activation requires a missing runtime version override",
			seedWithoutWebhook: true,
			oldRequest: betaDGDRForAdmission(func(request *nvidiacomv1beta1.DynamoGraphDeploymentRequest) {
				request.Spec.AutoApply = new(false)
				request.Spec.RuntimeVersionOverride = ""
				request.Status.Phase = nvidiacomv1beta1.DGDRPhaseReady
			}),
			request: betaDGDRForAdmission(func(request *nvidiacomv1beta1.DynamoGraphDeploymentRequest) {
				request.Spec.AutoApply = new(true)
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
				request.Spec.AutoApply = new(false)
				request.Spec.RuntimeVersionOverride = ""
				request.Status.Phase = nvidiacomv1beta1.DGDRPhaseReady
			}),
			request: betaDGDRForAdmission(func(request *nvidiacomv1beta1.DynamoGraphDeploymentRequest) {
				request.Spec.AutoApply = new(false)
				request.Status.Phase = nvidiacomv1beta1.DGDRPhaseReady
			}),
			gpuDiscovery: true,
		},
		{
			name: "runtime version override can change while ready with auto apply disabled",
			oldRequest: betaDGDRForAdmission(func(request *nvidiacomv1beta1.DynamoGraphDeploymentRequest) {
				request.Spec.AutoApply = new(false)
				request.Status.Phase = nvidiacomv1beta1.DGDRPhaseReady
			}),
			request: betaDGDRForAdmission(func(request *nvidiacomv1beta1.DynamoGraphDeploymentRequest) {
				request.Spec.AutoApply = new(false)
				request.Spec.RuntimeVersionOverride = "1.2.0"
				request.Status.Phase = nvidiacomv1beta1.DGDRPhaseReady
			}),
			gpuDiscovery: true,
		},
		{
			name: "runtime version override can change while profiling with auto apply disabled",
			oldRequest: betaDGDRForAdmission(func(request *nvidiacomv1beta1.DynamoGraphDeploymentRequest) {
				request.Spec.AutoApply = new(false)
				request.Status.Phase = nvidiacomv1beta1.DGDRPhaseProfiling
			}),
			request: betaDGDRForAdmission(func(request *nvidiacomv1beta1.DynamoGraphDeploymentRequest) {
				request.Spec.AutoApply = new(false)
				request.Spec.RuntimeVersionOverride = "1.2.0"
				request.Status.Phase = nvidiacomv1beta1.DGDRPhaseProfiling
			}),
			gpuDiscovery: true,
		},
		{
			name: "other spec updates remain forbidden while ready with auto apply disabled",
			oldRequest: betaDGDRForAdmission(func(request *nvidiacomv1beta1.DynamoGraphDeploymentRequest) {
				request.Spec.AutoApply = new(false)
				request.Status.Phase = nvidiacomv1beta1.DGDRPhaseReady
			}),
			request: betaDGDRForAdmission(func(request *nvidiacomv1beta1.DynamoGraphDeploymentRequest) {
				request.Spec.AutoApply = new(false)
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

		{
			name: "v1beta2 unknown hardware without capacity or topology recommends the manual override",
			request: v1beta2DGDRForAdmission(func(spec map[string]any) {
				spec["hardware"] = map[string]any{"gpu": map[string]any{
					"sku": "future_accelerator",
				}}
			}),
			gpuDiscovery: true,
			wantWebhook: []string{
				`spec.hardware.gpu.budget: Required value: cannot discover GPU capacity or topology for SKU "future_accelerator" because it is not known to the bundled AISimulate catalog; set spec.hardware.gpu.budget and spec.hardware.numGpusPerNode explicitly`,
			},
		},
		{
			name: "v1beta2 unknown explicit hardware is passed through with a warning",
			request: v1beta2DGDRForAdmission(func(spec map[string]any) {
				spec["hardware"] = map[string]any{
					"gpu":            map[string]any{"sku": "future_accelerator", "budget": int64(4)},
					"numGpusPerNode": int64(2),
				}
			}),
			gpuDiscovery: true,
			discoveryErr: errors.New("discovery must not run for complete explicit hardware"),
			wantWarnings: []string{
				`spec.hardware.gpu.sku: "future_accelerator" is not known to the AISimulate release bundled with this operator; the value is passed through unchanged and may work with an external AISimulate systems catalog`,
			},
			check: func(t *testing.T, result *unstructured.Unstructured) {
				if got := v1beta2String(t, result, "spec", "hardware", "gpu", "sku"); got != "future_accelerator" {
					t.Fatalf("spec.hardware.gpu.sku = %q, want unchanged override", got)
				}
			},
		},
		{
			name: "v1beta2 bundled hardware does not warn",
			request: v1beta2DGDRForAdmission(func(spec map[string]any) {
				spec["hardware"] = map[string]any{
					"gpu":            map[string]any{"sku": "h200_sxm", "budget": int64(4)},
					"numGpusPerNode": int64(8),
				}
			}),
			gpuDiscovery: true,
			discoveryErr: errors.New("discovery must not run for complete explicit hardware"),
		},

		{
			name: "v1beta2 public input overrides may configure non-reserved dgdr run fields",
			request: v1beta2DGDRForAdmission(func(spec map[string]any) {
				setV1beta2InputOverrides(spec, map[string]any{"dgdr_run": map[string]any{
					"generator":                 "aic",
					"snapshot_interval_seconds": int64(30),
				}})
			}),
			gpuDiscovery: true,
			discovered:   discoveredH200,
			check: func(t *testing.T, result *unstructured.Unstructured) {
				if got := v1beta2String(t, result, "spec", "aisimulate", "inputOverrides", "dgdr_run", "generator"); got != "aic" {
					t.Fatalf("dgdr_run.generator = %q, want aic", got)
				}
			},
		},
		{
			name: "v1beta2 operator-owned dgdr run input cannot be overridden",
			request: v1beta2DGDRForAdmission(func(spec map[string]any) {
				setV1beta2InputOverrides(spec, map[string]any{"dgdr_run": map[string]any{
					"derive_runtime_images":    false,
					"max_candidates":           int64(3),
					"name":                     "foreign-run",
					"namespace":                "other",
					"num_gpus_per_node":        int64(4),
					"runtime_image":            "example.invalid/runtime:latest",
					"runtime_version_override": "1.2.3",
					"snapshot_dir":             "/tmp/snapshots",
				}})
			}),
			gpuDiscovery: true,
			discovered:   discoveredH200,
			wantWebhook: []string{
				"spec.aisimulate.inputOverrides.dgdr_run.derive_runtime_images: Forbidden: is managed by the operator and cannot be overridden",
				"spec.aisimulate.inputOverrides.dgdr_run.max_candidates: Forbidden: is managed by the operator and cannot be overridden",
				"spec.aisimulate.inputOverrides.dgdr_run.name: Forbidden: is managed by the operator and cannot be overridden",
				"spec.aisimulate.inputOverrides.dgdr_run.namespace: Forbidden: is managed by the operator and cannot be overridden",
				"spec.aisimulate.inputOverrides.dgdr_run.num_gpus_per_node: Forbidden: is managed by the operator and cannot be overridden",
				"spec.aisimulate.inputOverrides.dgdr_run.runtime_image: Forbidden: is managed by the operator and cannot be overridden",
				"spec.aisimulate.inputOverrides.dgdr_run.runtime_version_override: Forbidden: is managed by the operator and cannot be overridden",
				"spec.aisimulate.inputOverrides.dgdr_run.snapshot_dir: Forbidden: is managed by the operator and cannot be overridden",
			},
		},
		{
			name: "v1beta2 dgdr run input cannot be deleted",
			request: v1beta2DGDRForAdmission(func(spec map[string]any) {
				setV1beta2InputOverrides(spec, map[string]any{"dgdr_run": nil})
			}),
			gpuDiscovery: true,
			discovered:   discoveredH200,
			wantWebhook: []string{
				"spec.aisimulate.inputOverrides.dgdr_run: Forbidden: must be an object because the operator owns required execution fields within it",
			},
		},
		{
			name: "v1beta2 dgdr run input cannot be replaced by a scalar",
			request: v1beta2DGDRForAdmission(func(spec map[string]any) {
				setV1beta2InputOverrides(spec, map[string]any{"dgdr_run": "wrong"})
			}),
			gpuDiscovery: true,
			discovered:   discoveredH200,
			wantWebhook: []string{
				"spec.aisimulate.inputOverrides.dgdr_run: Forbidden: must be an object because the operator owns required execution fields within it",
			},
		},
		{
			name: "v1beta2 dgdr run input cannot be replaced by a list",
			request: v1beta2DGDRForAdmission(func(spec map[string]any) {
				setV1beta2InputOverrides(spec, map[string]any{"dgdr_run": []any{"wrong"}})
			}),
			gpuDiscovery: true,
			discovered:   discoveredH200,
			wantWebhook: []string{
				"spec.aisimulate.inputOverrides.dgdr_run: Forbidden: must be an object because the operator owns required execution fields within it",
			},
		},
		{
			name: "v1beta2 pareto objective is reserved until Candidate publication supports it",
			request: v1beta2DGDRForAdmission(func(spec map[string]any) {
				spec["objective"] = map[string]any{"mode": "Pareto", "metrics": []any{"ThroughputPerGPU", "ThroughputPerUser"}}
			}),
			gpuDiscovery:  true,
			discovered:    discoveredH200,
			wantSchemaErr: `spec.objective.mode: Unsupported value: "Pareto": supported values: "Optimize"`,
		},
		{
			name: "v1beta2 optimize rejects the reserved Pareto metrics field",
			request: v1beta2DGDRForAdmission(func(spec map[string]any) {
				spec["objective"] = map[string]any{"mode": "Optimize", "metrics": []any{"ThroughputPerGPU", "ThroughputPerUser"}}
			}),
			gpuDiscovery: true,
			discovered:   discoveredH200,
			wantCELErr:   `spec.objective: Invalid value: metrics is reserved for the future Pareto implementation`,
		},
		{
			name: "v1beta2 duplicate backends are rejected on create",
			request: v1beta2DGDRForAdmission(func(spec map[string]any) {
				spec["backends"] = []any{"vllm", "vllm"}
			}),
			gpuDiscovery:  true,
			discovered:    discoveredH200,
			wantSchemaErr: `spec.backends[1]: Duplicate value: "vllm"`,
		},
		{
			name:         "v1beta2 missing model is rejected by the representation CEL",
			request:      v1beta2DGDRForAdmission(func(spec map[string]any) { delete(spec, "model") }),
			gpuDiscovery: true,
			discovered:   discoveredH200,
			wantCELErr:   `spec: Invalid value: exactly one of native v1beta2 fields or v1beta1 must be specified`,
		},
		{
			name: "v1beta2 empty model name is rejected by the schema",
			request: v1beta2DGDRForAdmission(func(spec map[string]any) {
				spec["model"] = map[string]any{"name": ""}
			}),
			gpuDiscovery:  true,
			discovered:    discoveredH200,
			wantSchemaErr: `spec.model.name: Invalid value: "": spec.model.name in body should be at least 1 chars long`,
		},
		{
			name: "v1beta2 KV capacity fraction range needs min below max",
			request: v1beta2DGDRForAdmission(func(spec map[string]any) {
				spec["workload"] = map[string]any{"static": map[string]any{"load": map[string]any{
					"kvCapacityFraction": map[string]any{"range": map[string]any{"min": 0.9, "max": 0.1}},
				}}}
			}),
			gpuDiscovery: true,
			discovered:   discoveredH200,
			wantCELErr:   `spec.workload.static.load.kvCapacityFraction.range: Invalid value: min must be less than max`,
		},
		{
			name: "v1beta2 KV capacity fraction must be positive",
			request: v1beta2DGDRForAdmission(func(spec map[string]any) {
				spec["workload"] = map[string]any{"static": map[string]any{"load": map[string]any{
					"kvCapacityFraction": map[string]any{"value": 0.0},
				}}}
			}),
			gpuDiscovery:  true,
			discovered:    discoveredH200,
			wantSchemaErr: `spec.workload.static.load.kvCapacityFraction.value: Invalid value: 0: spec.workload.static.load.kvCapacityFraction.value in body should be greater than 0`,
		},
		{
			name: "v1beta2 load requires exactly one variant",
			request: v1beta2DGDRForAdmission(func(spec map[string]any) {
				spec["workload"] = map[string]any{"static": map[string]any{"load": map[string]any{
					"concurrency": int64(10), "requestsPerSecond": 10.0,
				}}}
			}),
			gpuDiscovery: true,
			discovered:   discoveredH200,
			wantCELErr:   `spec.workload.static.load: Invalid value: exactly one of requestsPerSecond, concurrency, or kvCapacityFraction must be specified`,
		},
		{
			name: "v1beta2 empty load is not a second default opt-out",
			request: v1beta2DGDRForAdmission(func(spec map[string]any) {
				spec["workload"] = map[string]any{"static": map[string]any{"load": map[string]any{}}}
			}),
			gpuDiscovery: true,
			discovered:   discoveredH200,
			wantCELErr:   `spec.workload.static.load: Invalid value: exactly one of requestsPerSecond, concurrency, or kvCapacityFraction must be specified`,
		},
		{
			name:       "v1beta2 update to duplicate backends is rejected",
			oldRequest: v1beta2DGDRForAdmission(nil),
			request: v1beta2DGDRForAdmission(func(spec map[string]any) {
				spec["backends"] = []any{"sglang", "sglang"}
			}),
			gpuDiscovery:  true,
			discovered:    discoveredH200,
			wantSchemaErr: `spec.backends[1]: Duplicate value: "sglang"`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			admissionHardware.set(tt.discovered, tt.discoveryErr)
			t.Cleanup(func() { admissionHardware.set(gpu.GPUInfo{}, nil) })

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

func TestDynamoGraphDeploymentRunSpecIsImmutable(t *testing.T) {
	tests := []struct {
		name       string
		oldRun     *nvidiacomv1beta2.DynamoGraphDeploymentRun
		run        *nvidiacomv1beta2.DynamoGraphDeploymentRun
		wantCELErr string
	}{
		{
			name:   "metadata-only update keeps the immutable spec",
			oldRun: v1beta2RunForAdmission(v1beta2AdmissionModel, nil),
			run:    v1beta2RunForAdmission(v1beta2AdmissionModel, map[string]string{"updated": "true"}),
		},
		{
			name:       "spec update is rejected by the API server",
			oldRun:     v1beta2RunForAdmission(v1beta2AdmissionModel, nil),
			run:        v1beta2RunForAdmission(alternateAdmissionModel, nil),
			wantCELErr: `<nil>: Invalid value: spec is immutable`,
		},
		{
			name:       "model-only Run is rejected before controller reconciliation",
			run:        v1beta2ModelOnlyRunForAdmission(v1beta2AdmissionModel),
			wantCELErr: `spec: Invalid value: spec must be a fully resolved Request admission snapshot`,
		},
		{
			name: "empty hardware is an accepted resolved opt-out",
			run: func() *nvidiacomv1beta2.DynamoGraphDeploymentRun {
				run := v1beta2RunForAdmission(v1beta2AdmissionModel, nil)
				run.Spec.Hardware = &nvidiacomv1beta2.HardwareSpec{}
				return run
			}(),
		},
		{
			name: "partial hardware is rejected",
			run: func() *nvidiacomv1beta2.DynamoGraphDeploymentRun {
				run := v1beta2RunForAdmission(v1beta2AdmissionModel, nil)
				run.Spec.Hardware = &nvidiacomv1beta2.HardwareSpec{GPU: &nvidiacomv1beta2.GPUHardwareSpec{
					SKU: new("h200_sxm"), Budget: new(int32(8)),
				}}
				return run
			}(),
			wantCELErr: `spec: Invalid value: spec must be a fully resolved Request admission snapshot`,
		},
		{
			name: "partial static workload is rejected",
			run: func() *nvidiacomv1beta2.DynamoGraphDeploymentRun {
				run := v1beta2RunForAdmission(v1beta2AdmissionModel, nil)
				run.Spec.Workload.Static.InputTokens = new(int32(1024))
				return run
			}(),
			wantCELErr: `spec: Invalid value: spec must be a fully resolved Request admission snapshot`,
		},
		{
			name: "missing SLA snapshot is rejected",
			run: func() *nvidiacomv1beta2.DynamoGraphDeploymentRun {
				run := v1beta2RunForAdmission(v1beta2AdmissionModel, nil)
				run.Spec.Objective.SLA = nil
				return run
			}(),
			wantCELErr: `spec: Invalid value: spec must be a fully resolved Request admission snapshot`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			test := admissionTestCase{
				object:       tt.run,
				wantCELError: tt.wantCELErr,
			}
			if tt.oldRun != nil {
				test.oldObject = tt.oldRun
			}
			actual := runAdmissionTest(t, test)
			if tt.wantCELErr == "" && actual.GetGeneration() != 1 {
				t.Fatalf("generation = %d, want 1 for a metadata-only update", actual.GetGeneration())
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
			setAdmissionH100Topology()
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
	setAdmissionH100Topology()
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
		APIVersion: nvidiacomv1beta1.GroupVersion.String(),
		Kind:       "DynamoGraphDeploymentRequest",
		Name:       "test-dgdr", Namespace: "default",
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
		APIVersion: nvidiacomv1alpha1.GroupVersion.String(),
		Kind:       "DynamoGraphDeploymentRequest",
		Name:       "test-dgdr", Namespace: "default",
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
		APIVersion: nvidiacomv1beta2.GroupVersion.String(),
		Kind:       "DynamoGraphDeploymentRequest",
		Name:       "test-dgdr", Namespace: "default",
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

func v1beta2RunForAdmission(model string, labels map[string]string) *nvidiacomv1beta2.DynamoGraphDeploymentRun {
	metric := nvidiacomv1beta2.ObjectiveMetric("ThroughputPerGPU")
	return &nvidiacomv1beta2.DynamoGraphDeploymentRun{
		APIVersion: nvidiacomv1beta2.GroupVersion.String(),
		Kind:       "DynamoGraphDeploymentRun",
		Name:       "test-v1beta2-dgdr-run", Namespace: "default", Labels: labels,
		Spec: nvidiacomv1beta2.DynamoGraphDeploymentRunSpec{NativeSearchInput: nvidiacomv1beta2.NativeSearchInput{
			Model:    &nvidiacomv1beta2.ModelSpec{Name: model},
			Backends: []nvidiacomv1beta2.Backend{nvidiacomv1beta2.BackendVLLM},
			Hardware: &nvidiacomv1beta2.HardwareSpec{
				GPU:            &nvidiacomv1beta2.GPUHardwareSpec{SKU: new("h200_sxm"), Budget: new(int32(8))},
				NumGPUsPerNode: new(int32(8)),
			},
			Workload: &nvidiacomv1beta2.WorkloadSpec{Static: &nvidiacomv1beta2.StaticWorkloadSpec{}},
			Objective: &nvidiacomv1beta2.ObjectiveSpec{
				Mode: nvidiacomv1beta2.ObjectiveModeOptimize, Metric: &metric, SLA: &nvidiacomv1beta2.ObjectiveSLASpec{},
			},
			Recommendation: &nvidiacomv1beta2.RecommendationSpec{MaxCandidates: 1},
		}},
	}
}

func v1beta2ModelOnlyRunForAdmission(model string) *nvidiacomv1beta2.DynamoGraphDeploymentRun {
	run := v1beta2RunForAdmission(model, nil)
	run.Spec.NativeSearchInput = nvidiacomv1beta2.NativeSearchInput{
		Model: &nvidiacomv1beta2.ModelSpec{Name: model},
	}
	return run
}

const v1beta2AdmissionModel = "Qwen/Qwen3-0.6B"

// v1beta2DGDRForAdmission builds a v1beta2 Request as a client sends it, before the CRD
// defaults. It is unstructured on purpose: an omitted field,
// an explicit empty object and a null are different requests, and typed Go cannot tell them
// apart.
func v1beta2DGDRForAdmission(mutate func(spec map[string]any)) *unstructured.Unstructured {
	spec := map[string]any{"model": map[string]any{"name": v1beta2AdmissionModel}}
	if mutate != nil {
		mutate(spec)
	}
	metadata := map[string]any{"name": "test-v1beta2-dgdr", "namespace": "default"}
	return &unstructured.Unstructured{Object: map[string]any{
		"apiVersion": "nvidia.com/v1beta2",
		"kind":       "DynamoGraphDeploymentRequest",
		"metadata":   metadata,
		"spec":       spec,
	}}
}

func v1beta2Field(t *testing.T, obj *unstructured.Unstructured, path ...string) (any, bool) {
	t.Helper()
	value, found, err := unstructured.NestedFieldNoCopy(obj.Object, path...)
	if err != nil {
		t.Fatalf("read %v: %v", path, err)
	}
	return value, found
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

// setV1beta2InputOverrides replaces any input overrides the fixture carried.
func setV1beta2InputOverrides(spec map[string]any, overrides map[string]any) {
	aisimulate, _ := spec["aisimulate"].(map[string]any)
	if aisimulate == nil {
		aisimulate = map[string]any{}
		spec["aisimulate"] = aisimulate
	}
	aisimulate["inputOverrides"] = overrides
}

// setBeta2AdmissionHardware names the hardware, so fixtures do not depend on GPU discovery.
func setBeta2AdmissionHardware(request *nvidiacomv1beta2.DynamoGraphDeploymentRequest) {
	sku, budget, numGPUsPerNode := "h100_sxm", int32(4), int32(4)
	request.Spec.Hardware = &nvidiacomv1beta2.HardwareSpec{
		GPU:            &nvidiacomv1beta2.GPUHardwareSpec{SKU: &sku, Budget: &budget},
		NumGPUsPerNode: &numGPUsPerNode,
	}
}
