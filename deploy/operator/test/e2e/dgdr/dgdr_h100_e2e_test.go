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

package dgdr

import (
	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	. "github.com/onsi/ginkgo/v2"
	"k8s.io/utils/ptr"
)

// DGDR Support Matrix on H100 SKU exercises the full DGDR lifecycle (create -> profile ->
// DGD generation -> DGD readiness) across a curated
var _ = Describe("DGDR Support Matrix on H100 SKU", Label("gpu_0", "nightly", "integration", "k8s"), func() {
	// -----------------------------------------------------------------------
	// Support matrix — rapid profiling (in AIC matrix)
	// Models with pre-validated configs where rapid search finds a viable
	// configuration. All targeting the H100 test cluster.
	// -----------------------------------------------------------------------

	Context("Models that support AIC rapid mode", func() {
		// ---- Qwen3-32B (all backends) ----
		Context("should complete full lifecycle of Qwen3-32B", func() {
			qwen32bBase := func(backend v1beta1.BackendType, suffix string) DGDRLifecycleInput {
				return DGDRLifecycleInput{
					Name:           uniqueName("qwen32b-" + suffix),
					Model:          "Qwen/Qwen3-32B",
					Backend:        backend,
					SearchStrategy: v1beta1.SearchStrategyRapid,
					AutoApply:      ptr.To(true),
					SLA: &v1beta1.SLASpec{
						TTFT: ptr.To(500.0),
						ITL:  ptr.To(30.0),
					},
					Workload: &v1beta1.WorkloadSpec{
						ISL: ptr.To(int32(3000)),
						OSL: ptr.To(int32(300)),
					},
					Features: &v1beta1.FeaturesSpec{
						Planner: plannerRawExtension(map[string]interface{}{
							"mode":                      "disagg",
							"enable_throughput_scaling": true,
							"enable_load_scaling":       true,
							"max_gpu_budget":            32,
						}),
					},
					Hardware: &v1beta1.HardwareSpec{
						GPUSKU:         v1beta1.GPUSKUTypeH100SXM,
						VRAMMB:         ptr.To(float64(81920)),
						NumGPUsPerNode: ptr.To(int32(8)),
						TotalGPUs:      ptr.To(int32(32)),
					},
					ExpectDGDReady:  true,
					VerifyConfigMap: true,
				}
			}

			DescribeTable("Qwen3-32B on H100 with planner",
				func(backend v1beta1.BackendType) {
					By("Running DGDR lifecycle with Qwen3-32B + " + string(backend) + " + rapid + planner")
					input := qwen32bBase(backend, string(backend))
					input.VerifyServices = map[string]ServiceExpectation{
						"Planner": {MinReplicas: 1},
					}
					DGDRLifecycleSpec(ctx, func() DGDRLifecycleInput { return input })
				},
				Entry("vllm", v1beta1.BackendTypeVllm),
				Entry("sglang", v1beta1.BackendTypeSglang),
				Entry("trtllm", v1beta1.BackendTypeTrtllm),
			)
		})

		// ---- Qwen3-235B-A22B-FP8 (all backends) ----
		Context("should complete full lifecycle of Qwen3-235B-A22B-FP8", func() {
			qwen235bBase := func(backend v1beta1.BackendType, suffix string) DGDRLifecycleInput {
				return DGDRLifecycleInput{
					Name:           uniqueName("qwen235b-" + suffix),
					Model:          "Qwen/Qwen3-235B-A22B-FP8",
					Backend:        backend,
					SearchStrategy: v1beta1.SearchStrategyRapid,
					AutoApply:      ptr.To(true),
					SLA: &v1beta1.SLASpec{
						TTFT: ptr.To(500.0),
						ITL:  ptr.To(30.0),
					},
					Workload: &v1beta1.WorkloadSpec{
						ISL: ptr.To(int32(3000)),
						OSL: ptr.To(int32(300)),
					},
					Hardware: &v1beta1.HardwareSpec{
						GPUSKU:         v1beta1.GPUSKUTypeH100SXM,
						VRAMMB:         ptr.To(float64(81920)),
						NumGPUsPerNode: ptr.To(int32(8)),
						TotalGPUs:      ptr.To(int32(32)),
					},
					Features: &v1beta1.FeaturesSpec{
						Planner: plannerRawExtension(map[string]interface{}{
							"mode":                      "disagg",
							"enable_throughput_scaling": true,
							"enable_load_scaling":       true,
							"max_gpu_budget":            32,
						}),
					},
					Overrides: &v1beta1.OverridesSpec{
						DGD: dgdOverrideRawExtension(map[string]interface{}{
							"apiVersion": "nvidia.com/v1alpha1",
							"kind":       "DynamoGraphDeployment",
							"metadata":   map[string]interface{}{"name": "q235"},
							"spec": map[string]interface{}{
								"services": map[string]interface{}{
									"prefill": map[string]interface{}{
										"sharedMemory": map[string]interface{}{"size": "256Gi"},
									},
									"decode": map[string]interface{}{
										"sharedMemory": map[string]interface{}{"size": "256Gi"},
									},
								},
							},
						}),
					},
					ExpectDGDReady:  true,
					VerifyConfigMap: true,
				}
			}

			DescribeTable("Qwen3-235B-A22B-FP8 on H100 with planner",
				func(backend v1beta1.BackendType) {
					By("Running DGDR lifecycle with Qwen3-235B-A22B-FP8 + " + string(backend) + " + rapid + planner")
					input := qwen235bBase(backend, string(backend))
					input.VerifyServices = map[string]ServiceExpectation{
						"Planner": {MinReplicas: 1},
					}
					DGDRLifecycleSpec(ctx, func() DGDRLifecycleInput { return input })
				},
				Entry("vllm", v1beta1.BackendTypeVllm),
				Entry("sglang", v1beta1.BackendTypeSglang),
				Entry("trtllm", v1beta1.BackendTypeTrtllm),
			)
		})

		// ---- GPT-OSS-20B (trtllm only, vllm and sglang do not support moe quant mode 'w4a16_mxfp4'----
		Context("should complete full lifecycle of GPT-OSS-20B", func() {
			It("Running DGDR lifecycle with GPT-OSS-20B + trtllm + rapid", func() {
				DGDRLifecycleSpec(ctx, func() DGDRLifecycleInput {
					return DGDRLifecycleInput{
						Name:           uniqueName("gptoss20b"),
						Model:          "openai/gpt-oss-20b",
						Backend:        v1beta1.BackendTypeTrtllm,
						SearchStrategy: v1beta1.SearchStrategyRapid,
						AutoApply:      ptr.To(true),
						SLA: &v1beta1.SLASpec{
							TTFT: ptr.To(500.0),
							ITL:  ptr.To(30.0),
						},
						Workload: &v1beta1.WorkloadSpec{
							ISL: ptr.To(int32(3000)),
							OSL: ptr.To(int32(300)),
						},
						Hardware: &v1beta1.HardwareSpec{
							GPUSKU:         v1beta1.GPUSKUTypeH100SXM,
							VRAMMB:         ptr.To(float64(81920)),
							NumGPUsPerNode: ptr.To(int32(8)),
							TotalGPUs:      ptr.To(int32(32)),
						},
						Features: &v1beta1.FeaturesSpec{
							Planner: plannerRawExtension(map[string]interface{}{
								"mode":                      "disagg",
								"enable_throughput_scaling": true,
								"enable_load_scaling":       true,
								"max_gpu_budget":            32,
							}),
						},
						Overrides: &v1beta1.OverridesSpec{
							DGD: dgdOverrideRawExtension(map[string]interface{}{
								"apiVersion": "nvidia.com/v1alpha1",
								"kind":       "DynamoGraphDeployment",
								"metadata":   map[string]interface{}{"name": "gptoss"},
								"spec": map[string]interface{}{
									"services": map[string]interface{}{
										"worker": map[string]interface{}{
											"sharedMemory": map[string]interface{}{"size": "80Gi"},
										},
									},
								},
							}),
						},
						ExpectDGDReady:  true,
						VerifyConfigMap: true,
					}
				})
			})
		})

		// Other models:
		// ---- GPT-OSS-120B (trtllm only, works on AIC 1.2.0rc5 but not current AIC version 1.3.0rc10) ----
		// ---- meta-llama/Meta-Llama-3.1-70B (all backends) ----
		// Note: moonshotai/Kimi-K2.5 works with sglang and vllm, but it might not fit on H100s.

	})

	// -----------------------------------------------------------------------
	// Support matrix — thorough profiling (not in AIC matrix)
	// Models that need thorough search to explore the config space.
	// All targeting the H100 test cluster.
	// -----------------------------------------------------------------------

	Context("Models that require thorough profiling", func() {
		// TODO: add Qwen3-VL-30B-FP8, Llama-3.3-70B-FP8, and Nemotron-3-Super-120B-FP8
	})

})
