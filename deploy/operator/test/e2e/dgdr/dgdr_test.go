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
	. "github.com/onsi/gomega"
	"k8s.io/utils/ptr"
)

// DGDR Lifecycle Scenarios exercises the full DGDR lifecycle (create -> profile ->
// DGD generation -> DGD readiness) across different backend, strategy, and feature
// configurations. Modeled after the CAAPH helm_test.go pattern: each It block calls
// DGDRLifecycleSpec with a different input, and multiple specs can be composed
// sequentially within a single It to test multi-step workflows.
var _ = Describe("DGDR Lifecycle Scenarios", Label("gpu_0", "nightly", "integration", "k8s"), func() {

	// -----------------------------------------------------------------------
	// Backend variations — rapid profiling with each supported backend
	// -----------------------------------------------------------------------

	Context("Backend variations with rapid profiling", func() {

		It("should complete full lifecycle with vllm backend", func() {
			By("Running DGDR lifecycle with vllm + rapid + autoApply")
			DGDRLifecycleSpec(ctx, func() DGDRLifecycleInput {
				return rapidLifecycleInput(uniqueName("vllm-rapid"), v1beta1.BackendTypeVllm)
			})
		})

		It("should complete full lifecycle with sglang backend", func() {
			By("Running DGDR lifecycle with sglang + rapid + autoApply")
			DGDRLifecycleSpec(ctx, func() DGDRLifecycleInput {
				return rapidLifecycleInput(uniqueName("sglang-rapid"), v1beta1.BackendTypeSglang)
			})
		})

		It("should complete full lifecycle with trtllm backend", func() {
			By("Running DGDR lifecycle with trtllm + rapid + autoApply")
			DGDRLifecycleSpec(ctx, func() DGDRLifecycleInput {
				return rapidLifecycleInput(uniqueName("trtllm-rapid"), v1beta1.BackendTypeTrtllm)
			})
		})
	})

	// -----------------------------------------------------------------------
	// Search strategy and autoApply variations
	// -----------------------------------------------------------------------

	Context("Search strategy and autoApply variations", func() {

		It("should complete thorough profiling without deploying", func() {
			By("Running DGDR lifecycle with thorough + autoApply=false")
			DGDRLifecycleSpec(ctx, func() DGDRLifecycleInput {
				return DGDRLifecycleInput{
					Name:            uniqueName("vllm-thorough"),
					Backend:         v1beta1.BackendTypeVllm,
					SearchStrategy:  v1beta1.SearchStrategyThorough,
					AutoApply:       ptr.To(false),
					ExpectDGDReady:  false,
					VerifyConfigMap: true,
				}
			})
		})

		It("should reach Ready but not Deployed with autoApply=false", func() {
			By("Running DGDR lifecycle with rapid + autoApply=false")
			DGDRLifecycleSpec(ctx, func() DGDRLifecycleInput {
				return DGDRLifecycleInput{
					Name:            uniqueName("no-autoapply"),
					AutoApply:       ptr.To(false),
					ExpectDGDReady:  false,
					VerifyConfigMap: true,
				}
			})
		})
	})

	// -----------------------------------------------------------------------
	// Feature combinations
	// -----------------------------------------------------------------------

	Context("Feature combinations", func() {

		It("should include Planner service when planner is enabled", func() {
			By("Running DGDR lifecycle with trtllm + planner enabled")
			DGDRLifecycleSpec(ctx, func() DGDRLifecycleInput {
				return DGDRLifecycleInput{
					Name:           uniqueName("planner"),
					Backend:        v1beta1.BackendTypeTrtllm,
					SearchStrategy: v1beta1.SearchStrategyRapid,
					AutoApply:      ptr.To(true),
					Features: &v1beta1.FeaturesSpec{
						Planner: plannerRawExtension(map[string]interface{}{
							"enabled":                      true,
							"optimization_target":          "sla",
							"pre_deployment_sweeping_mode": "rapid",
						}),
					},
					ExpectDGDReady: true,
					VerifyServices: map[string]ServiceExpectation{
						"Planner": {MinReplicas: 1},
					},
					VerifyConfigMap: true,
				}
			})
		})
	})

	// -----------------------------------------------------------------------
	// SLA and workload parameter variations
	// -----------------------------------------------------------------------

	Context("SLA and workload parameter variations", func() {

		It("should profile with custom SLA and workload constraints", func() {
			By("Running DGDR lifecycle with custom TTFT/ITL and ISL/OSL")
			DGDRLifecycleSpec(ctx, func() DGDRLifecycleInput {
				return DGDRLifecycleInput{
					Name:      uniqueName("custom-sla"),
					AutoApply: ptr.To(true),
					SLA: &v1beta1.SLASpec{
						TTFT: ptr.To(2000.0),
						ITL:  ptr.To(30.0),
					},
					Workload: &v1beta1.WorkloadSpec{
						ISL: ptr.To(int32(4000)),
						OSL: ptr.To(int32(1000)),
					},
					ExpectDGDReady:  true,
					VerifyConfigMap: true,
				}
			})
		})

		It("should profile with latency optimization", func() {
			latencyOpt := v1beta1.OptimizationTypeLatency
			By("Running DGDR lifecycle with latency-optimized SLA")
			DGDRLifecycleSpec(ctx, func() DGDRLifecycleInput {
				return DGDRLifecycleInput{
					Name:      uniqueName("latency-opt"),
					AutoApply: ptr.To(true),
					SLA: &v1beta1.SLASpec{
						TTFT:             ptr.To(500.0),
						ITL:              ptr.To(15.0),
						OptimizationType: &latencyOpt,
					},
					ExpectDGDReady:  true,
					VerifyConfigMap: true,
				}
			})
		})

		It("should profile with throughput optimization", func() {
			throughputOpt := v1beta1.OptimizationTypeThroughput
			By("Running DGDR lifecycle with throughput-optimized SLA")
			DGDRLifecycleSpec(ctx, func() DGDRLifecycleInput {
				return DGDRLifecycleInput{
					Name:      uniqueName("throughput-opt"),
					AutoApply: ptr.To(true),
					SLA: &v1beta1.SLASpec{
						TTFT:             ptr.To(5000.0),
						ITL:              ptr.To(100.0),
						OptimizationType: &throughputOpt,
					},
					ExpectDGDReady:  true,
					VerifyConfigMap: true,
				}
			})
		})
	})

	// -----------------------------------------------------------------------
	// Hardware configuration
	// -----------------------------------------------------------------------

	Context("Hardware configuration", func() {

		It("should profile with specified GPU configuration", func() {
			By("Running DGDR lifecycle with custom A100 hardware spec")
			DGDRLifecycleSpec(ctx, func() DGDRLifecycleInput {
				return DGDRLifecycleInput{
					Name:      uniqueName("custom-hw"),
					AutoApply: ptr.To(true),
					Hardware: &v1beta1.HardwareSpec{
						GPUSKU:         v1beta1.GPUSKUTypeA100SXM,
						VRAMMB:         ptr.To(float64(81920)),
						NumGPUsPerNode: ptr.To(int32(8)),
						TotalGPUs:      ptr.To(int32(8)),
					},
					ExpectDGDReady:  true,
					VerifyConfigMap: true,
				}
			})
		})
	})

	// -----------------------------------------------------------------------
	// Support matrix — rapid profiling (in AIC matrix)
	// Models with pre-validated configs where rapid search finds a viable
	// configuration. All targeting the H100 test cluster.
	// -----------------------------------------------------------------------

	Context("Model-specific DGDR scenarios — rapid profiling", func() {
		// supports all backends
		It("should complete full lifecycle with Qwen3-32B-FP8 agg on H100", func() {
			// Also available: TRT-LLM disagg 8 H100/H200/A100, vLLM disagg 8 A100
			By("Running DGDR lifecycle with Qwen3-32B-FP8 + trtllm + rapid + agg")
			DGDRLifecycleSpec(ctx, func() DGDRLifecycleInput {
				return DGDRLifecycleInput{
					Name:           uniqueName("qwen32bfp8-agg"),
					Model:          "Qwen/Qwen3-32B-FP8",
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
					Overrides: &v1beta1.OverridesSpec{
						DGD: dgdOverrideRawExtension(map[string]interface{}{
							"apiVersion": "nvidia.com/v1alpha1",
							"kind":       "DynamoGraphDeployment",
							"metadata":   map[string]interface{}{"name": "q32fp8"},
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

		// only trtllm works for AIC 1.2.0rc5
		It("should complete full lifecycle with GPT-OSS-120B agg on H100", func() {
			// Also available: TRT-LLM disagg 5 GB200/B200
			By("Running DGDR lifecycle with GPT-OSS-120B + trtllm + rapid + agg")
			DGDRLifecycleSpec(ctx, func() DGDRLifecycleInput {
				return DGDRLifecycleInput{
					Name:           uniqueName("gptoss120b-agg"),
					Model:          "openai/gpt-oss-120b",
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

		// works with any backend
		It("should complete full lifecycle with Qwen3-235B-A22B-FP8 disagg on H100 [RAPID][PLANNER]", func() {
			// Also available: TRT-LLM agg 16 H100/H200
			By("Running DGDR lifecycle with Qwen3-235B-A22B-FP8 + trtllm + rapid + disagg")
			DGDRLifecycleSpec(ctx, func() DGDRLifecycleInput {
				return DGDRLifecycleInput{
					Name:           uniqueName("qwen235b-disagg"),
					Model:          "Qwen/Qwen3-235B-A22B-FP8",
					Backend:        v1beta1.c,
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
			})
		})

		// all backends
		It("should complete full lifecycle with Qwen3-32B on H100", func() {
			// Also available: vLLM disagg-kv-router 16 H200
			By("Running DGDR lifecycle with Qwen3-32B + trtllm + rapid") // Change for temp testing
			// By("Running DGDR lifecycle with Qwen3-32B + vllm + rapid")
			DGDRLifecycleSpec(ctx, func() DGDRLifecycleInput {
				return DGDRLifecycleInput{
					Name:  uniqueName("qwen32b"),
					Model: "Qwen/Qwen3-32B",
					// Backend:        v1beta1.BackendTypeVllm,
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
					ExpectDGDReady:  true,
					VerifyConfigMap: true,
				}
			})
		})

		// meta-llama/Meta-Llama-3.1-70B works on all backends with rapid.
	})

	// -----------------------------------------------------------------------
	// Support matrix — thorough profiling (not in AIC matrix)
	// Models that need thorough search to explore the config space.
	// All targeting the H100 test cluster.
	// -----------------------------------------------------------------------

	Context("Model-specific DGDR scenarios — thorough profiling", func() {

		// It("should complete full lifecycle with Qwen3-VL-30B-FP8 agg on H100", func() {
		// 	// Multimodal model; only recipe: vLLM agg 1 GB200
		// 	By("Running DGDR lifecycle with Qwen3-VL-30B-FP8 + vllm + thorough + agg")
		// 	DGDRLifecycleSpec(ctx, func() DGDRLifecycleInput {
		// 		return DGDRLifecycleInput{
		// 			Name:           uniqueName("qwen3vl30b"),
		// 			Model:          "Qwen/Qwen3-VL-30B-A3B-Instruct-FP8",
		// 			Backend:        v1beta1.BackendTypeVllm,
		// 			SearchStrategy: v1beta1.SearchStrategyThorough,
		// 			AutoApply:      ptr.To(true),
		// 			SLA: &v1beta1.SLASpec{
		// 				TTFT: ptr.To(2000.0),
		// 				ITL:  ptr.To(50.0),
		// 			},
		// 			Workload: &v1beta1.WorkloadSpec{
		// 				ISL: ptr.To(int32(4000)),
		// 				OSL: ptr.To(int32(1000)),
		// 			},
		// 			Hardware: &v1beta1.HardwareSpec{
		// 				GPUSKU:         v1beta1.GPUSKUTypeH100SXM,
		// 				VRAMMB:         ptr.To(float64(81920)),
		// 				NumGPUsPerNode: ptr.To(int32(8)),
		// 				TotalGPUs:      ptr.To(int32(32)),
		// 			},
		// 			ExpectDGDReady:  true,
		// 			VerifyConfigMap: true,
		// 		}
		// 	})
		// })

		// It("should complete full lifecycle with Llama-3.3-70B-FP8 agg on H100", func() {
		// 	// Also available: vLLM disagg single-node 8 H100/H200,
		// 	//   vLLM disagg multi-node 16 H100/H200,
		// 	//   vLLM agg+GAIE 4 H100/H200, vLLM disagg+GAIE 8 H100/H200
		// 	By("Running DGDR lifecycle with Llama-3.3-70B-FP8 + vllm + thorough + agg")
		// 	DGDRLifecycleSpec(ctx, func() DGDRLifecycleInput {
		// 		return DGDRLifecycleInput{
		// 			Name:           uniqueName("llama70b-agg"),
		// 			Model:          "RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic",
		// 			Backend:        v1beta1.BackendTypeVllm,
		// 			SearchStrategy: v1beta1.SearchStrategyThorough,
		// 			AutoApply:      ptr.To(true),
		// 			SLA: &v1beta1.SLASpec{
		// 				TTFT: ptr.To(2000.0),
		// 				ITL:  ptr.To(50.0),
		// 			},
		// 			Workload: &v1beta1.WorkloadSpec{
		// 				ISL: ptr.To(int32(4000)),
		// 				OSL: ptr.To(int32(1000)),
		// 			},
		// 			Hardware: &v1beta1.HardwareSpec{
		// 				GPUSKU:         v1beta1.GPUSKUTypeH100SXM,
		// 				VRAMMB:         ptr.To(float64(81920)),
		// 				NumGPUsPerNode: ptr.To(int32(8)),
		// 				TotalGPUs:      ptr.To(int32(32)),
		// 			},
		// 			Overrides: &v1beta1.OverridesSpec{
		// 				DGD: dgdOverrideRawExtension(map[string]interface{}{
		// 					"apiVersion": "nvidia.com/v1alpha1",
		// 					"kind":       "DynamoGraphDeployment",
		// 					"metadata":   map[string]interface{}{"name": "llama70b"},
		// 					"spec": map[string]interface{}{
		// 						"services": map[string]interface{}{
		// 							"worker": map[string]interface{}{
		// 								"sharedMemory": map[string]interface{}{"size": "20Gi"},
		// 							},
		// 						},
		// 					},
		// 				}),
		// 			},
		// 			ExpectDGDReady:  true,
		// 			VerifyConfigMap: true,
		// 		}
		// 	})
		// })

		// It("should complete full lifecycle with Nemotron-3-Super-120B-FP8 agg on H100", func() {
		// 	// Also available: SGLang disagg 4 H100/H200,
		// 	//   TRT-LLM disagg 4 H100/H200, vLLM agg 4 H100/H200
		// 	By("Running DGDR lifecycle with Nemotron-3-Super-120B-FP8 + sglang + thorough + agg")
		// 	DGDRLifecycleSpec(ctx, func() DGDRLifecycleInput {
		// 		return DGDRLifecycleInput{
		// 			Name:           uniqueName("nem120b-agg"),
		// 			Model:          "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8",
		// 			Backend:        v1beta1.BackendTypeSglang,
		// 			SearchStrategy: v1beta1.SearchStrategyThorough,
		// 			AutoApply:      ptr.To(true),
		// 			SLA: &v1beta1.SLASpec{
		// 				TTFT: ptr.To(2000.0),
		// 				ITL:  ptr.To(50.0),
		// 			},
		// 			Workload: &v1beta1.WorkloadSpec{
		// 				ISL: ptr.To(int32(4000)),
		// 				OSL: ptr.To(int32(1000)),
		// 			},
		// 			Hardware: &v1beta1.HardwareSpec{
		// 				GPUSKU:         v1beta1.GPUSKUTypeH100SXM,
		// 				VRAMMB:         ptr.To(float64(81920)),
		// 				NumGPUsPerNode: ptr.To(int32(8)),
		// 				TotalGPUs:      ptr.To(int32(32)),
		// 			},
		// 			Overrides: &v1beta1.OverridesSpec{
		// 				DGD: dgdOverrideRawExtension(map[string]interface{}{
		// 					"apiVersion": "nvidia.com/v1alpha1",
		// 					"kind":       "DynamoGraphDeployment",
		// 					"metadata":   map[string]interface{}{"name": "nem120b"},
		// 					"spec": map[string]interface{}{
		// 						"services": map[string]interface{}{
		// 							"worker": map[string]interface{}{
		// 								"sharedMemory": map[string]interface{}{"size": "16Gi"},
		// 							},
		// 						},
		// 					},
		// 				}),
		// 			},
		// 			ExpectDGDReady:  true,
		// 			VerifyConfigMap: true,
		// 		}
		// 	})
		// })

		// Note: moonshotai/Kimi-K2.5 does support AIC rapid

	})

	// -----------------------------------------------------------------------
	// Support matrix — models that don't fit on H100
	// These large MoE models require non-H100 SKUs (H200/GB200) and are
	// excluded from the H100 test cluster. Kept here for completeness;
	// run these on clusters with the appropriate GPU SKU.
	// -----------------------------------------------------------------------

	// Context("Model-specific DGDR scenarios — non-H100 SKUs", func() {

	// 	It("should complete full lifecycle with DeepSeek-R1 disagg on H200", func() {
	// 		// Also available: SGLang disagg 32 H200,
	// 		//   TRT-LLM disagg wide-EP 36 GB200, vLLM disagg 32 H200
	// 		By("Running DGDR lifecycle with DeepSeek-R1 + sglang + thorough + disagg")
	// 		DGDRLifecycleSpec(ctx, func() DGDRLifecycleInput {
	// 			return DGDRLifecycleInput{
	// 				Name:           uniqueName("dsr1-disagg"),
	// 				Model:          "deepseek-ai/DeepSeek-R1",
	// 				Backend:        v1beta1.BackendTypeSglang,
	// 				SearchStrategy: v1beta1.SearchStrategyThorough,
	// 				AutoApply:      ptr.To(true),
	// 				SLA: &v1beta1.SLASpec{
	// 					TTFT: ptr.To(2000.0),
	// 					ITL:  ptr.To(50.0),
	// 				},
	// 				Workload: &v1beta1.WorkloadSpec{
	// 					ISL: ptr.To(int32(4000)),
	// 					OSL: ptr.To(int32(1000)),
	// 				},
	// 				Hardware: &v1beta1.HardwareSpec{
	// 					GPUSKU:         v1beta1.GPUSKUTypeH200SXM,
	// 					VRAMMB:         ptr.To(float64(144384)),
	// 					NumGPUsPerNode: ptr.To(int32(8)),
	// 					TotalGPUs:      ptr.To(int32(16)),
	// 				},
	// 				Features: &v1beta1.FeaturesSpec{
	// 					Planner: plannerRawExtension(map[string]interface{}{
	// 						"mode":                      "disagg",
	// 						"enable_throughput_scaling": true,
	// 						"enable_load_scaling":       true,
	// 						"max_gpu_budget":            16,
	// 					}),
	// 				},
	// 				Overrides: &v1beta1.OverridesSpec{
	// 					DGD: dgdOverrideRawExtension(map[string]interface{}{
	// 						"apiVersion": "nvidia.com/v1alpha1",
	// 						"kind":       "DynamoGraphDeployment",
	// 						"metadata":   map[string]interface{}{"name": "dsr1"},
	// 						"spec": map[string]interface{}{
	// 							"services": map[string]interface{}{
	// 								"prefill": map[string]interface{}{
	// 									"sharedMemory": map[string]interface{}{"size": "80Gi"},
	// 								},
	// 								"decode": map[string]interface{}{
	// 									"sharedMemory": map[string]interface{}{"size": "80Gi"},
	// 								},
	// 							},
	// 						},
	// 					}),
	// 				},
	// 				ExpectDGDReady:  true,
	// 				VerifyConfigMap: true,
	// 			}
	// 		})
	// 	})
	// })

	// -----------------------------------------------------------------------
	// Multi-step workflows — compose multiple DGDRLifecycleSpec calls within
	// a single It block to test sequential scenarios (CAAPH helm_test.go style).
	// -----------------------------------------------------------------------

	Context("Multi-step workflows", func() {

		It("should run rapid profiling across multiple backends sequentially", func() {
			By("Running vllm rapid lifecycle")
			DGDRLifecycleSpec(ctx, func() DGDRLifecycleInput {
				return DGDRLifecycleInput{
					Name:            uniqueName("multi-vllm"),
					Backend:         v1beta1.BackendTypeVllm,
					SearchStrategy:  v1beta1.SearchStrategyRapid,
					AutoApply:       ptr.To(false),
					VerifyConfigMap: true,
				}
			})

			By("Running sglang rapid lifecycle")
			DGDRLifecycleSpec(ctx, func() DGDRLifecycleInput {
				return DGDRLifecycleInput{
					Name:            uniqueName("multi-sglang"),
					Backend:         v1beta1.BackendTypeSglang,
					SearchStrategy:  v1beta1.SearchStrategyRapid,
					AutoApply:       ptr.To(false),
					VerifyConfigMap: true,
				}
			})

			By("Running trtllm rapid lifecycle")
			DGDRLifecycleSpec(ctx, func() DGDRLifecycleInput {
				return DGDRLifecycleInput{
					Name:            uniqueName("multi-trtllm"),
					Backend:         v1beta1.BackendTypeTrtllm,
					SearchStrategy:  v1beta1.SearchStrategyRapid,
					AutoApply:       ptr.To(false),
					VerifyConfigMap: true,
				}
			})
		})

		It("should profile then deploy with custom SLA constraints", func() {
			// Step 1: Profile without deploying
			name := uniqueName("two-step")
			By("Running profiling-only step (autoApply=false)")
			DGDRLifecycleSpec(ctx, func() DGDRLifecycleInput {
				return DGDRLifecycleInput{
					Name:      name,
					AutoApply: ptr.To(false),
					SLA: &v1beta1.SLASpec{
						TTFT: ptr.To(2000.0),
						ITL:  ptr.To(30.0),
					},
					Workload: &v1beta1.WorkloadSpec{
						ISL: ptr.To(int32(4000)),
						OSL: ptr.To(int32(1000)),
					},
					VerifyConfigMap: true,
				}
			})

			// Step 2: Verify the output ConfigMap is accessible after profiling
			By("Verifying output ConfigMap is still accessible")
			data := getOutputConfigMap(name)
			_, ok := data["final_config.yaml"]
			Expect(ok).To(BeTrue(), "ConfigMap should still contain final_config.yaml")
		})
	})
})
