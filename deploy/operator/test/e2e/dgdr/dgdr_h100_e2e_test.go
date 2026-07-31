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

// h100VRAMMB is the per-GPU VRAM (MiB) for the H100 SXM test cluster (80 GiB).
// Kept as a const because the test cluster's GPU SKU is fixed; unlike the
// SLA/workload/budget knobs below it is not expected to vary between runs.
const h100VRAMMB float64 = 81920

// H100 support-matrix tunables. These default to the values the suite was
// originally written against and can be overridden via environment variables so
// the same specs can target a differently-sized cluster or different SLAs
// without code changes. Defaults are applied when the env var is unset or
// unparseable (see getenvFloat64 / getenvInt32).
var (
	// SLA targets.
	h100TTFTMillis = getenvFloat64("DGDR_H100_TTFT_MS", 500.0) // time-to-first-token (ms)
	h100ITLMillis  = getenvFloat64("DGDR_H100_ITL_MS", 30.0)   // inter-token latency (ms)

	// Workload shape.
	h100ISL = getenvInt32("DGDR_H100_ISL", 3000) // input sequence length
	h100OSL = getenvInt32("DGDR_H100_OSL", 300)  // output sequence length

	// Cluster hardware.
	h100TotalGPUs      = getenvInt32("DGDR_H100_TOTAL_GPUS", 32)       // total GPUs in the cluster
	h100NumGPUsPerNode = getenvInt32("DGDR_H100_NUM_GPUS_PER_NODE", 8) // GPUs per node

	// Planner budget (max GPUs the planner may allocate).
	h100MaxGPUBudget = getenvInt32("DGDR_H100_MAX_GPU_BUDGET", 32)
)

// disaggPlannerBase builds the DGDRLifecycleInput shared by the larger models
// (Qwen3-235B-A22B-FP8, Meta-Llama-3.1-70B) that run disagg with the planner and
// need 256Gi of shared memory on the prefill/decode workers. Only the resource
// name prefix, model, and DGD override name vary between these models, so they
// are parameterized here to keep the per-model specs DRY.
func disaggPlannerBase(
	namePrefix, model, dgdName string,
	backend v1beta1.BackendType,
	suffix string,
) DGDRLifecycleInput {
	return DGDRLifecycleInput{
		Name:           uniqueName(namePrefix + "-" + suffix),
		Model:          model,
		Backend:        backend,
		SearchStrategy: v1beta1.SearchStrategyRapid,
		AutoApply:      ptr.To(true),
		SLA: &v1beta1.SLASpec{
			TTFT: ptr.To(h100TTFTMillis),
			ITL:  ptr.To(h100ITLMillis),
		},
		Workload: &v1beta1.WorkloadSpec{
			ISL: ptr.To(h100ISL),
			OSL: ptr.To(h100OSL),
		},
		Hardware: &v1beta1.HardwareSpec{
			GPUSKU:         v1beta1.GPUSKUTypeH100SXM,
			VRAMMB:         ptr.To(h100VRAMMB),
			NumGPUsPerNode: ptr.To(h100NumGPUsPerNode),
			TotalGPUs:      ptr.To(h100TotalGPUs),
		},
		Features: &v1beta1.FeaturesSpec{
			Planner: plannerRawExtension(map[string]interface{}{
				"mode":                      "disagg",
				"enable_throughput_scaling": true,
				"enable_load_scaling":       true,
				"max_gpu_budget":            h100MaxGPUBudget,
			}),
		},
		Overrides: &v1beta1.OverridesSpec{
			DGD: dgdOverrideRawExtension(map[string]interface{}{
				"apiVersion": "nvidia.com/v1alpha1",
				"kind":       "DynamoGraphDeployment",
				"metadata":   map[string]interface{}{"name": dgdName},
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
		VerifyInference: true,
	}
}

// DGDR Support Matrix on H100 SKU exercises the full DGDR lifecycle (create -> profile ->
// DGD generation -> DGD readiness) across a curated
var _ = Describe("DGDR Support Matrix on H100 SKU", Label("gpu_0", "nightly", "e2e", "integration", "k8s"), func() {
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
						TTFT: ptr.To(h100TTFTMillis),
						ITL:  ptr.To(h100ITLMillis),
					},
					Workload: &v1beta1.WorkloadSpec{
						ISL: ptr.To(h100ISL),
						OSL: ptr.To(h100OSL),
					},
					Features: &v1beta1.FeaturesSpec{
						Planner: plannerRawExtension(map[string]interface{}{
							"mode":                      "disagg",
							"enable_throughput_scaling": true,
							"enable_load_scaling":       true,
							"max_gpu_budget":            h100MaxGPUBudget,
						}),
					},
					Hardware: &v1beta1.HardwareSpec{
						GPUSKU:         v1beta1.GPUSKUTypeH100SXM,
						VRAMMB:         ptr.To(h100VRAMMB),
						NumGPUsPerNode: ptr.To(h100NumGPUsPerNode),
						TotalGPUs:      ptr.To(h100TotalGPUs),
					},
					ExpectDGDReady:  true,
					VerifyConfigMap: true,
					VerifyInference: true,
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
				return disaggPlannerBase("qwen235b", "Qwen/Qwen3-235B-A22B-FP8", "q235", backend, suffix)
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
							TTFT: ptr.To(h100TTFTMillis),
							ITL:  ptr.To(h100ITLMillis),
						},
						Workload: &v1beta1.WorkloadSpec{
							ISL: ptr.To(h100ISL),
							OSL: ptr.To(h100OSL),
						},
						Hardware: &v1beta1.HardwareSpec{
							GPUSKU:         v1beta1.GPUSKUTypeH100SXM,
							VRAMMB:         ptr.To(h100VRAMMB),
							NumGPUsPerNode: ptr.To(h100NumGPUsPerNode),
							TotalGPUs:      ptr.To(h100TotalGPUs),
						},
						Features: &v1beta1.FeaturesSpec{
							Planner: plannerRawExtension(map[string]interface{}{
								"mode":                      "disagg",
								"enable_throughput_scaling": true,
								"enable_load_scaling":       true,
								"max_gpu_budget":            h100MaxGPUBudget,
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
						VerifyInference: true,
					}
				})
			})
		})

		// ---- Meta-Llama-3.1-70B (all backends) ----
		Context("should complete full lifecycle of Meta-Llama-3.1-70B", func() {
			llama31Base := func(backend v1beta1.BackendType, suffix string) DGDRLifecycleInput {
				return disaggPlannerBase("llama31-70b", "meta-llama/Meta-Llama-3.1-70B", "llama31-70b", backend, suffix)
			}

			DescribeTable("Meta-Llama-3.1-70B on H100 with planner",
				func(backend v1beta1.BackendType) {
					By("Running DGDR lifecycle with Meta-Llama-3.1-70B + " + string(backend) + " + rapid + planner")
					input := llama31Base(backend, string(backend))
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
	})

	// -----------------------------------------------------------------------
	// Support matrix — thorough profiling (not in AIC matrix)
	// Models that need thorough search to explore the config space.
	// All targeting the H100 test cluster.
	// -----------------------------------------------------------------------

	// Models we can test include Qwen3-VL-30B-FP8, Llama-3.3-70B-FP8, and Nemotron-3-Super-120B-FP8.
})
