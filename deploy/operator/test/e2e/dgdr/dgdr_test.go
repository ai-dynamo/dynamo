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
							"plannerPreDeploymentSweeping": "rapid",
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
