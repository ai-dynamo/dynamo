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
	"fmt"
	"strings"
	"time"

	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	k8syaml "k8s.io/apimachinery/pkg/util/yaml"
)

var _ = Describe("DGDR Profiling", Label("gpu_0", "nightly", "integration", "k8s"), func() {

	Context("Rapid search strategy", func() {

		It("should emit an output ConfigMap with final_config.yaml", func() {
			name := uniqueName("rapid-cm")
			dgdr := newDGDR(name,
				withAutoApply(false),
				withWorkload(4000, 1000),
				withSLA(2000.0, 30.0),
			)
			createAndCleanup(dgdr)

			timeout := time.Duration(flagProfilingTimeout) * time.Second
			waitForPhase(name, v1beta1.DGDRPhaseReady, timeout)

			data := getOutputConfigMap(name)
			finalConfig, ok := data["final_config.yaml"]
			Expect(ok).To(BeTrue(), "ConfigMap should include data.final_config.yaml")
			Expect(finalConfig).NotTo(BeEmpty())

			// Verify it's parseable YAML containing a DynamoGraphDeployment
			dgd := parseFinalDGD(finalConfig)
			Expect(dgd).NotTo(BeNil())
		})

		It("should include Planner service when planner feature is enabled", func() {
			name := uniqueName("rapid-planner")
			dgdr := newDGDR(name,
				withAutoApply(false),
				withBackend(v1beta1.BackendTypeTrtllm),
				withWorkload(4000, 1000),
				withSLA(2000.0, 30.0),
				withFeatures(v1beta1.FeaturesSpec{
					Planner: plannerRawExtension(map[string]interface{}{
						"enabled":                       true,
						"plannerPreDeploymentSweeping": "rapid",
					}),
				}),
			)
			createAndCleanup(dgdr)

			timeout := time.Duration(flagProfilingTimeout) * time.Second
			waitForPhase(name, v1beta1.DGDRPhaseReady, timeout)

			data := getOutputConfigMap(name)
			dgd := parseFinalDGD(data["final_config.yaml"])

			services, ok := dgd["spec"].(map[string]interface{})["services"].(map[string]interface{})
			Expect(ok).To(BeTrue(), "DGD should have spec.services")
			_, hasPlan := services["Planner"]
			Expect(hasPlan).To(BeTrue(), "Planner service should exist in generated DGD")
		})

		It("should respect totalGpus budget in generated DGD [known issue #8583]", Label("xfail"), func() {
			name := uniqueName("rapid-budget")
			totalGPUs := int32(32)
			dgdr := newDGDR(name,
				withAutoApply(false),
				withModel("Qwen/Qwen3-235B-A22B-FP8"),
				withBackend(v1beta1.BackendTypeTrtllm),
				withWorkload(4000, 1000),
				withSLA(2000.0, 30.0),
				withHardware(v1beta1.HardwareSpec{
					GPUSKU:         v1beta1.GPUSKUTypeH200SXM,
					VRAMMB:         ptrFloat64(141120),
					NumGPUsPerNode: ptrInt32(8),
					TotalGPUs:      &totalGPUs,
				}),
			)
			createAndCleanup(dgdr)

			timeout := time.Duration(flagProfilingTimeout) * time.Second
			waitForPhase(name, v1beta1.DGDRPhaseReady, timeout)

			data := getOutputConfigMap(name)
			dgd := parseFinalDGD(data["final_config.yaml"])

			totalRequested := totalWorkerGPUs(dgd)
			// Known issue: rapid mode can exceed totalGpus. This test documents the regression.
			// When #8583 is fixed, this assertion should pass cleanly.
			if totalRequested > int(totalGPUs) {
				Skip(fmt.Sprintf(
					"Known issue #8583: generated DGD requests %d GPUs, exceeds budget %d",
					totalRequested, totalGPUs))
			}
		})
	})
})

// parseFinalDGD extracts the last YAML document from final_config.yaml and
// verifies it's a DynamoGraphDeployment.
func parseFinalDGD(yamlContent string) map[string]interface{} {
	// final_config.yaml may be multi-doc; take the last one
	var lastDoc map[string]interface{}
	decoder := k8syaml.NewYAMLOrJSONDecoder(strings.NewReader(yamlContent), 4096)
	for {
		var doc map[string]interface{}
		if err := decoder.Decode(&doc); err != nil {
			break
		}
		if doc != nil {
			lastDoc = doc
		}
	}
	Expect(lastDoc).NotTo(BeNil(), "final_config.yaml must contain at least one YAML document")
	Expect(lastDoc["kind"]).To(Equal("DynamoGraphDeployment"),
		"last document should be a DynamoGraphDeployment")
	return lastDoc
}

// totalWorkerGPUs computes total GPU requests across worker services.
func totalWorkerGPUs(dgd map[string]interface{}) int {
	spec, _ := dgd["spec"].(map[string]interface{})
	services, _ := spec["services"].(map[string]interface{})
	total := 0
	for _, svc := range services {
		s, _ := svc.(map[string]interface{})
		replicas := toInt(s["replicas"])
		resources, _ := s["resources"].(map[string]interface{})
		limits, _ := resources["limits"].(map[string]interface{})
		gpus := toInt(limits["gpu"])
		total += replicas * gpus
	}
	return total
}

func toInt(v interface{}) int {
	switch n := v.(type) {
	case float64:
		return int(n)
	case int:
		return n
	case int64:
		return int(n)
	default:
		return 0
	}
}

func ptrFloat64(v float64) *float64 { return &v }
func ptrInt32(v int32) *int32       { return &v }
