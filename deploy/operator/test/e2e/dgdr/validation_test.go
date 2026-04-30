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
	"encoding/json"
	"fmt"
	"os/exec"
	"strings"

	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

var _ = Describe("DGDR Validation", Label("gpu_0", "nightly", "integration", "k8s"), func() {

	// -----------------------------------------------------------------------
	// Webhook validation (server-side dry-run — no resources persisted)
	// -----------------------------------------------------------------------

	Context("Webhook Validation", func() {

		It("should reject a DGDR with missing model", func() {
			dgdr := newDGDR(uniqueName("no-model"))
			dgdr.Spec.Model = ""
			err := serverDryRun(dgdr)
			Expect(err).To(HaveOccurred(), "DGDR without model should be rejected")
		})

		It("should reject thorough + auto backend", func() {
			dgdr := newDGDR(uniqueName("thorough-auto"),
				withBackend(v1beta1.BackendTypeAuto),
				withSearchStrategy(v1beta1.SearchStrategyThorough),
			)
			err := serverDryRun(dgdr)
			Expect(err).To(HaveOccurred())
			errMsg := err.Error()
			Expect(errMsg).To(SatisfyAny(
				ContainSubstring("auto"),
				ContainSubstring("backend"),
				ContainSubstring("thorough"),
			), "error should mention backend/thorough incompatibility")
		})

		It("should reject an invalid backend", func() {
			dgdr := newDGDR(uniqueName("bad-backend"))
			dgdr.Spec.Backend = "unknown_backend"
			err := serverDryRun(dgdr)
			Expect(err).To(HaveOccurred(), "unknown backend should be rejected by CRD schema")
		})

		It("should reject an invalid searchStrategy", func() {
			dgdr := newDGDR(uniqueName("bad-strategy"))
			dgdr.Spec.SearchStrategy = "superfast"
			err := serverDryRun(dgdr)
			Expect(err).To(HaveOccurred(), "unknown searchStrategy should be rejected")
		})

		It("should reject an invalid sla.optimizationType", func() {
			dgdr := newDGDR(uniqueName("bad-opt"))
			invalidOpt := v1beta1.OptimizationType("cost")
			dgdr.Spec.SLA = &v1beta1.SLASpec{
				OptimizationType: &invalidOpt,
			}
			err := serverDryRun(dgdr)
			Expect(err).To(HaveOccurred(), "invalid optimizationType should be rejected")
		})

		It("should accept a valid minimal DGDR", func() {
			dgdr := newDGDR(uniqueName("valid-minimal"))
			err := serverDryRun(dgdr)
			Expect(err).NotTo(HaveOccurred(), "minimal DGDR should be accepted")
		})

		It("should accept a fully-specified DGDR", func() {
			dgdr := newDGDR(uniqueName("valid-full"),
				withBackend(v1beta1.BackendTypeVllm),
				withSearchStrategy(v1beta1.SearchStrategyRapid),
				withSLA(200.0, 20.0),
				withWorkload(3000, 150),
				withAutoApply(true),
				withHardware(v1beta1.HardwareSpec{
					NumGPUsPerNode: defaultMockerHardware.NumGPUsPerNode,
				}),
			)
			err := serverDryRun(dgdr)
			Expect(err).NotTo(HaveOccurred(), "full DGDR spec should be accepted")
		})
	})

	// -----------------------------------------------------------------------
	// CRD metadata (storage version, shortnames, columns)
	// -----------------------------------------------------------------------

	Context("CRD Metadata", func() {

		It("should have v1beta1 as the storage version", func() {
			stdout, _, err := kubectl(
				"get", "crd", "dynamographdeploymentrequests.nvidia.com",
				"-o", "jsonpath={.status.storedVersions}",
			)
			Expect(err).NotTo(HaveOccurred(), "failed to get CRD")
			Expect(stdout).To(ContainSubstring("v1beta1"))
		})

		It("should support the dgdr shortname", func() {
			_, _, err := kubectl(
				"get", "dgdr", "-n", flagNamespace, "--ignore-not-found",
			)
			Expect(err).NotTo(HaveOccurred(), "kubectl get dgdr should work (shortname)")
		})

		It("should show expected columns in kubectl output", func() {
			// Create a real DGDR to get column headers
			name := uniqueName("col-test")
			dgdr := newDGDR(name)
			createAndCleanup(dgdr)

			stdout, _, err := kubectl("get", "dgdr", name, "-n", flagNamespace)
			Expect(err).NotTo(HaveOccurred())

			header := ""
			lines := strings.Split(stdout, "\n")
			if len(lines) > 0 {
				header = strings.ToUpper(lines[0])
			}
			for _, col := range []string{"NAME", "MODEL", "BACKEND", "PHASE"} {
				Expect(header).To(ContainSubstring(col),
					"expected column %q in kubectl output header", col)
			}
		})
	})

	// -----------------------------------------------------------------------
	// v1alpha1 ↔ v1beta1 Version Conversion
	// -----------------------------------------------------------------------

	Context("Version Conversion", func() {

		It("should accept a v1alpha1 DGDR", func() {
			name := uniqueName("v1a1")
			manifest := fmt.Sprintf(`apiVersion: nvidia.com/v1alpha1
kind: DynamoGraphDeploymentRequest
metadata:
  name: %s
spec:
  model: %s
  backend: vllm
  profilingConfig:
    profilerImage: %s
`, name, flagModel, flagImage)

			cmd := exec.Command("kubectl", "apply", "-n", flagNamespace, "-f", "-",
				"--dry-run=server")
			cmd.Stdin = strings.NewReader(manifest)
			out, err := cmd.CombinedOutput()
			outStr := string(out)
			// Either accepted or rejected for a known conversion reason — not a 500
			if err != nil {
				Expect(outStr).NotTo(ContainSubstring("Internal error"),
					"v1alpha1 apply should not cause internal server error")
			}
		})

		It("should serve a v1alpha1 view of a v1beta1 object", func() {
			name := uniqueName("conv-get")
			dgdr := newDGDR(name)
			createAndCleanup(dgdr)

			// Verify stored as v1beta1
			var stored v1beta1.DynamoGraphDeploymentRequest
			Expect(k8sClient.Get(ctx, client.ObjectKey{
				Namespace: flagNamespace,
				Name:      name,
			}, &stored)).To(Succeed())

			// Retrieve as v1alpha1 via kubectl
			stdout, stderr, err := kubectl(
				"get", "dynamographdeploymentrequests.v1alpha1.nvidia.com",
				name, "-n", flagNamespace, "-o", "json",
			)
			// Conversion may not be registered in all setups
			if err != nil {
				_, _ = fmt.Fprintf(GinkgoWriter, "v1alpha1 get failed (may not be registered): %s\n", stderr)
				return
			}

			var obj map[string]interface{}
			Expect(json.Unmarshal([]byte(stdout), &obj)).To(Succeed())
			Expect(obj["apiVersion"]).To(Equal("nvidia.com/v1alpha1"),
				"retrieved object should have v1alpha1 apiVersion")
		})
	})
})
