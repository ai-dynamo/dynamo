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
	"context"
	"time"

	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"k8s.io/utils/ptr"
)

// DGDRLifecycleInput specifies the input for a full DGDR lifecycle test.
// It configures how the DGDR is created and what verification steps are performed.
type DGDRLifecycleInput struct {
	// Name is the unique name for this DGDR resource.
	Name string

	// DGDR spec options — zero values fall back to flag defaults.
	Model          string
	Backend        v1beta1.BackendType
	SearchStrategy v1beta1.SearchStrategy
	AutoApply      *bool
	SLA            *v1beta1.SLASpec
	Workload       *v1beta1.WorkloadSpec
	Hardware       *v1beta1.HardwareSpec
	Features       *v1beta1.FeaturesSpec

	// Verification options
	ExpectDGDReady  bool                          // verify DGD reaches state=successful (skipped in mocker mode)
	VerifyServices  map[string]ServiceExpectation // per-service expectations on the DGD (optional)
	VerifyConfigMap bool                          // check dgdr-output-<name> ConfigMap exists and contains a DGD
}

// DGDRLifecycleSpec implements a test that exercises the full DGDR lifecycle:
// creation -> profiling -> DGD generation -> (optional) DGD readiness verification.
//
// Modeled after the CAAPH HelmInstallSpec pattern: a single spec function driven
// by a parameterized input struct, so different DGDR scenarios are just different
// DGDRLifecycleInput values.
func DGDRLifecycleSpec(ctx context.Context, inputGetter func() DGDRLifecycleInput) {
	var (
		specName = "dgdr-lifecycle"
		input    DGDRLifecycleInput
	)

	input = inputGetter()
	Expect(input.Name).NotTo(BeEmpty(), "Invalid argument. input.Name can't be empty when calling %s spec", specName)

	// Build option functions from input
	var opts []func(*v1beta1.DynamoGraphDeploymentRequest)

	if input.AutoApply != nil {
		opts = append(opts, withAutoApply(*input.AutoApply))
	}
	if input.Backend != "" {
		opts = append(opts, withBackend(input.Backend))
	}
	if input.SearchStrategy != "" {
		opts = append(opts, withSearchStrategy(input.SearchStrategy))
	}
	if input.Model != "" {
		opts = append(opts, withModel(input.Model))
	}
	if input.SLA != nil {
		opts = append(opts, func(d *v1beta1.DynamoGraphDeploymentRequest) {
			d.Spec.SLA = input.SLA
		})
	}
	if input.Workload != nil {
		opts = append(opts, func(d *v1beta1.DynamoGraphDeploymentRequest) {
			d.Spec.Workload = input.Workload
		})
	}
	if input.Hardware != nil {
		opts = append(opts, withHardware(*input.Hardware))
	}
	if input.Features != nil {
		opts = append(opts, withFeatures(*input.Features))
	}

	// Step 1: Create DGDR
	By("Creating DGDR " + input.Name)
	dgdr := newDGDR(input.Name, opts...)
	createAndCleanup(dgdr)

	// Resolve autoApply (default true per CRD)
	autoApply := input.AutoApply == nil || *input.AutoApply

	// Step 2: Wait for profiling to complete (DGDR reaches Ready)
	By("Waiting for DGDR to reach Ready phase (profiling complete)")
	profilingTimeout := time.Duration(flagProfilingTimeout) * time.Second
	dgdrResult := waitForPhase(input.Name, v1beta1.DGDRPhaseReady, profilingTimeout)

	Expect(dgdrResult.Status.Phase).To(Equal(v1beta1.DGDRPhaseReady))
	Expect(dgdrResult.Status.ProfilingJobName).NotTo(BeEmpty(),
		"profilingJobName should be set after profiling completes")

	// Step 3: Verify ConfigMap output
	if input.VerifyConfigMap {
		By("Verifying output ConfigMap contains a valid DGD manifest")
		data := getOutputConfigMap(input.Name)
		finalConfig, ok := data["final_config.yaml"]
		Expect(ok).To(BeTrue(), "ConfigMap should include data.final_config.yaml")
		Expect(finalConfig).NotTo(BeEmpty())

		dgdDoc := parseFinalDGD(finalConfig)
		Expect(dgdDoc).NotTo(BeNil())

		// If services are specified, verify they appear in the generated DGD spec
		if len(input.VerifyServices) > 0 {
			services, ok := dgdDoc["spec"].(map[string]interface{})["services"].(map[string]interface{})
			Expect(ok).To(BeTrue(), "DGD should have spec.services")
			for svcName := range input.VerifyServices {
				_, hasSvc := services[svcName]
				Expect(hasSvc).To(BeTrue(), "service %q should exist in generated DGD spec", svcName)
			}
		}
	}

	// Step 4: If autoApply, wait for deployment
	if !autoApply {
		By("autoApply=false: skipping deployment verification")
		return
	}

	By("Waiting for DGDR to reach Deployed phase")
	deployTimeout := time.Duration(flagProfilingTimeout+flagDeployTimeout) * time.Second
	dgdrResult = waitForPhase(input.Name, v1beta1.DGDRPhaseDeployed, deployTimeout)

	Expect(dgdrResult.Status.Phase).To(Equal(v1beta1.DGDRPhaseDeployed))
	Expect(dgdrResult.Status.DGDName).NotTo(BeEmpty(),
		"dgdName should be set after deployment")

	// Step 5: Verify DGD reaches successful state
	if !input.ExpectDGDReady {
		return
	}

	if useMocker() {
		By("Skipping DGD pod verification in mocker mode")
		return
	}

	By("Waiting for DGD to reach successful state")
	dgdTimeout := time.Duration(flagDeployTimeout) * time.Second
	dgd := waitForDGDSuccessful(dgdrResult.Status.DGDName, dgdTimeout)
	Expect(dgd).NotTo(BeNil())

	// Step 6: Verify individual DGD services
	if len(input.VerifyServices) > 0 {
		By("Verifying DGD service replica status")
		verifyDGDServices(dgdrResult.Status.DGDName, input.VerifyServices)
	}
}

// defaultAutoApply returns the effective autoApply value, defaulting to true.
func defaultAutoApply(v *bool) bool {
	if v == nil {
		return true
	}
	return *v
}

// Convenience for creating common inputs.

// rapidLifecycleInput returns a DGDRLifecycleInput for a rapid profiling + deploy scenario.
func rapidLifecycleInput(name string, backend v1beta1.BackendType) DGDRLifecycleInput {
	return DGDRLifecycleInput{
		Name:            name,
		Backend:         backend,
		SearchStrategy:  v1beta1.SearchStrategyRapid,
		AutoApply:       ptr.To(true),
		ExpectDGDReady:  true,
		VerifyConfigMap: true,
	}
}
