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
	"time"

	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

var _ = Describe("DGDR Lifecycle", Label("gpu_0", "nightly", "integration", "k8s"), func() {

	Context("Rapid profiling", func() {

		It("should reach Ready with autoApply=false", func() {
			name := uniqueName("lifecycle-ready")
			dgdr := newDGDR(name, withAutoApply(false))
			createAndCleanup(dgdr)

			timeout := time.Duration(flagProfilingTimeout) * time.Second
			result := waitForPhase(name, v1beta1.DGDRPhaseReady, timeout)

			Expect(result.Status.Phase).To(Equal(v1beta1.DGDRPhaseReady))
			Expect(result.Status.ProfilingJobName).NotTo(BeEmpty(),
				"profilingJobName should be set after profiling completes")

			// Verify the profiling job itself completed successfully
			verifyProfilingJobCompleted(result.Status.ProfilingJobName)
		})

		It("should reach Deployed with autoApply=true (non-mocker only)", func() {
			if useMocker() {
				Skip("In mocker mode, autoApply=true can race on generated DGD; " +
					"lifecycle deployment coverage is run with --dgdr-no-mocker")
			}

			name := uniqueName("lifecycle-deployed")
			dgdr := newDGDR(name, withAutoApply(true))
			createAndCleanup(dgdr)

			timeout := time.Duration(flagProfilingTimeout+flagDeployTimeout) * time.Second
			result := waitForPhase(name, v1beta1.DGDRPhaseDeployed, timeout)

			Expect(result.Status.Phase).To(Equal(v1beta1.DGDRPhaseDeployed))
			Expect(result.Status.DGDName).NotTo(BeEmpty(),
				"dgdName should be set after deployment")

			// Verify the DGD reaches successful state
			dgdTimeout := time.Duration(flagDeployTimeout) * time.Second
			dgd := waitForDGDSuccessful(result.Status.DGDName, dgdTimeout)
			Expect(dgd).NotTo(BeNil())
		})
	})
})
