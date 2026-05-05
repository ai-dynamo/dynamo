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
	"time"

	v1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	k8sruntime "k8s.io/apimachinery/pkg/runtime"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

// Default hardware for mocker mode (AIC simulation needs hardware metadata).
// Use H100_SXM because AIC has complete perf data for all backends (vllm, sglang, trtllm)
// on this GPU type. A100_SXM has an incomplete sglang 0.5.8 database (missing context_attention_perf.txt).
var defaultMockerHardware = v1beta1.HardwareSpec{
	GPUSKU:         v1beta1.GPUSKUTypeH100SXM,
	VRAMMB:         ptr.To(float64(81920)),
	NumGPUsPerNode: ptr.To(int32(8)),
	TotalGPUs:      ptr.To(int32(8)),
}

// newDGDR builds a v1beta1 DynamoGraphDeploymentRequest with sensible defaults.
// Options can override any field.
func newDGDR(name string, opts ...func(*v1beta1.DynamoGraphDeploymentRequest)) *v1beta1.DynamoGraphDeploymentRequest {
	dgdr := &v1beta1.DynamoGraphDeploymentRequest{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: flagNamespace,
			Labels: map[string]string{
				"test.dynamo/managed": "true",
			},
		},
		Spec: v1beta1.DynamoGraphDeploymentRequestSpec{
			Model:          flagModel,
			Backend:        v1beta1.BackendType(flagBackend),
			Image:          flagImage,
			SearchStrategy: v1beta1.SearchStrategyRapid,
		},
	}
	for _, o := range opts {
		o(dgdr)
	}
	// Inject mocker config when enabled
	if useMocker() {
		injectMockerConfig(dgdr)
	}
	return dgdr
}

// Option functions for newDGDR

func withAutoApply(v bool) func(*v1beta1.DynamoGraphDeploymentRequest) {
	return func(d *v1beta1.DynamoGraphDeploymentRequest) {
		d.Spec.AutoApply = ptr.To(v)
	}
}

func withBackend(b v1beta1.BackendType) func(*v1beta1.DynamoGraphDeploymentRequest) {
	return func(d *v1beta1.DynamoGraphDeploymentRequest) {
		d.Spec.Backend = b
	}
}

func withSearchStrategy(s v1beta1.SearchStrategy) func(*v1beta1.DynamoGraphDeploymentRequest) {
	return func(d *v1beta1.DynamoGraphDeploymentRequest) {
		d.Spec.SearchStrategy = s
	}
}

func withModel(m string) func(*v1beta1.DynamoGraphDeploymentRequest) {
	return func(d *v1beta1.DynamoGraphDeploymentRequest) {
		d.Spec.Model = m
	}
}

func withSLA(ttft, itl float64) func(*v1beta1.DynamoGraphDeploymentRequest) {
	return func(d *v1beta1.DynamoGraphDeploymentRequest) {
		d.Spec.SLA = &v1beta1.SLASpec{
			TTFT: ptr.To(ttft),
			ITL:  ptr.To(itl),
		}
	}
}

func withWorkload(isl, osl int32) func(*v1beta1.DynamoGraphDeploymentRequest) {
	return func(d *v1beta1.DynamoGraphDeploymentRequest) {
		d.Spec.Workload = &v1beta1.WorkloadSpec{
			ISL: ptr.To(isl),
			OSL: ptr.To(osl),
		}
	}
}

func withHardware(hw v1beta1.HardwareSpec) func(*v1beta1.DynamoGraphDeploymentRequest) {
	return func(d *v1beta1.DynamoGraphDeploymentRequest) {
		d.Spec.Hardware = &hw
	}
}

func withFeatures(f v1beta1.FeaturesSpec) func(*v1beta1.DynamoGraphDeploymentRequest) {
	return func(d *v1beta1.DynamoGraphDeploymentRequest) {
		d.Spec.Features = &f
	}
}

// injectMockerConfig mutates a DGDR for GPU-free testing.
func injectMockerConfig(d *v1beta1.DynamoGraphDeploymentRequest) {
	if d.Spec.Features == nil {
		d.Spec.Features = &v1beta1.FeaturesSpec{}
	}
	d.Spec.Features.Mocker = &v1beta1.MockerSpec{Enabled: true}

	if d.Spec.Hardware == nil {
		hw := defaultMockerHardware
		d.Spec.Hardware = &hw
	} else {
		// Fill missing fields from defaults
		if d.Spec.Hardware.GPUSKU == "" {
			d.Spec.Hardware.GPUSKU = defaultMockerHardware.GPUSKU
		}
		if d.Spec.Hardware.VRAMMB == nil {
			d.Spec.Hardware.VRAMMB = defaultMockerHardware.VRAMMB
		}
		if d.Spec.Hardware.NumGPUsPerNode == nil {
			d.Spec.Hardware.NumGPUsPerNode = defaultMockerHardware.NumGPUsPerNode
		}
		if d.Spec.Hardware.TotalGPUs == nil {
			d.Spec.Hardware.TotalGPUs = defaultMockerHardware.TotalGPUs
		}
	}
}

// uniqueName generates a K8s-safe test name with a timestamp suffix.
func uniqueName(prefix string) string {
	return fmt.Sprintf("dgdr-test-%s-%d", prefix, time.Now().UnixMilli()%100000)
}

// createAndCleanup creates a DGDR and registers it for cleanup via DeferCleanup.
func createAndCleanup(dgdr *v1beta1.DynamoGraphDeploymentRequest) {
	Expect(k8sClient.Create(ctx, dgdr)).To(Succeed(), "failed to create DGDR %s", dgdr.Name)
	DeferCleanup(func() {
		_ = k8sClient.Delete(ctx, dgdr)
	})
}

// serverDryRun attempts to create a DGDR with server-side dry-run and returns the error (if any).
func serverDryRun(dgdr *v1beta1.DynamoGraphDeploymentRequest) error {
	return k8sClient.Create(ctx, dgdr, client.DryRunAll)
}

// phaseOrder defines the lifecycle ordering for DGDR phases.
var phaseOrder = map[v1beta1.DGDRPhase]int{
	v1beta1.DGDRPhasePending:   0,
	v1beta1.DGDRPhaseProfiling: 1,
	v1beta1.DGDRPhaseReady:     2,
	v1beta1.DGDRPhaseDeploying: 3,
	v1beta1.DGDRPhaseDeployed:  4,
	v1beta1.DGDRPhaseFailed:    -1,
}

// waitForPhase polls until the DGDR reaches the target phase or times out.
// Fails immediately if the DGDR enters the Failed phase (unless that's the target).
func waitForPhase(name string, target v1beta1.DGDRPhase, timeout time.Duration) *v1beta1.DynamoGraphDeploymentRequest {
	var dgdr v1beta1.DynamoGraphDeploymentRequest
	Eventually(func(g Gomega) {
		g.Expect(k8sClient.Get(ctx, client.ObjectKey{
			Namespace: flagNamespace,
			Name:      name,
		}, &dgdr)).To(Succeed())
		if target != v1beta1.DGDRPhaseFailed && dgdr.Status.Phase == v1beta1.DGDRPhaseFailed {
			msg := "unknown"
			for _, c := range dgdr.Status.Conditions {
				if c.Message != "" {
					msg = c.Message
					break
				}
			}
			Fail(fmt.Sprintf("DGDR %s entered Failed phase while waiting for %s: %s", name, target, msg))
		}
		g.Expect(dgdr.Status.Phase).To(Equal(target),
			"DGDR %s phase is %s, waiting for %s", name, dgdr.Status.Phase, target)
	}, timeout, 5*time.Second).Should(Succeed())
	return &dgdr
}

// waitForPhaseAtLeast polls until the DGDR reaches the target phase or a later phase.
// This is useful when autoApply=true causes the operator to skip the Ready phase
// and transition directly to Deploying.
// Fails immediately if the DGDR enters the Failed phase.
func waitForPhaseAtLeast(name string, target v1beta1.DGDRPhase, timeout time.Duration) *v1beta1.DynamoGraphDeploymentRequest {
	var dgdr v1beta1.DynamoGraphDeploymentRequest
	targetOrder := phaseOrder[target]
	Eventually(func(g Gomega) {
		g.Expect(k8sClient.Get(ctx, client.ObjectKey{
			Namespace: flagNamespace,
			Name:      name,
		}, &dgdr)).To(Succeed())
		currentOrder, ok := phaseOrder[dgdr.Status.Phase]
		g.Expect(ok).To(BeTrue(), "unknown phase %q", dgdr.Status.Phase)
		if dgdr.Status.Phase == v1beta1.DGDRPhaseFailed {
			msg := "unknown"
			for _, c := range dgdr.Status.Conditions {
				if c.Message != "" {
					msg = c.Message
					break
				}
			}
			Fail(fmt.Sprintf("DGDR %s entered Failed phase while waiting for at least %s: %s", name, target, msg))
		}
		g.Expect(currentOrder).To(BeNumerically(">=", targetOrder),
			"DGDR %s phase is %s (order %d), waiting for at least %s (order %d)",
			name, dgdr.Status.Phase, currentOrder, target, targetOrder)
	}, timeout, 5*time.Second).Should(Succeed())
	return &dgdr
}

// kubectl runs a kubectl command and returns stdout/stderr.
func kubectl(args ...string) (string, string, error) {
	cmd := exec.Command("kubectl", args...)
	var stdout, stderr strings.Builder
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	err := cmd.Run()
	return stdout.String(), stderr.String(), err
}

// getOutputConfigMap fetches the dgdr-output-<name> ConfigMap and returns its data map.
func getOutputConfigMap(name string) map[string]string {
	cmName := fmt.Sprintf("dgdr-output-%s", name)
	stdout, _, err := kubectl("get", "configmap", cmName, "-n", flagNamespace, "-o", "json")
	Expect(err).NotTo(HaveOccurred(), "ConfigMap %s not found", cmName)

	var raw map[string]interface{}
	Expect(json.Unmarshal([]byte(stdout), &raw)).To(Succeed())

	data, ok := raw["data"].(map[string]interface{})
	Expect(ok).To(BeTrue(), "ConfigMap %s has no data field", cmName)

	result := make(map[string]string, len(data))
	for k, v := range data {
		result[k] = fmt.Sprint(v)
	}
	return result
}

// plannerRawExtension creates a raw JSON extension from a map (for Features.Planner).
func plannerRawExtension(m map[string]interface{}) *k8sruntime.RawExtension {
	raw, err := json.Marshal(m)
	Expect(err).NotTo(HaveOccurred())
	return &k8sruntime.RawExtension{Raw: raw}
}

// waitForDGDSuccessful polls until the DynamoGraphDeployment reaches state=successful or times out.
func waitForDGDSuccessful(name string, timeout time.Duration) *v1alpha1.DynamoGraphDeployment {
	var dgd v1alpha1.DynamoGraphDeployment
	Eventually(func(g Gomega) {
		g.Expect(k8sClient.Get(ctx, client.ObjectKey{
			Namespace: flagNamespace,
			Name:      name,
		}, &dgd)).To(Succeed())
		g.Expect(dgd.Status.State).To(Equal(v1alpha1.DGDStateSuccessful),
			"DGD %s state is %s, waiting for successful (conditions: %+v)",
			name, dgd.Status.State, dgd.Status.Conditions)
	}, timeout, 10*time.Second).Should(Succeed())
	return &dgd
}

// ServiceExpectation defines expected properties for a DGD service.
type ServiceExpectation struct {
	MinReplicas int32
	ExpectGPUs  bool // verify nvidia.com/gpu resource requests exist
}

// verifyDGDServices checks that each expected service exists in the DGD status
// and has the expected replica counts.
func verifyDGDServices(dgdName string, expectations map[string]ServiceExpectation) {
	var dgd v1alpha1.DynamoGraphDeployment
	Expect(k8sClient.Get(ctx, client.ObjectKey{
		Namespace: flagNamespace,
		Name:      dgdName,
	}, &dgd)).To(Succeed())

	for svcName, expected := range expectations {
		By(fmt.Sprintf("Verifying service %q in DGD %s", svcName, dgdName))
		svcStatus, ok := dgd.Status.Services[svcName]
		Expect(ok).To(BeTrue(), "service %q not found in DGD status.services", svcName)
		Expect(svcStatus.Replicas).To(BeNumerically(">=", expected.MinReplicas),
			"service %q replicas %d < expected minimum %d", svcName, svcStatus.Replicas, expected.MinReplicas)

		// Check readiness based on component kind
		switch svcStatus.ComponentKind {
		case v1alpha1.ComponentKindPodCliqueScalingGroup:
			if svcStatus.AvailableReplicas != nil {
				Expect(*svcStatus.AvailableReplicas).To(Equal(svcStatus.Replicas),
					"service %q availableReplicas should match replicas", svcName)
			}
		default:
			if svcStatus.ReadyReplicas != nil {
				Expect(*svcStatus.ReadyReplicas).To(Equal(svcStatus.Replicas),
					"service %q readyReplicas should match replicas", svcName)
			}
		}
	}
}
