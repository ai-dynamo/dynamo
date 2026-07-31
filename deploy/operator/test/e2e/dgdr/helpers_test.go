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
	"hash/fnv"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"time"

	v1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	k8sruntime "k8s.io/apimachinery/pkg/runtime"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

// getenvFloat64 returns the float64 value of the named environment variable,
// or def when the variable is unset, empty, or not a valid float. A parse
// failure logs a warning to GinkgoWriter and falls back to def so a typo in an
// env override never silently passes a zero value into a test.
func getenvFloat64(name string, def float64) float64 {
	v := os.Getenv(name)
	if v == "" {
		return def
	}
	parsed, err := strconv.ParseFloat(v, 64)
	if err != nil {
		_, _ = fmt.Fprintf(GinkgoWriter,
			"WARNING: %s=%q is not a valid float64, using default %v\n", name, v, def)
		return def
	}
	return parsed
}

// getenvInt32 returns the int32 value of the named environment variable, or def
// when the variable is unset, empty, or not a valid int32. See getenvFloat64
// for the parse-failure behaviour.
func getenvInt32(name string, def int32) int32 {
	v := os.Getenv(name)
	if v == "" {
		return def
	}
	parsed, err := strconv.ParseInt(v, 10, 32)
	if err != nil {
		_, _ = fmt.Fprintf(GinkgoWriter,
			"WARNING: %s=%q is not a valid int32, using default %d\n", name, v, def)
		return def
	}
	return int32(parsed)
}

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
	} else {
		// Real-GPU mode: apply CLI overrides for PVC, totalGpus, HF token.
		injectRecipeOverrides(dgdr)
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

func withOverrides(o v1beta1.OverridesSpec) func(*v1beta1.DynamoGraphDeploymentRequest) {
	return func(d *v1beta1.DynamoGraphDeploymentRequest) {
		if d.Spec.Overrides == nil {
			d.Spec.Overrides = &o
		} else {
			// Merge: only set fields that the caller provided, so
			// injectRecipeOverrides can still fill in ProfilingJob.
			if o.DGD != nil {
				d.Spec.Overrides.DGD = o.DGD
			}
			if o.ProfilingJob != nil {
				d.Spec.Overrides.ProfilingJob = o.ProfilingJob
			}
		}
	}
}

// dgdOverrideRawExtension builds a RawExtension containing a partial DGD manifest
// suitable for OverridesSpec.DGD. The map should follow the DGD spec structure,
// e.g. map[string]interface{}{"spec": map[string]interface{}{"services": ...}}.
func dgdOverrideRawExtension(m map[string]interface{}) *k8sruntime.RawExtension {
	raw, err := json.Marshal(m)
	ExpectWithOffset(1, err).NotTo(HaveOccurred(), "failed to marshal DGD override")
	return &k8sruntime.RawExtension{Raw: raw}
}

// injectRecipeOverrides applies CLI-provided real-GPU overrides to a DGDR:
// PVC model cache, totalGpus, and HF token secret env injection on the profiling job.
// Existing user-set values on the DGDR are preserved.
func injectRecipeOverrides(d *v1beta1.DynamoGraphDeploymentRequest) {
	if flagPVCName != "" && d.Spec.ModelCache == nil {
		d.Spec.ModelCache = &v1beta1.ModelCacheSpec{
			PVCName:      flagPVCName,
			PVCModelPath: flagPVCModelPath,
			PVCMountPath: flagPVCMountPath,
		}
	}
	if flagTotalGPUs > 0 {
		if d.Spec.Hardware == nil {
			d.Spec.Hardware = &v1beta1.HardwareSpec{}
		}
		if d.Spec.Hardware.TotalGPUs == nil {
			d.Spec.Hardware.TotalGPUs = ptr.To(int32(flagTotalGPUs))
		}
	}
	if flagHFTokenSecret != "" {
		if d.Spec.Overrides == nil {
			d.Spec.Overrides = &v1beta1.OverridesSpec{}
		}
		if d.Spec.Overrides.ProfilingJob == nil {
			d.Spec.Overrides.ProfilingJob = &batchv1.JobSpec{}
		}
		d.Spec.Overrides.ProfilingJob.Template.Spec.Containers = []corev1.Container{{
			Name: "profiler",
			Env: []corev1.EnvVar{{
				Name: "HF_TOKEN",
				ValueFrom: &corev1.EnvVarSource{
					SecretKeyRef: &corev1.SecretKeySelector{
						LocalObjectReference: corev1.LocalObjectReference{Name: flagHFTokenSecret},
						Key:                  "HF_TOKEN",
					},
				},
			}},
		}}
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
// If --dgdr-name-prefix is set on the CLI, the prefix is combined with
// a 6-hex-char FNV-1a hash of the per-test prefix so multiple tests in
// one suite invocation get distinct DGDR names while staying under the
// 45-char pod naming limit (see profile_sla.py). The hash avoids the
// collisions that a first-N-chars suffix would produce (e.g. "vllm-rapid"
// and "sglang-rapid" both collapsing to "rapi").
func uniqueName(prefix string) string {
	if flagNamePrefix != "" {
		h := fnv.New32a()
		_, _ = h.Write([]byte(prefix))
		short := fmt.Sprintf("%08x", h.Sum32())[:6]
		return fmt.Sprintf("%s-%s", flagNamePrefix, short)
	}
	return fmt.Sprintf("dgdr-test-%s-%d", prefix, time.Now().UnixMilli()%100000)
}

// createAndCleanup creates a DGDR and registers it for cleanup via DeferCleanup.
// Cleanup deletes both the DGDR and its child DGD, then waits for all associated
// pods to terminate so the next test starts with a clean slate (no GPU contention).
func createAndCleanup(dgdr *v1beta1.DynamoGraphDeploymentRequest) {
	Expect(k8sClient.Create(ctx, dgdr)).To(Succeed(), "failed to create DGDR %s", dgdr.Name)
	DeferCleanup(func() {
		// Fetch the DGDR to discover its child DGD name before deleting.
		var current v1beta1.DynamoGraphDeploymentRequest
		if err := k8sClient.Get(ctx, client.ObjectKey{
			Namespace: dgdr.Namespace,
			Name:      dgdr.Name,
		}, &current); err == nil && current.Status.DGDName != "" {
			dgdName := current.Status.DGDName

			// Delete the DGD (it is not owned by the DGDR, so it won't be
			// garbage-collected automatically).
			dgd := &v1alpha1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:      dgdName,
					Namespace: dgdr.Namespace,
				},
			}
			_ = k8sClient.Delete(ctx, dgd)

			// Wait for the DGD object to be fully removed.
			Eventually(func() bool {
				err := k8sClient.Get(ctx, client.ObjectKey{
					Namespace: dgdr.Namespace,
					Name:      dgdName,
				}, &v1alpha1.DynamoGraphDeployment{})
				return apierrors.IsNotFound(err)
			}, 5*time.Minute, 5*time.Second).Should(BeTrue(),
				"DGD %s should be fully deleted", dgdName)

			// Wait for all pods belonging to this DGD to terminate.
			dgdLabel := labels.SelectorFromSet(labels.Set{
				"nvidia.com/dynamo-graph-deployment-name": dgdName,
			})
			Eventually(func() int {
				var pods corev1.PodList
				if err := k8sClient.List(ctx, &pods,
					client.InNamespace(dgdr.Namespace),
					client.MatchingLabelsSelector{Selector: dgdLabel},
				); err != nil {
					return -1
				}
				return len(pods.Items)
			}, 5*time.Minute, 5*time.Second).Should(Equal(0),
				"all pods for DGD %s should be terminated", dgdName)
		}

		// Delete the DGDR itself.
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
func waitForPhaseAtLeast(
	name string, target v1beta1.DGDRPhase, timeout time.Duration,
) *v1beta1.DynamoGraphDeploymentRequest {
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

// verifyProfilingJobCompleted fetches the profiling Job by name and asserts it completed successfully.
func verifyProfilingJobCompleted(jobName string) {
	var job batchv1.Job
	Expect(k8sClient.Get(ctx, client.ObjectKey{
		Namespace: flagNamespace,
		Name:      jobName,
	}, &job)).To(Succeed(), "profiling job %q should exist", jobName)

	// A completed job has status.succeeded >= 1
	Expect(job.Status.Succeeded).To(BeNumerically(">=", int32(1)),
		"profiling job %q should have succeeded (status: succeeded=%d, failed=%d)",
		jobName, job.Status.Succeeded, job.Status.Failed)
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

// verifyInference runs a smoke test against the DGD's frontend service to confirm
// the model is actually serving. It checks v1/models and sends a short
// v1/chat/completions request.
func verifyInference(dgdName, model string) {
	frontendURL := fmt.Sprintf("http://%s-frontend.%s.svc.cluster.local:8000", dgdName, flagNamespace)

	// Build a short suffix for ephemeral pod names.
	suffix := dgdName
	if len(suffix) > 12 {
		suffix = suffix[:12]
	}

	// v1/models — verify the model is listed
	By("Checking v1/models endpoint")
	modelsOut, modelsErr, err := kubectl("run", "inference-models-"+suffix,
		"--rm", "-i", "--restart=Never",
		"-n", flagNamespace,
		"--image=curlimages/curl:latest",
		"--", "curl", "-sf", "--max-time", "10",
		frontendURL+"/v1/models",
	)
	Expect(err).NotTo(HaveOccurred(),
		"v1/models request failed: stdout=%s stderr=%s", modelsOut, modelsErr)
	Expect(modelsOut).To(ContainSubstring(model),
		"v1/models response should contain model %s, got: %s", model, modelsOut)

	// v1/chat/completions — verify the model can generate a response
	By("Sending a chat completion request")
	chatBody := fmt.Sprintf(
		`{"model":"%s","messages":[{"role":"user","content":"Say hello in one word."}],"max_tokens":16}`,
		model)
	chatOut, chatErr, err := kubectl("run", "inference-chat-"+suffix,
		"--rm", "-i", "--restart=Never",
		"-n", flagNamespace,
		"--image=curlimages/curl:latest",
		"--", "curl", "-sf", "--max-time", "60",
		"-H", "Content-Type: application/json",
		"-d", chatBody,
		frontendURL+"/v1/chat/completions",
	)
	Expect(err).NotTo(HaveOccurred(),
		"v1/chat/completions request failed: stdout=%s stderr=%s", chatOut, chatErr)
	Expect(chatOut).To(ContainSubstring("choices"),
		"chat response should contain 'choices', got: %s", chatOut)
}

// are Running with all containers ready. This catches cases where the DGD status
// reports Ready/Successful but the underlying pods are not actually healthy.
func verifyDGDPodsReady(dgdName string) {
	dgdLabel := labels.SelectorFromSet(labels.Set{
		"nvidia.com/dynamo-graph-deployment-name": dgdName,
	})
	var pods corev1.PodList
	Expect(k8sClient.List(ctx, &pods,
		client.InNamespace(flagNamespace),
		client.MatchingLabelsSelector{Selector: dgdLabel},
	)).To(Succeed(), "failed to list pods for DGD %s", dgdName)

	Expect(pods.Items).NotTo(BeEmpty(),
		"DGD %s should have at least one pod", dgdName)

	for _, pod := range pods.Items {
		// Skip completed pods (e.g. profiling jobs)
		if pod.Status.Phase == corev1.PodSucceeded {
			continue
		}
		By(fmt.Sprintf("Verifying pod %s is Running and Ready", pod.Name))
		Expect(pod.Status.Phase).To(Equal(corev1.PodRunning),
			"pod %s should be Running but is %s", pod.Name, pod.Status.Phase)

		for _, cs := range pod.Status.ContainerStatuses {
			Expect(cs.Ready).To(BeTrue(),
				"container %s in pod %s should be ready", cs.Name, pod.Name)
		}
	}
}
