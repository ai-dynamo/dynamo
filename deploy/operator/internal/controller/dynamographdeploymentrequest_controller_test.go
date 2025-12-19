/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

package controller

import (
	"context"
	"encoding/json"
	"time"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	commonController "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/tools/record"
	"sigs.k8s.io/controller-runtime/pkg/reconcile"
	"sigs.k8s.io/yaml"
)

const (
	defaultNamespace = "default"
)

// MockRBACManager implements RBACManager for testing
type MockRBACManager struct {
	EnsureServiceAccountWithRBACFunc func(ctx context.Context, targetNamespace, serviceAccountName, clusterRoleName string) error
}

func (m *MockRBACManager) EnsureServiceAccountWithRBAC(ctx context.Context, targetNamespace, serviceAccountName, clusterRoleName string) error {
	if m.EnsureServiceAccountWithRBACFunc != nil {
		return m.EnsureServiceAccountWithRBACFunc(ctx, targetNamespace, serviceAccountName, clusterRoleName)
	}
	return nil
}

// Helper function to create JSON config for tests
func createTestConfig(config map[string]interface{}) *apiextensionsv1.JSON {
	jsonBytes, err := json.Marshal(config)
	if err != nil {
		panic(err)
	}
	return &apiextensionsv1.JSON{Raw: jsonBytes}
}

var _ = Describe("DynamoGraphDeploymentRequest Controller", func() {
	const (
		timeout  = time.Second * 10
		interval = time.Millisecond * 250
	)

	var (
		reconciler *DynamoGraphDeploymentRequestReconciler
		recorder   *record.FakeRecorder
	)

	BeforeEach(func() {
		recorder = record.NewFakeRecorder(100)
		reconciler = &DynamoGraphDeploymentRequestReconciler{
			Client:   k8sClient,
			Recorder: recorder,
			Config: commonController.Config{
				RestrictedNamespace: "",
				RBAC: commonController.RBACConfig{
					DGDRProfilingClusterRoleName: "test-cluster-role",
				},
			},
			RBACManager: &MockRBACManager{},
		}
	})

	Context("When reconciling initial DGDR", func() {
		It("Should validate spec and transition to Pending", func() {
			ctx := context.Background()
			dgdrName := "test-dgdr-initial"
			namespace := defaultNamespace

			dgdr := &nvidiacomv1alpha1.DynamoGraphDeploymentRequest{
				ObjectMeta: metav1.ObjectMeta{
					Name:      dgdrName,
					Namespace: namespace,
				},
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentRequestSpec{
					Model:   "test-model",
					Backend: "vllm",
					ProfilingConfig: nvidiacomv1alpha1.ProfilingConfigSpec{
						ProfilerImage: "test-profiler:latest",
						Config: createTestConfig(map[string]interface{}{
							"engine": map[string]interface{}{
								"config": "/tmp/test-config.yaml",
							},
							"sla": map[string]interface{}{
								"ttft": 100.0,
								"itl":  1500.0,
								"isl":  3000,
								"osl":  5,
							},
							"hardware": map[string]interface{}{
								"min_num_gpus_per_engine": 1,
								"max_num_gpus_per_engine": 8,
							},
						}),
					},
				},
			}

			Expect(k8sClient.Create(ctx, dgdr)).Should(Succeed())
			defer func() { _ = k8sClient.Delete(ctx, dgdr) }()

			// First reconcile: Empty -> Pending
			_, err := reconciler.Reconcile(ctx, reconcile.Request{
				NamespacedName: types.NamespacedName{
					Name:      dgdrName,
					Namespace: namespace,
				},
			})
			Expect(err).NotTo(HaveOccurred())

			// Check status
			Eventually(func() string {
				var updated nvidiacomv1alpha1.DynamoGraphDeploymentRequest
				_ = k8sClient.Get(ctx, types.NamespacedName{Name: dgdrName, Namespace: namespace}, &updated)
				return updated.Status.State
			}, timeout, interval).Should(Equal(StatePending))

			// Verify observedGeneration is set
			var updated nvidiacomv1alpha1.DynamoGraphDeploymentRequest
			_ = k8sClient.Get(ctx, types.NamespacedName{Name: dgdrName, Namespace: namespace}, &updated)
			Expect(updated.Status.ObservedGeneration).Should(Equal(updated.Generation))
		})

		It("Should pass validation with minimal config", func() {
			ctx := context.Background()
			dgdrName := "test-dgdr-minimal"
			namespace := defaultNamespace

			dgdr := &nvidiacomv1alpha1.DynamoGraphDeploymentRequest{
				ObjectMeta: metav1.ObjectMeta{
					Name:      dgdrName,
					Namespace: namespace,
				},
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentRequestSpec{
					Model:   "test-model",
					Backend: "vllm",
					ProfilingConfig: nvidiacomv1alpha1.ProfilingConfigSpec{
						ProfilerImage: "test-profiler:latest",
						Config: createTestConfig(map[string]interface{}{
							"sla": map[string]interface{}{
								"ttft": 100.0,
								"itl":  1500.0,
							},
						}),
					},
				},
			}

			Expect(k8sClient.Create(ctx, dgdr)).Should(Succeed())
			defer func() { _ = k8sClient.Delete(ctx, dgdr) }()

			// Reconcile - should succeed with minimal config
			_, err := reconciler.Reconcile(ctx, reconcile.Request{
				NamespacedName: types.NamespacedName{
					Name:      dgdrName,
					Namespace: namespace,
				},
			})
			Expect(err).NotTo(HaveOccurred())

			// Check status transitions to Pending (not Failed)
			Eventually(func() string {
				var updated nvidiacomv1alpha1.DynamoGraphDeploymentRequest
				_ = k8sClient.Get(ctx, types.NamespacedName{Name: dgdrName, Namespace: namespace}, &updated)
				return updated.Status.State
			}, timeout, interval).Should(Equal(StatePending))
		})
	})

	Context("When creating profiling job", func() {
		It("Should create online profiling job", func() {
			ctx := context.Background()
			dgdrName := "test-dgdr-profiling-online"
			namespace := defaultNamespace

			// Create ConfigMap for DGD base config
			configMap := &corev1.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-config",
					Namespace: namespace,
				},
				Data: map[string]string{
					"disagg.yaml": "test: config",
				},
			}
			Expect(k8sClient.Create(ctx, configMap)).Should(Succeed())
			defer func() { _ = k8sClient.Delete(ctx, configMap) }()

			// Create ServiceAccount
			sa := &corev1.ServiceAccount{
				ObjectMeta: metav1.ObjectMeta{
					Name:      ServiceAccountProfilingJob,
					Namespace: namespace,
				},
			}
			Expect(k8sClient.Create(ctx, sa)).Should(Succeed())
			defer func() { _ = k8sClient.Delete(ctx, sa) }()

			dgdr := &nvidiacomv1alpha1.DynamoGraphDeploymentRequest{
				ObjectMeta: metav1.ObjectMeta{
					Name:      dgdrName,
					Namespace: namespace,
				},
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentRequestSpec{
					Model:   "test-model",
					Backend: "vllm",
					ProfilingConfig: nvidiacomv1alpha1.ProfilingConfigSpec{
						ProfilerImage: "test-profiler:latest",
						Config: createTestConfig(map[string]interface{}{
							"engine": map[string]interface{}{
								"profiler_image": "test-profiler:latest",
							},
							"sla": map[string]interface{}{
								"ttft": 100.0,
								"itl":  1500.0,
								"isl":  3000,
								"osl":  5,
							},
							"hardware": map[string]interface{}{
								"min_num_gpus_per_engine": 1,
								"max_num_gpus_per_engine": 8,
							},
						}),
						ConfigMapRef: &nvidiacomv1alpha1.ConfigMapKeySelector{
							Name: "test-config",
							Key:  "disagg.yaml",
						},
					},
				},
			}

			Expect(k8sClient.Create(ctx, dgdr)).Should(Succeed())
			defer func() { _ = k8sClient.Delete(ctx, dgdr) }()

			// Reconcile multiple times to move through states
			_, err := reconciler.Reconcile(ctx, reconcile.Request{
				NamespacedName: types.NamespacedName{Name: dgdrName, Namespace: namespace},
			})
			Expect(err).NotTo(HaveOccurred())

			// Second reconcile: Pending -> Profiling
			_, err = reconciler.Reconcile(ctx, reconcile.Request{
				NamespacedName: types.NamespacedName{Name: dgdrName, Namespace: namespace},
			})
			Expect(err).NotTo(HaveOccurred())

			// Verify profiling job was created
			Eventually(func() bool {
				jobName := getProfilingJobName(dgdr)
				job := &batchv1.Job{}
				err := k8sClient.Get(ctx, types.NamespacedName{Name: jobName, Namespace: namespace}, job)
				return err == nil
			}, timeout, interval).Should(BeTrue())

			// Verify job has correct labels
			jobName := getProfilingJobName(dgdr)
			job := &batchv1.Job{}
			_ = k8sClient.Get(ctx, types.NamespacedName{Name: jobName, Namespace: namespace}, job)
			Expect(job.Labels[LabelApp]).Should(Equal(LabelValueDynamoProfiler))
			Expect(job.Labels[LabelDGDR]).Should(Equal(dgdrName))

			// Verify job has profiler container
			Expect(job.Spec.Template.Spec.Containers).Should(HaveLen(2))
			Expect(job.Spec.Template.Spec.Containers[0].Name).Should(Equal(ContainerNameProfiler))
			Expect(job.Spec.Template.Spec.Containers[1].Name).Should(Equal(ContainerNameOutputCopier))

			// Verify emptyDir volume (not PVC)
			Expect(job.Spec.Template.Spec.Volumes).Should(ContainElement(
				corev1.Volume{
					Name: VolumeNameProfilingOutput,
					VolumeSource: corev1.VolumeSource{
						EmptyDir: &corev1.EmptyDirVolumeSource{},
					},
				},
			))

			// Clean up job
			_ = k8sClient.Delete(ctx, job)
		})

		It("Should create offline (AIC) profiling job", func() {
			ctx := context.Background()
			dgdrName := "test-dgdr-profiling-aic"
			namespace := defaultNamespace

			// Create ServiceAccount
			sa := &corev1.ServiceAccount{
				ObjectMeta: metav1.ObjectMeta{
					Name:      ServiceAccountProfilingJob,
					Namespace: namespace,
				},
			}
			Expect(k8sClient.Create(ctx, sa)).Should(Succeed())
			defer func() { _ = k8sClient.Delete(ctx, sa) }()

			dgdr := &nvidiacomv1alpha1.DynamoGraphDeploymentRequest{
				ObjectMeta: metav1.ObjectMeta{
					Name:      dgdrName,
					Namespace: namespace,
				},
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentRequestSpec{
					Model:   "test-model",
					Backend: "trtllm",
					ProfilingConfig: nvidiacomv1alpha1.ProfilingConfigSpec{
						ProfilerImage: "test-profiler:latest",
						Config: createTestConfig(map[string]interface{}{
							"engine": map[string]interface{}{
								"config":         "/tmp/test-config.yaml",
								"profiler_image": "test-profiler:latest",
							},
							"sla": map[string]interface{}{
								"ttft": 100.0,
								"itl":  1500.0,
								"isl":  3000,
								"osl":  5,
							},
							"hardware": map[string]interface{}{
								"min_num_gpus_per_engine": 1,
								"max_num_gpus_per_engine": 8,
							},
							"sweep": map[string]interface{}{
								"use_ai_configurator": true,
								"aic_system":          "h200_sxm",
								"aic_hf_id":           "Qwen/Qwen3-32B",
								"aic_backend_version": "0.20.0",
							},
						}),
					},
				},
			}

			Expect(k8sClient.Create(ctx, dgdr)).Should(Succeed())
			defer func() { _ = k8sClient.Delete(ctx, dgdr) }()

			// Reconcile
			_, err := reconciler.Reconcile(ctx, reconcile.Request{
				NamespacedName: types.NamespacedName{Name: dgdrName, Namespace: namespace},
			})
			Expect(err).NotTo(HaveOccurred())

			_, err = reconciler.Reconcile(ctx, reconcile.Request{
				NamespacedName: types.NamespacedName{Name: dgdrName, Namespace: namespace},
			})
			Expect(err).NotTo(HaveOccurred())

			// Verify job was created with AIC label
			Eventually(func() string {
				jobName := getProfilingJobName(dgdr)
				job := &batchv1.Job{}
				if err := k8sClient.Get(ctx, types.NamespacedName{Name: jobName, Namespace: namespace}, job); err != nil {
					return ""
				}
				return job.Labels[LabelApp]
			}, timeout, interval).Should(Equal(LabelValueAICProfiler))

			// Clean up
			jobName := getProfilingJobName(dgdr)
			job := &batchv1.Job{}
			if err := k8sClient.Get(ctx, types.NamespacedName{Name: jobName, Namespace: namespace}, job); err == nil {
				_ = k8sClient.Delete(ctx, job)
			}
		})
	})

	Context("When profiling completes", func() {
		It("Should generate DGD spec from ConfigMap", func() {
			ctx := context.Background()
			dgdrName := "test-dgdr-profiling-complete"
			namespace := defaultNamespace

			dgdr := &nvidiacomv1alpha1.DynamoGraphDeploymentRequest{
				ObjectMeta: metav1.ObjectMeta{
					Name:      dgdrName,
					Namespace: namespace,
				},
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentRequestSpec{
					Model:   "test-model",
					Backend: "vllm",
					ProfilingConfig: nvidiacomv1alpha1.ProfilingConfigSpec{
						ProfilerImage: "test-profiler:latest",
						Config: createTestConfig(map[string]interface{}{
							"engine": map[string]interface{}{
								"config": "/tmp/test-config.yaml",
							},
							"sla": map[string]interface{}{
								"ttft": 100.0,
								"itl":  1500.0,
								"isl":  3000,
								"osl":  5,
							},
						}),
					},
				},
			}

			Expect(k8sClient.Create(ctx, dgdr)).Should(Succeed())
			defer func() { _ = k8sClient.Delete(ctx, dgdr) }()

			// Update status to Profiling using Status subresource
			dgdr.Status.State = StateProfiling
			Expect(k8sClient.Status().Update(ctx, dgdr)).Should(Succeed())

			// Create completed profiling job
			jobName := getProfilingJobName(dgdr)
			job := &batchv1.Job{
				ObjectMeta: metav1.ObjectMeta{
					Name:      jobName,
					Namespace: namespace,
				},
				Spec: batchv1.JobSpec{
					Template: corev1.PodTemplateSpec{
						Spec: corev1.PodSpec{
							Containers: []corev1.Container{{
								Name:  "test",
								Image: "test",
							}},
							RestartPolicy: corev1.RestartPolicyNever,
						},
					},
				},
				Status: batchv1.JobStatus{
					Conditions: []batchv1.JobCondition{{
						Type:   batchv1.JobComplete,
						Status: corev1.ConditionTrue,
					}},
				},
			}
			Expect(k8sClient.Create(ctx, job)).Should(Succeed())
			defer func() { _ = k8sClient.Delete(ctx, job) }()

			// Update job status to completed using Status subresource
			job.Status.Conditions = []batchv1.JobCondition{{
				Type:   batchv1.JobComplete,
				Status: corev1.ConditionTrue,
			}}
			Expect(k8sClient.Status().Update(ctx, job)).Should(Succeed())

			// Create output ConfigMap with DGD spec
			dgdYAML := `apiVersion: nvidia.com/v1alpha1
kind: DynamoGraphDeployment
metadata:
  name: test-dgd
spec:
  services:
    Frontend:
      replicas: 1`

			outputConfigMapName := getOutputConfigMapName(dgdr)
			cm := &corev1.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{
					Name:      outputConfigMapName,
					Namespace: namespace,
				},
				Data: map[string]string{
					ProfilingOutputFile: dgdYAML,
				},
			}
			Expect(k8sClient.Create(ctx, cm)).Should(Succeed())
			defer func() { _ = k8sClient.Delete(ctx, cm) }()

			// Reconcile to process the profiling completion
			_, err := reconciler.Reconcile(ctx, reconcile.Request{
				NamespacedName: types.NamespacedName{Name: dgdrName, Namespace: namespace},
			})
			Expect(err).NotTo(HaveOccurred())

			// Get the updated DGDR
			var updated nvidiacomv1alpha1.DynamoGraphDeploymentRequest
			Expect(k8sClient.Get(ctx, types.NamespacedName{Name: dgdrName, Namespace: namespace}, &updated)).Should(Succeed())

			// Check that DGD spec was generated
			Expect(updated.Status.GeneratedDeployment).NotTo(BeNil())

			// Verify state transitioned to Ready (since autoApply is false by default)
			Expect(updated.Status.State).Should(Equal(StateReady))
		})
	})

	Context("When autoApply is enabled", func() {
		It("Should create DGD after profiling", func() {
			ctx := context.Background()
			dgdrName := "test-dgdr-autoapply"
			namespace := defaultNamespace

			dgdr := &nvidiacomv1alpha1.DynamoGraphDeploymentRequest{
				ObjectMeta: metav1.ObjectMeta{
					Name:      dgdrName,
					Namespace: namespace,
				},
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentRequestSpec{
					Model:   "test-model",
					Backend: "vllm",
					ProfilingConfig: nvidiacomv1alpha1.ProfilingConfigSpec{
						ProfilerImage: "test-profiler:latest",
						Config: createTestConfig(map[string]interface{}{
							"engine": map[string]interface{}{
								"config": "/tmp/test-config.yaml",
							},
							"sla": map[string]interface{}{
								"ttft": 100.0,
								"itl":  1500.0,
								"isl":  3000,
								"osl":  5,
							},
						}),
					},
					AutoApply: true,
				},
			}

			Expect(k8sClient.Create(ctx, dgdr)).Should(Succeed())
			defer func() { _ = k8sClient.Delete(ctx, dgdr) }()

			// Update status to Profiling using Status subresource
			dgdr.Status.State = StateProfiling
			Expect(k8sClient.Status().Update(ctx, dgdr)).Should(Succeed())

			// Create completed profiling job
			jobName := getProfilingJobName(dgdr)
			job := &batchv1.Job{
				ObjectMeta: metav1.ObjectMeta{
					Name:      jobName,
					Namespace: namespace,
				},
				Spec: batchv1.JobSpec{
					Template: corev1.PodTemplateSpec{
						Spec: corev1.PodSpec{
							Containers: []corev1.Container{{
								Name:  "test",
								Image: "test",
							}},
							RestartPolicy: corev1.RestartPolicyNever,
						},
					},
				},
				Status: batchv1.JobStatus{
					Conditions: []batchv1.JobCondition{{
						Type:   batchv1.JobComplete,
						Status: corev1.ConditionTrue,
					}},
				},
			}
			Expect(k8sClient.Create(ctx, job)).Should(Succeed())
			defer func() { _ = k8sClient.Delete(ctx, job) }()

			// Update job status to completed using Status subresource
			job.Status.Conditions = []batchv1.JobCondition{{
				Type:   batchv1.JobComplete,
				Status: corev1.ConditionTrue,
			}}
			Expect(k8sClient.Status().Update(ctx, job)).Should(Succeed())

			// Create output ConfigMap
			dgdYAML := `apiVersion: nvidia.com/v1alpha1
kind: DynamoGraphDeployment
metadata:
  name: test-dgd-auto
spec:
  services:
    Frontend:
      replicas: 1`

			outputConfigMapName := getOutputConfigMapName(dgdr)
			cm := &corev1.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{
					Name:      outputConfigMapName,
					Namespace: namespace,
				},
				Data: map[string]string{
					ProfilingOutputFile: dgdYAML,
				},
			}
			Expect(k8sClient.Create(ctx, cm)).Should(Succeed())
			defer func() { _ = k8sClient.Delete(ctx, cm) }()

			// Reconcile to generate spec (transitions to Deploying because autoApply=true)
			_, err := reconciler.Reconcile(ctx, reconcile.Request{
				NamespacedName: types.NamespacedName{Name: dgdrName, Namespace: namespace},
			})
			Expect(err).NotTo(HaveOccurred())

			// Get updated DGDR and check state is Deploying
			var updated nvidiacomv1alpha1.DynamoGraphDeploymentRequest
			Expect(k8sClient.Get(ctx, types.NamespacedName{Name: dgdrName, Namespace: namespace}, &updated)).Should(Succeed())
			Expect(updated.Status.State).Should(Equal(StateDeploying))

			// Reconcile again to create DGD
			_, err = reconciler.Reconcile(ctx, reconcile.Request{
				NamespacedName: types.NamespacedName{Name: dgdrName, Namespace: namespace},
			})
			Expect(err).NotTo(HaveOccurred())

			// Verify DGD was created
			dgd := &nvidiacomv1alpha1.DynamoGraphDeployment{}
			Expect(k8sClient.Get(ctx, types.NamespacedName{Name: "test-dgd-auto", Namespace: namespace}, dgd)).Should(Succeed())

			// Get final DGDR status
			Expect(k8sClient.Get(ctx, types.NamespacedName{Name: dgdrName, Namespace: namespace}, &updated)).Should(Succeed())
			Expect(updated.Status.Deployment).NotTo(BeNil())
			Expect(updated.Status.Deployment.Created).Should(BeTrue())
			Expect(updated.Status.Deployment.Name).Should(Equal("test-dgd-auto"))

			// Clean up DGD
			Expect(k8sClient.Get(ctx, types.NamespacedName{Name: "test-dgd-auto", Namespace: namespace}, dgd)).Should(Succeed())
			_ = k8sClient.Delete(ctx, dgd)
		})
	})

	Context("When enforcing spec immutability", func() {
		It("Should reject spec changes after profiling starts", func() {
			ctx := context.Background()
			dgdrName := "test-dgdr-immutable"
			namespace := defaultNamespace

			dgdr := &nvidiacomv1alpha1.DynamoGraphDeploymentRequest{
				ObjectMeta: metav1.ObjectMeta{
					Name:      dgdrName,
					Namespace: namespace,
				},
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentRequestSpec{
					Model:   "test-model",
					Backend: "vllm",
					ProfilingConfig: nvidiacomv1alpha1.ProfilingConfigSpec{
						ProfilerImage: "test-profiler:latest",
						Config: createTestConfig(map[string]interface{}{
							"engine": map[string]interface{}{
								"config": "/tmp/test-config.yaml",
							},
							"sla": map[string]interface{}{
								"ttft": 100.0,
								"itl":  1500.0,
								"isl":  3000,
								"osl":  5,
							},
						}),
					},
				},
			}

			Expect(k8sClient.Create(ctx, dgdr)).Should(Succeed())
			defer func() { _ = k8sClient.Delete(ctx, dgdr) }()

			// Reconcile to initialize
			_, err := reconciler.Reconcile(ctx, reconcile.Request{
				NamespacedName: types.NamespacedName{Name: dgdrName, Namespace: namespace},
			})
			Expect(err).NotTo(HaveOccurred())

			// Get current generation
			var current nvidiacomv1alpha1.DynamoGraphDeploymentRequest
			Expect(k8sClient.Get(ctx, types.NamespacedName{Name: dgdrName, Namespace: namespace}, &current)).Should(Succeed())
			initialGeneration := current.Generation
			observedGeneration := current.Status.ObservedGeneration

			// Manually set state to Profiling to simulate in-progress profiling
			current.Status.State = StateProfiling
			Expect(k8sClient.Status().Update(ctx, &current)).Should(Succeed())

			// Try to modify spec
			Expect(k8sClient.Get(ctx, types.NamespacedName{Name: dgdrName, Namespace: namespace}, &current)).Should(Succeed())
			// Unmarshal config, modify it, and marshal back
			var config map[string]interface{}
			Expect(yaml.Unmarshal(current.Spec.ProfilingConfig.Config.Raw, &config)).Should(Succeed())
			config["sla"].(map[string]interface{})["ttft"] = 200.0
			current.Spec.ProfilingConfig.Config = createTestConfig(config)
			Expect(k8sClient.Update(ctx, &current)).Should(Succeed())

			// Reconcile
			_, err = reconciler.Reconcile(ctx, reconcile.Request{
				NamespacedName: types.NamespacedName{Name: dgdrName, Namespace: namespace},
			})
			Expect(err).NotTo(HaveOccurred())

			// Verify generation changed but observedGeneration stayed the same
			Expect(k8sClient.Get(ctx, types.NamespacedName{Name: dgdrName, Namespace: namespace}, &current)).Should(Succeed())
			Expect(current.Generation).Should(BeNumerically(">", initialGeneration))
			Expect(current.Status.ObservedGeneration).Should(Equal(observedGeneration))
			Expect(current.Status.State).Should(Equal(StateProfiling)) // State unchanged

			// Verify event was recorded
			Eventually(func() bool {
				select {
				case event := <-recorder.Events:
					return event == "Warning SpecChangeRejected Cannot modify spec in state 'Profiling'. DynamoGraphDeploymentRequest is immutable once profiling starts. Create a new resource with a different name instead."
				default:
					return false
				}
			}, timeout, interval).Should(BeTrue())
		})
	})

	Context("When handling DGD deletion", func() {
		It("Should transition to DeploymentDeleted state", func() {
			ctx := context.Background()
			dgdrName := "test-dgdr-dgd-deleted"
			namespace := defaultNamespace

			dgdr := &nvidiacomv1alpha1.DynamoGraphDeploymentRequest{
				ObjectMeta: metav1.ObjectMeta{
					Name:      dgdrName,
					Namespace: namespace,
				},
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentRequestSpec{
					Model:   "test-model",
					Backend: "vllm",
					ProfilingConfig: nvidiacomv1alpha1.ProfilingConfigSpec{
						ProfilerImage: "test-profiler:latest",
						Config: createTestConfig(map[string]interface{}{
							"engine": map[string]interface{}{
								"config": "/tmp/test-config.yaml",
							},
							"sla": map[string]interface{}{
								"ttft": 100.0,
								"itl":  1500.0,
								"isl":  3000,
								"osl":  5,
							},
						}),
					},
					AutoApply: true,
				},
			}

			Expect(k8sClient.Create(ctx, dgdr)).Should(Succeed())
			defer func() { _ = k8sClient.Delete(ctx, dgdr) }()

			// Update status to Ready with Deployment info using Status subresource
			dgdr.Status.State = StateReady
			dgdr.Status.Deployment = &nvidiacomv1alpha1.DeploymentStatus{
				Name:      "test-dgd-to-delete",
				Namespace: namespace,
				Created:   true,
				State:     "Ready",
			}
			Expect(k8sClient.Status().Update(ctx, dgdr)).Should(Succeed())

			// Reconcile when DGD doesn't exist (simulating deletion)
			_, err := reconciler.Reconcile(ctx, reconcile.Request{
				NamespacedName: types.NamespacedName{Name: dgdrName, Namespace: namespace},
			})
			Expect(err).NotTo(HaveOccurred())

			// Get updated DGDR and check state transitioned to DeploymentDeleted
			var updated nvidiacomv1alpha1.DynamoGraphDeploymentRequest
			Expect(k8sClient.Get(ctx, types.NamespacedName{Name: dgdrName, Namespace: namespace}, &updated)).Should(Succeed())
			Expect(updated.Status.State).Should(Equal(StateDeploymentDeleted))
		})
	})
})

var _ = Describe("DGDR Helper Functions", func() {
	Context("getProfilingJobName", func() {
		It("Should return correct job name", func() {
			dgdr := &nvidiacomv1alpha1.DynamoGraphDeploymentRequest{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test-dgdr",
				},
			}
			Expect(getProfilingJobName(dgdr)).Should(Equal("profile-test-dgdr"))
		})
	})

	Context("getOutputConfigMapName", func() {
		It("Should return correct ConfigMap name", func() {
			dgdr := &nvidiacomv1alpha1.DynamoGraphDeploymentRequest{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test-dgdr",
				},
			}
			Expect(getOutputConfigMapName(dgdr)).Should(Equal("dgdr-output-test-dgdr"))
		})
	})

	Context("isOnlineProfiling", func() {
		It("Should return true for online profiling (use_ai_configurator=false)", func() {
			dgdr := &nvidiacomv1alpha1.DynamoGraphDeploymentRequest{
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentRequestSpec{
					ProfilingConfig: nvidiacomv1alpha1.ProfilingConfigSpec{
						Config: createTestConfig(map[string]interface{}{
							"sweep": map[string]interface{}{
								"use_ai_configurator": false,
							},
						}),
					},
				},
			}
			Expect(isOnlineProfiling(dgdr)).Should(BeTrue())
		})

		It("Should return false for AI Configurator profiling (use_ai_configurator=true)", func() {
			dgdr := &nvidiacomv1alpha1.DynamoGraphDeploymentRequest{
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentRequestSpec{
					ProfilingConfig: nvidiacomv1alpha1.ProfilingConfigSpec{
						Config: createTestConfig(map[string]interface{}{
							"sweep": map[string]interface{}{
								"use_ai_configurator": true,
							},
						}),
					},
				},
			}
			Expect(isOnlineProfiling(dgdr)).Should(BeFalse())
		})

		It("Should return true by default when sweep section is missing", func() {
			dgdr := &nvidiacomv1alpha1.DynamoGraphDeploymentRequest{
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentRequestSpec{
					ProfilingConfig: nvidiacomv1alpha1.ProfilingConfigSpec{
						Config: createTestConfig(map[string]interface{}{
							"engine": map[string]interface{}{
								"backend": "vllm",
							},
						}),
					},
				},
			}
			Expect(isOnlineProfiling(dgdr)).Should(BeTrue())
		})

		It("Should return true by default when use_ai_configurator is not specified", func() {
			dgdr := &nvidiacomv1alpha1.DynamoGraphDeploymentRequest{
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentRequestSpec{
					ProfilingConfig: nvidiacomv1alpha1.ProfilingConfigSpec{
						Config: createTestConfig(map[string]interface{}{
							"sweep": map[string]interface{}{
								"prefill_interpolation_granularity": 16,
							},
						}),
					},
				},
			}
			Expect(isOnlineProfiling(dgdr)).Should(BeTrue())
		})
	})
})

var _ = Describe("DGDR Validation", func() {
	var reconciler *DynamoGraphDeploymentRequestReconciler

	BeforeEach(func() {
		reconciler = &DynamoGraphDeploymentRequestReconciler{
			Client: k8sClient,
		}
	})

	Context("validateSpec", func() {
		It("Should pass validation for valid spec", func() {
			ctx := context.Background()
			dgdr := &nvidiacomv1alpha1.DynamoGraphDeploymentRequest{
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentRequestSpec{
					Model:   "test-model",
					Backend: "vllm",
					ProfilingConfig: nvidiacomv1alpha1.ProfilingConfigSpec{
						ProfilerImage: "test-profiler:latest",
						Config: createTestConfig(map[string]interface{}{
							"engine": map[string]interface{}{
								"config": "/tmp/test-config.yaml",
							},
							"sla": map[string]interface{}{
								"ttft": 100.0,
								"itl":  1500.0,
								"isl":  3000,
								"osl":  5,
							},
						}),
					},
				},
			}

			err := reconciler.validateSpec(ctx, dgdr)
			Expect(err).NotTo(HaveOccurred())
		})

		It("Should pass validation with minimal config", func() {
			ctx := context.Background()
			dgdr := &nvidiacomv1alpha1.DynamoGraphDeploymentRequest{
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentRequestSpec{
					Model:   "test-model",
					Backend: "vllm",
					ProfilingConfig: nvidiacomv1alpha1.ProfilingConfigSpec{
						ProfilerImage: "test-profiler:latest",
						Config: createTestConfig(map[string]interface{}{
							"sla": map[string]interface{}{
								"ttft": 100.0,
								"itl":  1500.0,
							},
						}),
					},
				},
			}

			// Validation should pass - profiler will auto-generate missing config
			err := reconciler.validateSpec(ctx, dgdr)
			Expect(err).NotTo(HaveOccurred())
		})
	})
})

var _ = Describe("DGDR Profiler Arguments", func() {
	var reconciler *DynamoGraphDeploymentRequestReconciler

	BeforeEach(func() {
		reconciler = &DynamoGraphDeploymentRequestReconciler{
			Client:   k8sClient,
			Recorder: record.NewFakeRecorder(100),
			Config: commonController.Config{
				RestrictedNamespace: "",
			},
			RBACManager: &MockRBACManager{},
		}
	})

	Context("When creating profiling job with inline config", func() {
		It("Should pass config as --profile-config argument for online profiling", func() {
			ctx := context.Background()
			namespace := "default"
			dgdrName := "test-args-online"

			// Create ServiceAccount
			sa := &corev1.ServiceAccount{
				ObjectMeta: metav1.ObjectMeta{
					Name:      ServiceAccountProfilingJob,
					Namespace: namespace,
				},
			}
			Expect(k8sClient.Create(ctx, sa)).Should(Succeed())
			defer func() { _ = k8sClient.Delete(ctx, sa) }()

			dgdr := &nvidiacomv1alpha1.DynamoGraphDeploymentRequest{
				ObjectMeta: metav1.ObjectMeta{
					Name:      dgdrName,
					Namespace: namespace,
				},
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentRequestSpec{
					Model:   "test-model",
					Backend: "trtllm",
					ProfilingConfig: nvidiacomv1alpha1.ProfilingConfigSpec{
						ProfilerImage: "test-profiler:latest",
						Config: createTestConfig(map[string]interface{}{
							"engine": map[string]interface{}{
								"config":         "/tmp/test-config.yaml",
								"profiler_image": "test-profiler:latest",
							},
							"sla": map[string]interface{}{
								"ttft": 50.0,
								"itl":  10.0,
								"isl":  3000,
								"osl":  500,
							},
							"hardware": map[string]interface{}{
								"gpu_type":                "h200_sxm",
								"min_num_gpus_per_engine": 2,
								"max_num_gpus_per_engine": 4,
							},
							"sweep": map[string]interface{}{
								"use_ai_configurator": false,
							},
						}),
					},
				},
			}

			Expect(k8sClient.Create(ctx, dgdr)).Should(Succeed())
			defer func() { _ = k8sClient.Delete(ctx, dgdr) }()

			// Re-fetch DGDR to get proper metadata from API server
			var fetchedDGDR nvidiacomv1alpha1.DynamoGraphDeploymentRequest
			Expect(k8sClient.Get(ctx, types.NamespacedName{Name: dgdrName, Namespace: namespace}, &fetchedDGDR)).Should(Succeed())

			// Create profiling job with properly initialized DGDR
			err := reconciler.createProfilingJob(ctx, &fetchedDGDR)
			Expect(err).NotTo(HaveOccurred())

			// Verify job was created
			jobName := getProfilingJobName(&fetchedDGDR)
			job := &batchv1.Job{}
			Expect(k8sClient.Get(ctx, types.NamespacedName{Name: jobName, Namespace: namespace}, job)).Should(Succeed())

			// Verify profiler container has --profile-config argument
			profilerContainer := job.Spec.Template.Spec.Containers[0]
			args := profilerContainer.Args

			// Check that --profile-config argument is present
			Expect(args).Should(ContainElement("--profile-config"))

			// Clean up
			_ = k8sClient.Delete(ctx, job)
		})

		It("Should pass config with AI Configurator settings for offline profiling", func() {
			ctx := context.Background()
			namespace := defaultNamespace
			dgdrName := "test-args-offline"

			// Create ServiceAccount
			sa := &corev1.ServiceAccount{
				ObjectMeta: metav1.ObjectMeta{
					Name:      ServiceAccountProfilingJob,
					Namespace: namespace,
				},
			}
			Expect(k8sClient.Create(ctx, sa)).Should(Succeed())
			defer func() { _ = k8sClient.Delete(ctx, sa) }()

			dgdr := &nvidiacomv1alpha1.DynamoGraphDeploymentRequest{
				ObjectMeta: metav1.ObjectMeta{
					Name:      dgdrName,
					Namespace: namespace,
				},
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentRequestSpec{
					Model:   "test-model",
					Backend: "trtllm",
					ProfilingConfig: nvidiacomv1alpha1.ProfilingConfigSpec{
						ProfilerImage: "test-profiler:latest",
						Config: createTestConfig(map[string]interface{}{
							"engine": map[string]interface{}{
								"config":         "/tmp/test-config.yaml",
								"profiler_image": "test-profiler:latest",
							},
							"sla": map[string]interface{}{
								"ttft": 50.0,
								"itl":  10.0,
								"isl":  3000,
								"osl":  500,
							},
							"hardware": map[string]interface{}{
								"gpu_type":                "h200_sxm",
								"min_num_gpus_per_engine": 1,
								"max_num_gpus_per_engine": 8,
							},
							"sweep": map[string]interface{}{
								"use_ai_configurator": true,
								"aic_system":          "h200_sxm",
								"aic_hf_id":           "Qwen/Qwen3-32B",
								"aic_backend_version": "0.20.0",
							},
						}),
					},
				},
			}

			Expect(k8sClient.Create(ctx, dgdr)).Should(Succeed())
			defer func() { _ = k8sClient.Delete(ctx, dgdr) }()

			// Re-fetch DGDR to get proper metadata from API server
			var fetchedDGDR nvidiacomv1alpha1.DynamoGraphDeploymentRequest
			Expect(k8sClient.Get(ctx, types.NamespacedName{Name: dgdrName, Namespace: namespace}, &fetchedDGDR)).Should(Succeed())

			// Create profiling job with properly initialized DGDR
			err := reconciler.createProfilingJob(ctx, &fetchedDGDR)
			Expect(err).NotTo(HaveOccurred())

			// Verify job was created
			jobName := getProfilingJobName(&fetchedDGDR)
			job := &batchv1.Job{}
			Expect(k8sClient.Get(ctx, types.NamespacedName{Name: jobName, Namespace: namespace}, job)).Should(Succeed())

			// Verify profiler container has --profile-config argument
			profilerContainer := job.Spec.Template.Spec.Containers[0]
			args := profilerContainer.Args

			// Check that --profile-config argument is present
			Expect(args).Should(ContainElement("--profile-config"))

			// Clean up
			_ = k8sClient.Delete(ctx, job)
		})

		It("Should set fsGroup in pod security context for volume permissions", func() {
			ctx := context.Background()
			namespace := "default"
			dgdrName := "test-fsgroup"

			// Create ServiceAccount
			sa := &corev1.ServiceAccount{
				ObjectMeta: metav1.ObjectMeta{
					Name:      ServiceAccountProfilingJob,
					Namespace: namespace,
				},
			}
			Expect(k8sClient.Create(ctx, sa)).Should(Succeed())
			defer func() { _ = k8sClient.Delete(ctx, sa) }()

			dgdr := &nvidiacomv1alpha1.DynamoGraphDeploymentRequest{
				ObjectMeta: metav1.ObjectMeta{
					Name:      dgdrName,
					Namespace: namespace,
				},
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentRequestSpec{
					Model:   "test-model",
					Backend: "trtllm",
					ProfilingConfig: nvidiacomv1alpha1.ProfilingConfigSpec{
						ProfilerImage: "test-profiler:latest",
						Config: createTestConfig(map[string]interface{}{
							"sla": map[string]interface{}{
								"ttft": 50.0,
								"itl":  10.0,
								"isl":  3000,
								"osl":  500,
							},
						}),
					},
				},
			}

			Expect(k8sClient.Create(ctx, dgdr)).Should(Succeed())
			defer func() { _ = k8sClient.Delete(ctx, dgdr) }()

			// Re-fetch DGDR to get proper metadata from API server
			var fetchedDGDR nvidiacomv1alpha1.DynamoGraphDeploymentRequest
			Expect(k8sClient.Get(ctx, types.NamespacedName{Name: dgdrName, Namespace: namespace}, &fetchedDGDR)).Should(Succeed())

			// Create profiling job with properly initialized DGDR
			err := reconciler.createProfilingJob(ctx, &fetchedDGDR)
			Expect(err).NotTo(HaveOccurred())

			// Verify job was created
			jobName := getProfilingJobName(&fetchedDGDR)
			job := &batchv1.Job{}
			Expect(k8sClient.Get(ctx, types.NamespacedName{Name: jobName, Namespace: namespace}, job)).Should(Succeed())

			// Verify security context has all security fields set correctly
			podSecurityContext := job.Spec.Template.Spec.SecurityContext
			Expect(podSecurityContext).NotTo(BeNil())
			Expect(podSecurityContext.RunAsNonRoot).NotTo(BeNil())
			Expect(*podSecurityContext.RunAsNonRoot).To(BeTrue())
			Expect(podSecurityContext.RunAsUser).NotTo(BeNil())
			Expect(*podSecurityContext.RunAsUser).To(Equal(int64(1000)))
			Expect(podSecurityContext.RunAsGroup).NotTo(BeNil())
			Expect(*podSecurityContext.RunAsGroup).To(Equal(int64(1000)))
			Expect(podSecurityContext.FSGroup).NotTo(BeNil())
			Expect(*podSecurityContext.FSGroup).To(Equal(int64(1000)))

			// Clean up
			_ = k8sClient.Delete(ctx, job)
		})
	})
})

var _ = Describe("DGDR Error Handling", func() {
	var reconciler *DynamoGraphDeploymentRequestReconciler
	var recorder *record.FakeRecorder

	BeforeEach(func() {
		recorder = record.NewFakeRecorder(100)
		reconciler = &DynamoGraphDeploymentRequestReconciler{
			Client:   k8sClient,
			Recorder: recorder,
			Config: commonController.Config{
				RestrictedNamespace: "",
			},
			RBACManager: &MockRBACManager{},
		}
	})

	Context("When profiling job fails", func() {
		It("Should capture detailed error from pod termination state", func() {
			ctx := context.Background()
			namespace := defaultNamespace
			dgdrName := "test-error-capture"

			dgdr := &nvidiacomv1alpha1.DynamoGraphDeploymentRequest{
				ObjectMeta: metav1.ObjectMeta{
					Name:      dgdrName,
					Namespace: namespace,
				},
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentRequestSpec{
					Model:   "test-model",
					Backend: "vllm",
					ProfilingConfig: nvidiacomv1alpha1.ProfilingConfigSpec{
						ProfilerImage: "test-profiler:latest",
						Config: createTestConfig(map[string]interface{}{
							"engine": map[string]interface{}{
								"config": "/tmp/test-config.yaml",
							},
							"sla": map[string]interface{}{
								"ttft": 100.0,
								"itl":  1500.0,
								"isl":  3000,
								"osl":  5,
							},
							"hardware": map[string]interface{}{
								"min_num_gpus_per_engine": 1,
								"max_num_gpus_per_engine": 8,
							},
						}),
					},
				},
			}

			Expect(k8sClient.Create(ctx, dgdr)).Should(Succeed())
			defer func() { _ = k8sClient.Delete(ctx, dgdr) }()

			// Set status to Profiling
			dgdr.Status.State = StateProfiling
			Expect(k8sClient.Status().Update(ctx, dgdr)).Should(Succeed())

			// Create failed job
			jobName := getProfilingJobName(dgdr)
			job := &batchv1.Job{
				ObjectMeta: metav1.ObjectMeta{
					Name:      jobName,
					Namespace: namespace,
				},
				Spec: batchv1.JobSpec{
					Template: corev1.PodTemplateSpec{
						Spec: corev1.PodSpec{
							Containers: []corev1.Container{{
								Name:  ContainerNameProfiler,
								Image: "test",
							}},
							RestartPolicy: corev1.RestartPolicyNever,
						},
					},
				},
				Status: batchv1.JobStatus{
					Conditions: []batchv1.JobCondition{{
						Type:    batchv1.JobFailed,
						Status:  corev1.ConditionTrue,
						Message: "BackoffLimitExceeded",
					}},
				},
			}
			Expect(k8sClient.Create(ctx, job)).Should(Succeed())
			defer func() { _ = k8sClient.Delete(ctx, job) }()

			// Update job status
			job.Status.Conditions = []batchv1.JobCondition{{
				Type:    batchv1.JobFailed,
				Status:  corev1.ConditionTrue,
				Message: "BackoffLimitExceeded",
			}}
			Expect(k8sClient.Status().Update(ctx, job)).Should(Succeed())

			// Create failed pod with termination details
			pod := &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      jobName + "-pod",
					Namespace: namespace,
					Labels: map[string]string{
						"job-name": jobName,
					},
				},
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{{
						Name:  ContainerNameProfiler,
						Image: "test",
					}},
					RestartPolicy: corev1.RestartPolicyNever,
				},
				Status: corev1.PodStatus{
					Phase: corev1.PodFailed,
					ContainerStatuses: []corev1.ContainerStatus{{
						Name: ContainerNameProfiler,
						State: corev1.ContainerState{
							Terminated: &corev1.ContainerStateTerminated{
								ExitCode: 1,
								Reason:   "Error",
								Message:  "ValueError: Invalid model name for AI Configurator",
							},
						},
					}},
				},
			}
			Expect(k8sClient.Create(ctx, pod)).Should(Succeed())
			defer func() { _ = k8sClient.Delete(ctx, pod) }()

			// Reconcile - should capture error details
			_, err := reconciler.Reconcile(ctx, reconcile.Request{
				NamespacedName: types.NamespacedName{Name: dgdrName, Namespace: namespace},
			})
			Expect(err).NotTo(HaveOccurred())

			// Verify DGDR transitioned to Failed state
			var updated nvidiacomv1alpha1.DynamoGraphDeploymentRequest
			Expect(k8sClient.Get(ctx, types.NamespacedName{Name: dgdrName, Namespace: namespace}, &updated)).Should(Succeed())
			Expect(updated.Status.State).Should(Equal(StateFailed))

			// Verify error condition contains detailed error
			condition := meta.FindStatusCondition(updated.Status.Conditions, ConditionTypeProfiling)
			Expect(condition).NotTo(BeNil())
			Expect(condition.Status).Should(Equal(metav1.ConditionFalse))
			Expect(condition.Message).Should(ContainSubstring("profiling job failed"))
		})
	})

	Context("When parsing multi-document YAML", func() {
		It("Should extract DGD from ConfigMap + DGD YAML", func() {
			// Multi-document YAML with ConfigMap first, then DGD
			multiDocYAML := `---
apiVersion: v1
kind: ConfigMap
metadata:
  name: test-config
  namespace: default
data:
  some-data: "value"
---
apiVersion: nvidia.com/v1alpha1
kind: DynamoGraphDeployment
metadata:
  name: test-dgd
  namespace: default
spec:
  backendFramework: vllm
  services: {}`

			dgd, err := reconciler.extractDGDFromYAML([]byte(multiDocYAML))
			Expect(err).NotTo(HaveOccurred())
			Expect(dgd).NotTo(BeNil())
			Expect(dgd.Kind).Should(Equal("DynamoGraphDeployment"))
			Expect(dgd.Name).Should(Equal("test-dgd"))
			Expect(dgd.Spec.BackendFramework).Should(Equal("vllm"))
		})

		It("Should extract DGD from single-document YAML", func() {
			// Single document YAML without separator
			singleDocYAML := `apiVersion: nvidia.com/v1alpha1
kind: DynamoGraphDeployment
metadata:
  name: test-dgd-single
  namespace: default
spec:
  backendFramework: vllm
  services: {}`

			dgd, err := reconciler.extractDGDFromYAML([]byte(singleDocYAML))
			Expect(err).NotTo(HaveOccurred())
			Expect(dgd).NotTo(BeNil())
			Expect(dgd.Kind).Should(Equal("DynamoGraphDeployment"))
			Expect(dgd.Name).Should(Equal("test-dgd-single"))
		})

		It("Should handle DGD + ConfigMap order (DGD first)", func() {
			// Multi-document YAML with DGD first, then ConfigMap
			multiDocYAML := `---
apiVersion: nvidia.com/v1alpha1
kind: DynamoGraphDeployment
metadata:
  name: test-dgd-first
  namespace: default
spec:
  backendFramework: vllm
  services: {}
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: test-config
  namespace: default
data:
  some-data: "value"`

			dgd, err := reconciler.extractDGDFromYAML([]byte(multiDocYAML))
			Expect(err).NotTo(HaveOccurred())
			Expect(dgd).NotTo(BeNil())
			Expect(dgd.Kind).Should(Equal("DynamoGraphDeployment"))
			Expect(dgd.Name).Should(Equal("test-dgd-first"))
		})

		It("Should return error when no DGD found", func() {
			// YAML with only ConfigMap
			configMapOnlyYAML := `---
apiVersion: v1
kind: ConfigMap
metadata:
  name: test-config
  namespace: default
data:
  some-data: "value"`

			_, err := reconciler.extractDGDFromYAML([]byte(configMapOnlyYAML))
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).Should(ContainSubstring("no DynamoGraphDeployment found"))
		})

		It("Should handle YAML with leading separator", func() {
			// YAML starting with --- separator
			yamlWithLeadingSeparator := `---
apiVersion: nvidia.com/v1alpha1
kind: DynamoGraphDeployment
metadata:
  name: test-dgd-leading
  namespace: default
spec:
  backendFramework: vllm
  services: {}`

			dgd, err := reconciler.extractDGDFromYAML([]byte(yamlWithLeadingSeparator))
			Expect(err).NotTo(HaveOccurred())
			Expect(dgd).NotTo(BeNil())
			Expect(dgd.Name).Should(Equal("test-dgd-leading"))
		})

		It("Should extract DGD and additional resources correctly", func() {
			// Multi-document YAML with ConfigMap and DGD
			multiDocYAML := `---
apiVersion: v1
kind: ConfigMap
metadata:
  name: model-config
  namespace: default
data:
  model.json: '{"name": "test-model"}'
---
apiVersion: nvidia.com/v1alpha1
kind: DynamoGraphDeployment
metadata:
  name: test-dgd
  namespace: default
spec:
  backendFramework: vllm
  services: {}`

			dgd, additionalResources, err := reconciler.extractResourcesFromYAML([]byte(multiDocYAML))
			Expect(err).NotTo(HaveOccurred())
			Expect(dgd).NotTo(BeNil())
			Expect(dgd.Name).Should(Equal("test-dgd"))
			Expect(additionalResources).To(HaveLen(1))
			Expect(additionalResources[0].GetKind()).Should(Equal("ConfigMap"))
			Expect(additionalResources[0].GetName()).Should(Equal("model-config"))
		})

		It("Should handle multiple additional resources", func() {
			// Multi-document YAML with multiple ConfigMaps and DGD
			multiDocYAML := `---
apiVersion: v1
kind: ConfigMap
metadata:
  name: config1
data:
  key1: value1
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: config2
data:
  key2: value2
---
apiVersion: nvidia.com/v1alpha1
kind: DynamoGraphDeployment
metadata:
  name: test-dgd
spec:
  backendFramework: vllm
  services: {}`

			dgd, additionalResources, err := reconciler.extractResourcesFromYAML([]byte(multiDocYAML))
			Expect(err).NotTo(HaveOccurred())
			Expect(dgd).NotTo(BeNil())
			Expect(additionalResources).To(HaveLen(2))
			Expect(additionalResources[0].GetName()).Should(Equal("config1"))
			Expect(additionalResources[1].GetName()).Should(Equal("config2"))
		})
	})
})
