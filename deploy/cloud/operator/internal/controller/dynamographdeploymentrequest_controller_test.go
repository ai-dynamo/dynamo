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
	"time"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/cloud/operator/api/v1alpha1"
	commonController "github.com/ai-dynamo/dynamo/deploy/cloud/operator/internal/controller_common"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/tools/record"
	"sigs.k8s.io/controller-runtime/pkg/reconcile"
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
			Client:        k8sClient,
			Recorder:      recorder,
			ProfilerImage: "test-profiler:latest",
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
			namespace := "default"

			dgdr := &nvidiacomv1alpha1.DynamoGraphDeploymentRequest{
				ObjectMeta: metav1.ObjectMeta{
					Name:      dgdrName,
					Namespace: namespace,
				},
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentRequestSpec{
					ModelName: "test-model",
					Backend:   BackendVLLM,
					SLA: nvidiacomv1alpha1.SLASpec{
						TTFT: 100,
						ITL:  1500,
						ISL:  3000,
						OSL:  5,
					},
					Online: true,
				},
			}

			Expect(k8sClient.Create(ctx, dgdr)).Should(Succeed())
			defer k8sClient.Delete(ctx, dgdr)

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
				k8sClient.Get(ctx, types.NamespacedName{Name: dgdrName, Namespace: namespace}, &updated)
				return updated.Status.State
			}, timeout, interval).Should(Equal(StatePending))

			// Verify observedGeneration is set
			var updated nvidiacomv1alpha1.DynamoGraphDeploymentRequest
			k8sClient.Get(ctx, types.NamespacedName{Name: dgdrName, Namespace: namespace}, &updated)
			Expect(updated.Status.ObservedGeneration).Should(Equal(updated.Generation))
		})

		It("Should fail validation with missing modelName", func() {
			ctx := context.Background()
			dgdrName := "test-dgdr-invalid"
			namespace := "default"

			dgdr := &nvidiacomv1alpha1.DynamoGraphDeploymentRequest{
				ObjectMeta: metav1.ObjectMeta{
					Name:      dgdrName,
					Namespace: namespace,
				},
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentRequestSpec{
					Backend: BackendVLLM,
					SLA: nvidiacomv1alpha1.SLASpec{
						TTFT: 100,
						ITL:  1500,
						ISL:  3000,
						OSL:  5,
					},
				},
			}

			Expect(k8sClient.Create(ctx, dgdr)).Should(Succeed())
			defer k8sClient.Delete(ctx, dgdr)

			// Reconcile
			_, err := reconciler.Reconcile(ctx, reconcile.Request{
				NamespacedName: types.NamespacedName{
					Name:      dgdrName,
					Namespace: namespace,
				},
			})
			Expect(err).NotTo(HaveOccurred())

			// Check status transitions to Failed
			Eventually(func() string {
				var updated nvidiacomv1alpha1.DynamoGraphDeploymentRequest
				k8sClient.Get(ctx, types.NamespacedName{Name: dgdrName, Namespace: namespace}, &updated)
				return updated.Status.State
			}, timeout, interval).Should(Equal(StateFailed))
		})
	})

	Context("When creating profiling job", func() {
		It("Should create online profiling job", func() {
			ctx := context.Background()
			dgdrName := "test-dgdr-profiling-online"
			namespace := "default"

			// Create ConfigMap for profiling config
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
			defer k8sClient.Delete(ctx, configMap)

			// Create ServiceAccount
			sa := &corev1.ServiceAccount{
				ObjectMeta: metav1.ObjectMeta{
					Name:      ServiceAccountProfilingJob,
					Namespace: namespace,
				},
			}
			Expect(k8sClient.Create(ctx, sa)).Should(Succeed())
			defer k8sClient.Delete(ctx, sa)

			dgdr := &nvidiacomv1alpha1.DynamoGraphDeploymentRequest{
				ObjectMeta: metav1.ObjectMeta{
					Name:      dgdrName,
					Namespace: namespace,
				},
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentRequestSpec{
					ModelName: "test-model",
					Backend:   BackendVLLM,
					SLA: nvidiacomv1alpha1.SLASpec{
						TTFT: 100,
						ITL:  1500,
						ISL:  3000,
						OSL:  5,
					},
					Online: true,
					ProfilingConfig: &nvidiacomv1alpha1.ProfilingConfigSpec{
						ConfigMapRef: &nvidiacomv1alpha1.ConfigMapKeySelector{
							Name: "test-config",
							Key:  "disagg.yaml",
						},
					},
				},
			}

			Expect(k8sClient.Create(ctx, dgdr)).Should(Succeed())
			defer k8sClient.Delete(ctx, dgdr)

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
			k8sClient.Get(ctx, types.NamespacedName{Name: jobName, Namespace: namespace}, job)
			Expect(job.Labels[LabelApp]).Should(Equal(LabelValueDynamoProfiler))
			Expect(job.Labels[LabelDGDR]).Should(Equal(dgdrName))

			// Verify job has profiler container
			Expect(job.Spec.Template.Spec.Containers).Should(HaveLen(2))
			Expect(job.Spec.Template.Spec.Containers[0].Name).Should(Equal(ContainerNameProfiler))
			Expect(job.Spec.Template.Spec.Containers[1].Name).Should(Equal(ContainerNameOutputCopier))

			// Verify PVC volume mount
			Expect(job.Spec.Template.Spec.Volumes).Should(ContainElement(
				corev1.Volume{
					Name: VolumeNameProfilingOutput,
					VolumeSource: corev1.VolumeSource{
						PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSource{
							ClaimName: "dynamo-pvc",
						},
					},
				},
			))

			// Clean up job
			k8sClient.Delete(ctx, job)
		})

		It("Should create offline (AIC) profiling job", func() {
			ctx := context.Background()
			dgdrName := "test-dgdr-profiling-aic"
			namespace := "default"

			// Create ServiceAccount
			sa := &corev1.ServiceAccount{
				ObjectMeta: metav1.ObjectMeta{
					Name:      ServiceAccountProfilingJob,
					Namespace: namespace,
				},
			}
			_ = k8sClient.Create(ctx, sa)
			defer k8sClient.Delete(ctx, sa)

			dgdr := &nvidiacomv1alpha1.DynamoGraphDeploymentRequest{
				ObjectMeta: metav1.ObjectMeta{
					Name:      dgdrName,
					Namespace: namespace,
				},
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentRequestSpec{
					ModelName: "QWEN3_32B",
					Backend:   BackendTRTLLM,
					SLA: nvidiacomv1alpha1.SLASpec{
						TTFT: 100,
						ITL:  1500,
						ISL:  3000,
						OSL:  5,
					},
					Online: false, // Offline profiling
					GPU: &nvidiacomv1alpha1.GPUSpec{
						Type:                "h200_sxm",
						MinNumGPUsPerEngine: 1,
						MaxNumGPUsPerEngine: 8,
					},
				},
			}

			Expect(k8sClient.Create(ctx, dgdr)).Should(Succeed())
			defer k8sClient.Delete(ctx, dgdr)

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
				k8sClient.Delete(ctx, job)
			}
		})
	})

	Context("When profiling completes", func() {
		It("Should generate DGD spec from ConfigMap", func() {
			ctx := context.Background()
			dgdrName := "test-dgdr-profiling-complete"
			namespace := "default"

			dgdr := &nvidiacomv1alpha1.DynamoGraphDeploymentRequest{
				ObjectMeta: metav1.ObjectMeta{
					Name:      dgdrName,
					Namespace: namespace,
				},
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentRequestSpec{
					ModelName: "test-model",
					Backend:   BackendVLLM,
					SLA: nvidiacomv1alpha1.SLASpec{
						TTFT: 100,
						ITL:  1500,
						ISL:  3000,
						OSL:  5,
					},
					Online: true,
				},
			}

			Expect(k8sClient.Create(ctx, dgdr)).Should(Succeed())
			defer k8sClient.Delete(ctx, dgdr)

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
			defer k8sClient.Delete(ctx, job)

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
			defer k8sClient.Delete(ctx, cm)

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
			namespace := "default"

			dgdr := &nvidiacomv1alpha1.DynamoGraphDeploymentRequest{
				ObjectMeta: metav1.ObjectMeta{
					Name:      dgdrName,
					Namespace: namespace,
				},
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentRequestSpec{
					ModelName: "test-model",
					Backend:   BackendVLLM,
					SLA: nvidiacomv1alpha1.SLASpec{
						TTFT: 100,
						ITL:  1500,
						ISL:  3000,
						OSL:  5,
					},
					Online:    true,
					AutoApply: true,
				},
			}

			Expect(k8sClient.Create(ctx, dgdr)).Should(Succeed())
			defer k8sClient.Delete(ctx, dgdr)

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
			defer k8sClient.Delete(ctx, job)

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
			defer k8sClient.Delete(ctx, cm)

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
			k8sClient.Get(ctx, types.NamespacedName{Name: dgdrName, Namespace: namespace}, &updated)
			Expect(updated.Status.Deployment).NotTo(BeNil())
			Expect(updated.Status.Deployment.Created).Should(BeTrue())
			Expect(updated.Status.Deployment.Name).Should(Equal("test-dgd-auto"))

			// Clean up DGD
			k8sClient.Get(ctx, types.NamespacedName{Name: "test-dgd-auto", Namespace: namespace}, dgd)
			k8sClient.Delete(ctx, dgd)
		})
	})

	Context("When enforcing spec immutability", func() {
		It("Should reject spec changes after profiling starts", func() {
			ctx := context.Background()
			dgdrName := "test-dgdr-immutable"
			namespace := "default"

			dgdr := &nvidiacomv1alpha1.DynamoGraphDeploymentRequest{
				ObjectMeta: metav1.ObjectMeta{
					Name:      dgdrName,
					Namespace: namespace,
				},
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentRequestSpec{
					ModelName: "test-model",
					Backend:   BackendVLLM,
					SLA: nvidiacomv1alpha1.SLASpec{
						TTFT: 100,
						ITL:  1500,
						ISL:  3000,
						OSL:  5,
					},
					Online: true,
				},
			}

			Expect(k8sClient.Create(ctx, dgdr)).Should(Succeed())
			defer k8sClient.Delete(ctx, dgdr)

			// Reconcile to initialize
			_, err := reconciler.Reconcile(ctx, reconcile.Request{
				NamespacedName: types.NamespacedName{Name: dgdrName, Namespace: namespace},
			})
			Expect(err).NotTo(HaveOccurred())

			// Get current generation
			var current nvidiacomv1alpha1.DynamoGraphDeploymentRequest
			k8sClient.Get(ctx, types.NamespacedName{Name: dgdrName, Namespace: namespace}, &current)
			initialGeneration := current.Generation
			observedGeneration := current.Status.ObservedGeneration

			// Manually set state to Profiling to simulate in-progress profiling
			current.Status.State = StateProfiling
			k8sClient.Status().Update(ctx, &current)

			// Try to modify spec
			k8sClient.Get(ctx, types.NamespacedName{Name: dgdrName, Namespace: namespace}, &current)
			current.Spec.SLA.TTFT = 200
			k8sClient.Update(ctx, &current)

			// Reconcile
			_, err = reconciler.Reconcile(ctx, reconcile.Request{
				NamespacedName: types.NamespacedName{Name: dgdrName, Namespace: namespace},
			})
			Expect(err).NotTo(HaveOccurred())

			// Verify generation changed but observedGeneration stayed the same
			k8sClient.Get(ctx, types.NamespacedName{Name: dgdrName, Namespace: namespace}, &current)
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
			namespace := "default"

			dgdr := &nvidiacomv1alpha1.DynamoGraphDeploymentRequest{
				ObjectMeta: metav1.ObjectMeta{
					Name:      dgdrName,
					Namespace: namespace,
				},
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentRequestSpec{
					ModelName: "test-model",
					Backend:   BackendVLLM,
					SLA: nvidiacomv1alpha1.SLASpec{
						TTFT: 100,
						ITL:  1500,
						ISL:  3000,
						OSL:  5,
					},
					Online:    true,
					AutoApply: true,
				},
			}

			Expect(k8sClient.Create(ctx, dgdr)).Should(Succeed())
			defer k8sClient.Delete(ctx, dgdr)

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
		It("Should return correct job name for online profiling", func() {
			dgdr := &nvidiacomv1alpha1.DynamoGraphDeploymentRequest{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test-dgdr",
				},
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentRequestSpec{
					Online: true,
				},
			}
			Expect(getProfilingJobName(dgdr)).Should(Equal("profile-online-test-dgdr"))
		})

		It("Should return correct job name for offline profiling", func() {
			dgdr := &nvidiacomv1alpha1.DynamoGraphDeploymentRequest{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test-dgdr",
				},
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentRequestSpec{
					Online: false,
				},
			}
			Expect(getProfilingJobName(dgdr)).Should(Equal("profile-aic-test-dgdr"))
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
					ModelName: "test-model",
					Backend:   BackendVLLM,
					SLA: nvidiacomv1alpha1.SLASpec{
						TTFT: 100,
						ITL:  1500,
						ISL:  3000,
						OSL:  5,
					},
				},
			}

			err := reconciler.validateSpec(ctx, dgdr)
			Expect(err).NotTo(HaveOccurred())
		})

		It("Should fail validation when modelName is empty", func() {
			ctx := context.Background()
			dgdr := &nvidiacomv1alpha1.DynamoGraphDeploymentRequest{
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentRequestSpec{
					Backend: BackendVLLM,
					SLA: nvidiacomv1alpha1.SLASpec{
						TTFT: 100,
						ITL:  1500,
					},
				},
			}

			err := reconciler.validateSpec(ctx, dgdr)
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).Should(ContainSubstring("modelName"))
		})

		It("Should fail validation when TTFT is zero", func() {
			ctx := context.Background()
			dgdr := &nvidiacomv1alpha1.DynamoGraphDeploymentRequest{
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentRequestSpec{
					ModelName: "test-model",
					Backend:   BackendVLLM,
					SLA: nvidiacomv1alpha1.SLASpec{
						TTFT: 0,
						ITL:  1500,
						ISL:  3000,
						OSL:  500,
					},
				},
			}

			err := reconciler.validateSpec(ctx, dgdr)
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).Should(ContainSubstring("ttft"))
		})

		It("Should fail validation when TTFT is negative", func() {
			ctx := context.Background()
			dgdr := &nvidiacomv1alpha1.DynamoGraphDeploymentRequest{
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentRequestSpec{
					ModelName: "test-model",
					Backend:   BackendVLLM,
					SLA: nvidiacomv1alpha1.SLASpec{
						TTFT: -1,
						ITL:  1500,
					},
				},
			}

			err := reconciler.validateSpec(ctx, dgdr)
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).Should(ContainSubstring("ttft"))
		})

		It("Should fail validation when ITL is zero", func() {
			ctx := context.Background()
			dgdr := &nvidiacomv1alpha1.DynamoGraphDeploymentRequest{
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentRequestSpec{
					ModelName: "test-model",
					Backend:   BackendVLLM,
					SLA: nvidiacomv1alpha1.SLASpec{
						TTFT: 100,
						ITL:  0,
						ISL:  3000,
						OSL:  500,
					},
				},
			}

			err := reconciler.validateSpec(ctx, dgdr)
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).Should(ContainSubstring("itl"))
		})

		It("Should fail validation when ITL is negative", func() {
			ctx := context.Background()
			dgdr := &nvidiacomv1alpha1.DynamoGraphDeploymentRequest{
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentRequestSpec{
					ModelName: "test-model",
					Backend:   BackendVLLM,
					SLA: nvidiacomv1alpha1.SLASpec{
						TTFT: 100,
						ITL:  -1,
					},
				},
			}

			err := reconciler.validateSpec(ctx, dgdr)
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).Should(ContainSubstring("itl"))
		})

		It("Should fail validation for invalid backend", func() {
			ctx := context.Background()
			dgdr := &nvidiacomv1alpha1.DynamoGraphDeploymentRequest{
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentRequestSpec{
					ModelName: "test-model",
					Backend:   "invalid-backend",
					SLA: nvidiacomv1alpha1.SLASpec{
						TTFT: 100,
						ITL:  1500,
					},
				},
			}

			err := reconciler.validateSpec(ctx, dgdr)
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).Should(ContainSubstring("invalid backend"))
		})
	})
})
