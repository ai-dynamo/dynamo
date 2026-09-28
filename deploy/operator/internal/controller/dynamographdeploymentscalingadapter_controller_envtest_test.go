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

package controller

import (
	"context"
	"time"

	corev1 "k8s.io/api/core/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	"k8s.io/client-go/tools/events"
	"sigs.k8s.io/controller-runtime/pkg/envtest"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

const (
	adapterNameFrontend   = "test-dgd-frontend"
	dgdName               = "test-dgd"
	componentNameFrontend = "Frontend"
)

var _ = Describe("DynamoGraphDeploymentScalingAdapter Controller", func() {
	const timeout = time.Second * 10
	const interval = time.Millisecond * 250

	var (
		ctx            context.Context
		namespace      string
		reconciler     *DynamoGraphDeploymentScalingAdapterReconciler
		recorder       *events.FakeRecorder
		operatorClient client.Client
	)

	BeforeEach(func() {
		ctx = context.Background()
		env := sharedEnv.ForTest(GinkgoTB())
		namespace = env.Namespace()
		k8sClient := env.Client()

		// Create a client that impersonates the operator service account for SSA authorization
		operatorConfig, err := env.AddUser(envtest.User{
			Name:   testOperatorPrincipal,
			Groups: []string{"system:serviceaccounts:dynamo-system"},
		})
		Expect(err).NotTo(HaveOccurred())
		operatorClient, err = client.New(operatorConfig, client.Options{Scheme: k8sClient.Scheme()})
		Expect(err).NotTo(HaveOccurred())

		// Grant the operator service account permissions to access DGDSA resources
		role := &rbacv1.Role{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "operator-dgdsa-access",
				Namespace: namespace,
			},
			Rules: []rbacv1.PolicyRule{
				{
					APIGroups: []string{"nvidia.com"},
					Resources: []string{"dynamographdeploymentscalingadapters", "dynamographdeployments"},
					Verbs:     []string{"get", "list", "watch", "create", "update", "patch"},
				},
				{
					APIGroups: []string{"nvidia.com"},
					Resources: []string{"dynamographdeploymentscalingadapters/status"},
					Verbs:     []string{"get", "update", "patch"},
				},
			},
		}
		Expect(k8sClient.Create(ctx, role)).Should(Succeed())

		roleBinding := &rbacv1.RoleBinding{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "operator-dgdsa-access",
				Namespace: namespace,
			},
			Subjects: []rbacv1.Subject{
				{
					Kind:      "ServiceAccount",
					Name:      "operator",
					Namespace: "dynamo-system",
				},
			},
			RoleRef: rbacv1.RoleRef{
				Kind:     "Role",
				Name:     "operator-dgdsa-access",
				APIGroup: "rbac.authorization.k8s.io",
			},
		}
		Expect(k8sClient.Create(ctx, roleBinding)).Should(Succeed())

		recorder = events.NewFakeRecorder(100)
		reconciler = &DynamoGraphDeploymentScalingAdapterReconciler{
			Client:   operatorClient,
			Scheme:   k8sClient.Scheme(),
			Recorder: recorder,
		}
	})

	Context("when reconciling a scaling adapter", func() {
		It("updates DGD replicas when DGDSA spec differs", func() {
			dgd := &nvidiacomv1beta1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:      dgdName,
					Namespace: namespace,
				},
				Spec: nvidiacomv1beta1.DynamoGraphDeploymentSpec{
					Components: []nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{
						{
							ComponentName:  componentNameFrontend,
							Replicas:       new(int32(2)),
							ScalingAdapter: &nvidiacomv1beta1.ScalingAdapter{},
							PodTemplate: &corev1.PodTemplateSpec{
								Spec: corev1.PodSpec{
									Containers: []corev1.Container{
										{
											Name:  "main",
											Image: "nginx:1.25.0",
										},
									},
								},
							},
							RuntimeVersionOverride: "1.0.0",
						},
					},
				},
			}
			Expect(k8sClient.Create(ctx, dgd)).Should(Succeed())

			adapter := &nvidiacomv1beta1.DynamoGraphDeploymentScalingAdapter{
				ObjectMeta: metav1.ObjectMeta{
					Name:      adapterNameFrontend,
					Namespace: namespace,
				},
				Spec: nvidiacomv1beta1.DynamoGraphDeploymentScalingAdapterSpec{
					Replicas: 5,
					DGDRef: nvidiacomv1beta1.DynamoGraphDeploymentComponentRef{
						Name:          dgdName,
						ComponentName: componentNameFrontend,
					},
				},
			}
			Expect(k8sClient.Create(ctx, adapter)).Should(Succeed())

			By("Reconciling the adapter")
			req := ctrl.Request{
				NamespacedName: client.ObjectKey{Name: adapterNameFrontend, Namespace: namespace},
			}
			_, err := reconciler.Reconcile(ctx, req)
			Expect(err).NotTo(HaveOccurred())

			By("Verifying DGD replicas were updated")
			Eventually(func(g Gomega) {
				updatedDGD := &nvidiacomv1beta1.DynamoGraphDeployment{}
				g.Expect(k8sClient.Get(ctx, client.ObjectKey{Name: dgdName, Namespace: namespace}, updatedDGD)).Should(Succeed())
				component := updatedDGD.GetComponentByName(componentNameFrontend)
				g.Expect(component).NotTo(BeNil())
				g.Expect(component.Replicas).NotTo(BeNil())
				g.Expect(*component.Replicas).To(Equal(int32(5)))
			}, timeout, interval).Should(Succeed())

			By("Verifying adapter status was updated")
			Eventually(func(g Gomega) {
				updatedAdapter := &nvidiacomv1beta1.DynamoGraphDeploymentScalingAdapter{}
				g.Expect(k8sClient.Get(ctx, client.ObjectKey{Name: adapterNameFrontend, Namespace: namespace}, updatedAdapter)).Should(Succeed())
				g.Expect(updatedAdapter.Status.Replicas).To(Equal(int32(5)))
				expectedSelector := "nvidia.com/dynamo-component=" + componentNameFrontend + ",nvidia.com/dynamo-graph-deployment-name=" + dgdName
				g.Expect(updatedAdapter.Status.Selector).To(Equal(expectedSelector))
			}, timeout, interval).Should(Succeed())
		})

		It("does not update when replicas already match", func() {
			dgd := &nvidiacomv1beta1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:      dgdName,
					Namespace: namespace,
				},
				Spec: nvidiacomv1beta1.DynamoGraphDeploymentSpec{
					Components: []nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{
						{
							ComponentName:  componentNameFrontend,
							Replicas:       new(int32(3)),
							ScalingAdapter: &nvidiacomv1beta1.ScalingAdapter{},
							PodTemplate: &corev1.PodTemplateSpec{
								Spec: corev1.PodSpec{
									Containers: []corev1.Container{
										{
											Name:  "main",
											Image: "nginx:1.25.0",
										},
									},
								},
							},
							RuntimeVersionOverride: "1.0.0",
						},
					},
				},
			}
			Expect(k8sClient.Create(ctx, dgd)).Should(Succeed())

			adapter := &nvidiacomv1beta1.DynamoGraphDeploymentScalingAdapter{
				ObjectMeta: metav1.ObjectMeta{
					Name:      adapterNameFrontend,
					Namespace: namespace,
				},
				Spec: nvidiacomv1beta1.DynamoGraphDeploymentScalingAdapterSpec{
					Replicas: 3,
					DGDRef: nvidiacomv1beta1.DynamoGraphDeploymentComponentRef{
						Name:          dgdName,
						ComponentName: componentNameFrontend,
					},
				},
			}
			Expect(k8sClient.Create(ctx, adapter)).Should(Succeed())

			By("Reconciling the adapter")
			req := ctrl.Request{
				NamespacedName: client.ObjectKey{Name: adapterNameFrontend, Namespace: namespace},
			}
			_, err := reconciler.Reconcile(ctx, req)
			Expect(err).NotTo(HaveOccurred())

			By("Verifying DGD replicas remain unchanged")
			Eventually(func(g Gomega) {
				updatedDGD := &nvidiacomv1beta1.DynamoGraphDeployment{}
				g.Expect(k8sClient.Get(ctx, client.ObjectKey{Name: dgdName, Namespace: namespace}, updatedDGD)).Should(Succeed())
				component := updatedDGD.GetComponentByName(componentNameFrontend)
				g.Expect(component).NotTo(BeNil())
				g.Expect(component.Replicas).NotTo(BeNil())
				g.Expect(*component.Replicas).To(Equal(int32(3)))
			}, timeout, interval).Should(Succeed())
		})

		It("uses default replicas when DGD component has no replicas set", func() {
			adapterName := "test-dgd-worker"
			componentName := "worker"

			dgd := &nvidiacomv1beta1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:      dgdName,
					Namespace: namespace,
				},
				Spec: nvidiacomv1beta1.DynamoGraphDeploymentSpec{
					Components: []nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{
						{
							ComponentName:  componentName,
							ScalingAdapter: &nvidiacomv1beta1.ScalingAdapter{},
							PodTemplate: &corev1.PodTemplateSpec{
								Spec: corev1.PodSpec{
									Containers: []corev1.Container{
										{
											Name:  "main",
											Image: "nginx:1.25.0",
										},
									},
								},
							},
							RuntimeVersionOverride: "1.0.0",
						},
					},
				},
			}
			Expect(k8sClient.Create(ctx, dgd)).Should(Succeed())

			adapter := &nvidiacomv1beta1.DynamoGraphDeploymentScalingAdapter{
				ObjectMeta: metav1.ObjectMeta{
					Name:      adapterName,
					Namespace: namespace,
				},
				Spec: nvidiacomv1beta1.DynamoGraphDeploymentScalingAdapterSpec{
					Replicas: 4,
					DGDRef: nvidiacomv1beta1.DynamoGraphDeploymentComponentRef{
						Name:          dgdName,
						ComponentName: componentName,
					},
				},
			}
			Expect(k8sClient.Create(ctx, adapter)).Should(Succeed())

			By("Reconciling the adapter")
			req := ctrl.Request{
				NamespacedName: client.ObjectKey{Name: adapterName, Namespace: namespace},
			}
			_, err := reconciler.Reconcile(ctx, req)
			Expect(err).NotTo(HaveOccurred())

			By("Verifying DGD replicas were updated to desired value")
			Eventually(func(g Gomega) {
				updatedDGD := &nvidiacomv1beta1.DynamoGraphDeployment{}
				g.Expect(k8sClient.Get(ctx, client.ObjectKey{Name: dgdName, Namespace: namespace}, updatedDGD)).Should(Succeed())
				component := updatedDGD.GetComponentByName(componentName)
				g.Expect(component).NotTo(BeNil())
				g.Expect(component.Replicas).NotTo(BeNil())
				g.Expect(*component.Replicas).To(Equal(int32(4)))
			}, timeout, interval).Should(Succeed())
		})

		It("does not propagate replicas after component opts out", func() {
			dgd := &nvidiacomv1beta1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:      dgdName,
					Namespace: namespace,
				},
				Spec: nvidiacomv1beta1.DynamoGraphDeploymentSpec{
					Components: []nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{
						{
							ComponentName: componentNameFrontend,
							Replicas:      new(int32(2)),
							// No ScalingAdapter - component has opted out
							PodTemplate: &corev1.PodTemplateSpec{
								Spec: corev1.PodSpec{
									Containers: []corev1.Container{
										{
											Name:  "main",
											Image: "nginx:1.25.0",
										},
									},
								},
							},
							RuntimeVersionOverride: "1.0.0",
						},
					},
				},
			}
			Expect(k8sClient.Create(ctx, dgd)).Should(Succeed())

			adapter := &nvidiacomv1beta1.DynamoGraphDeploymentScalingAdapter{
				ObjectMeta: metav1.ObjectMeta{
					Name:      adapterNameFrontend,
					Namespace: namespace,
				},
				Spec: nvidiacomv1beta1.DynamoGraphDeploymentScalingAdapterSpec{
					Replicas: 5,
					DGDRef: nvidiacomv1beta1.DynamoGraphDeploymentComponentRef{
						Name:          dgdName,
						ComponentName: componentNameFrontend,
					},
				},
			}
			Expect(k8sClient.Create(ctx, adapter)).Should(Succeed())

			By("Reconciling the adapter")
			req := ctrl.Request{
				NamespacedName: client.ObjectKey{Name: adapterNameFrontend, Namespace: namespace},
			}
			_, err := reconciler.Reconcile(ctx, req)
			Expect(err).NotTo(HaveOccurred())

			By("Verifying DGD replicas were not updated")
			Eventually(func(g Gomega) {
				updatedDGD := &nvidiacomv1beta1.DynamoGraphDeployment{}
				g.Expect(k8sClient.Get(ctx, client.ObjectKey{Name: dgdName, Namespace: namespace}, updatedDGD)).Should(Succeed())
				component := updatedDGD.GetComponentByName(componentNameFrontend)
				g.Expect(component).NotTo(BeNil())
				g.Expect(*component.Replicas).To(Equal(int32(2)))
			}, timeout, interval).Should(Succeed())

			By("Verifying adapter status was not updated")
			Eventually(func(g Gomega) {
				updatedAdapter := &nvidiacomv1beta1.DynamoGraphDeploymentScalingAdapter{}
				g.Expect(k8sClient.Get(ctx, client.ObjectKey{Name: adapterNameFrontend, Namespace: namespace}, updatedAdapter)).Should(Succeed())
				g.Expect(updatedAdapter.Status.Selector).To(BeEmpty())
				g.Expect(updatedAdapter.Status.Replicas).To(BeZero())
			}, timeout, interval).Should(Succeed())
		})

		It("returns without retry when component not found in DGD", func() {
			adapterName := "test-dgd-missing"

			dgd := &nvidiacomv1beta1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:      dgdName,
					Namespace: namespace,
				},
				Spec: nvidiacomv1beta1.DynamoGraphDeploymentSpec{
					Components: []nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{
						{
							ComponentName: componentNameFrontend,
							Replicas:      new(int32(1)),
							PodTemplate: &corev1.PodTemplateSpec{
								Spec: corev1.PodSpec{
									Containers: []corev1.Container{
										{
											Name:  "main",
											Image: "nginx:1.25.0",
										},
									},
								},
							},
							RuntimeVersionOverride: "1.0.0",
						},
					},
				},
			}
			Expect(k8sClient.Create(ctx, dgd)).Should(Succeed())

			adapter := &nvidiacomv1beta1.DynamoGraphDeploymentScalingAdapter{
				ObjectMeta: metav1.ObjectMeta{
					Name:      adapterName,
					Namespace: namespace,
				},
				Spec: nvidiacomv1beta1.DynamoGraphDeploymentScalingAdapterSpec{
					Replicas: 2,
					DGDRef: nvidiacomv1beta1.DynamoGraphDeploymentComponentRef{
						Name:          dgdName,
						ComponentName: "nonexistent",
					},
				},
			}
			Expect(k8sClient.Create(ctx, adapter)).Should(Succeed())

			By("Reconciling the adapter")
			req := ctrl.Request{
				NamespacedName: client.ObjectKey{Name: adapterName, Namespace: namespace},
			}
			result, err := reconciler.Reconcile(ctx, req)
			Expect(err).NotTo(HaveOccurred())
			Expect(result.RequeueAfter).To(Equal(time.Duration(0)))

			By("Verifying adapter status has empty selector")
			updatedAdapter := &nvidiacomv1beta1.DynamoGraphDeploymentScalingAdapter{}
			Expect(k8sClient.Get(ctx, client.ObjectKey{Name: adapterName, Namespace: namespace}, updatedAdapter)).Should(Succeed())
			Expect(updatedAdapter.Status.Selector).To(BeEmpty())
		})

		It("returns no error when adapter not found", func() {
			By("Reconciling a non-existent adapter")
			req := ctrl.Request{
				NamespacedName: client.ObjectKey{Name: "nonexistent", Namespace: namespace},
			}
			result, err := reconciler.Reconcile(ctx, req)
			Expect(err).NotTo(HaveOccurred())
			Expect(result.RequeueAfter).To(Equal(time.Duration(0)))
		})

		It("returns error when referenced DGD not found", func() {
			adapter := &nvidiacomv1beta1.DynamoGraphDeploymentScalingAdapter{
				ObjectMeta: metav1.ObjectMeta{
					Name:      adapterNameFrontend,
					Namespace: namespace,
				},
				Spec: nvidiacomv1beta1.DynamoGraphDeploymentScalingAdapterSpec{
					Replicas: 5,
					DGDRef: nvidiacomv1beta1.DynamoGraphDeploymentComponentRef{
						Name:          "nonexistent-dgd",
						ComponentName: componentNameFrontend,
					},
				},
			}
			Expect(k8sClient.Create(ctx, adapter)).Should(Succeed())

			By("Reconciling the adapter with missing DGD")
			req := ctrl.Request{
				NamespacedName: client.ObjectKey{Name: adapterNameFrontend, Namespace: namespace},
			}
			_, err := reconciler.Reconcile(ctx, req)
			Expect(err).To(HaveOccurred())
		})

		It("skips reconciliation when adapter is being deleted", func() {
			dgd := &nvidiacomv1beta1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:      dgdName,
					Namespace: namespace,
				},
				Spec: nvidiacomv1beta1.DynamoGraphDeploymentSpec{
					Components: []nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{
						{
							ComponentName: componentNameFrontend,
							Replicas:      new(int32(2)),
							PodTemplate: &corev1.PodTemplateSpec{
								Spec: corev1.PodSpec{
									Containers: []corev1.Container{
										{
											Name:  "main",
											Image: "nginx:1.25.0",
										},
									},
								},
							},
							RuntimeVersionOverride: "1.0.0",
						},
					},
				},
			}
			Expect(k8sClient.Create(ctx, dgd)).Should(Succeed())

			now := metav1.Now()
			adapter := &nvidiacomv1beta1.DynamoGraphDeploymentScalingAdapter{
				ObjectMeta: metav1.ObjectMeta{
					Name:              adapterNameFrontend,
					Namespace:         namespace,
					DeletionTimestamp: &now,
					Finalizers:        []string{"test-finalizer"},
				},
				Spec: nvidiacomv1beta1.DynamoGraphDeploymentScalingAdapterSpec{
					Replicas: 5,
					DGDRef: nvidiacomv1beta1.DynamoGraphDeploymentComponentRef{
						Name:          dgdName,
						ComponentName: componentNameFrontend,
					},
				},
			}
			Expect(k8sClient.Create(ctx, adapter)).Should(Succeed())

			By("Reconciling the deleting adapter")
			req := ctrl.Request{
				NamespacedName: client.ObjectKey{Name: adapterNameFrontend, Namespace: namespace},
			}
			result, err := reconciler.Reconcile(ctx, req)
			Expect(err).NotTo(HaveOccurred())
			Expect(result.RequeueAfter).To(Equal(time.Duration(0)))

			By("Verifying DGD replicas remain unchanged")
			updatedDGD := &nvidiacomv1beta1.DynamoGraphDeployment{}
			Expect(k8sClient.Get(ctx, client.ObjectKey{Name: dgdName, Namespace: namespace}, updatedDGD)).Should(Succeed())
			component := updatedDGD.GetComponentByName(componentNameFrontend)
			Expect(component).NotTo(BeNil())
			Expect(*component.Replicas).To(Equal(int32(2)))
		})
	})

	Context("when mapping DGD changes to adapters", func() {
		It("finds all adapters referencing the DGD", func() {
			// Create DGD
			dgd := &nvidiacomv1beta1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:      dgdName,
					Namespace: namespace,
				},
				Spec: nvidiacomv1beta1.DynamoGraphDeploymentSpec{
					Components: []nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{
						{
							ComponentName: componentNameFrontend,
							Replicas:      new(int32(1)),
							PodTemplate: &corev1.PodTemplateSpec{
								Spec: corev1.PodSpec{
									Containers: []corev1.Container{
										{
											Name:  "main",
											Image: "nginx:1.25.0",
										},
									},
								},
							},
							RuntimeVersionOverride: "1.0.0",
						},
					},
				},
			}
			Expect(k8sClient.Create(ctx, dgd)).Should(Succeed())

			// Adapters belonging to test-dgd
			adapter1 := &nvidiacomv1beta1.DynamoGraphDeploymentScalingAdapter{
				ObjectMeta: metav1.ObjectMeta{
					Name:      adapterNameFrontend,
					Namespace: namespace,
				},
				Spec: nvidiacomv1beta1.DynamoGraphDeploymentScalingAdapterSpec{
					DGDRef: nvidiacomv1beta1.DynamoGraphDeploymentComponentRef{
						Name:          dgdName,
						ComponentName: componentNameFrontend,
					},
				},
			}

			adapter2 := &nvidiacomv1beta1.DynamoGraphDeploymentScalingAdapter{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-dgd-decode",
					Namespace: namespace,
				},
				Spec: nvidiacomv1beta1.DynamoGraphDeploymentScalingAdapterSpec{
					DGDRef: nvidiacomv1beta1.DynamoGraphDeploymentComponentRef{
						Name:          dgdName,
						ComponentName: "decode",
					},
				},
			}

			// Adapter belonging to different DGD
			adapterOther := &nvidiacomv1beta1.DynamoGraphDeploymentScalingAdapter{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "other-dgd-frontend",
					Namespace: namespace,
				},
				Spec: nvidiacomv1beta1.DynamoGraphDeploymentScalingAdapterSpec{
					DGDRef: nvidiacomv1beta1.DynamoGraphDeploymentComponentRef{
						Name:          "other-dgd",
						ComponentName: componentNameFrontend,
					},
				},
			}

			Expect(k8sClient.Create(ctx, adapter1)).Should(Succeed())
			Expect(k8sClient.Create(ctx, adapter2)).Should(Succeed())
			Expect(k8sClient.Create(ctx, adapterOther)).Should(Succeed())

			By("Mapping DGD to adapter reconcile requests")
			requests := reconciler.findAdaptersForDGD(ctx, dgd)

			// Should return 2 requests (for test-dgd adapters only)
			Expect(requests).To(HaveLen(2))

			// Verify correct adapters are returned
			expectedNames := map[string]bool{
				adapterNameFrontend: true,
				"test-dgd-decode":   true,
			}

			for _, req := range requests {
				Expect(expectedNames).To(HaveKey(req.Name))
			}
		})
	})
})
