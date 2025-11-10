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

	"github.com/ai-dynamo/dynamo/deploy/cloud/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/cloud/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/cloud/operator/internal/dynamo"
	"github.com/ai-dynamo/dynamo/deploy/cloud/operator/internal/modelendpoint"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	corev1 "k8s.io/api/core/v1"
	discoveryv1 "k8s.io/api/discovery/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/tools/record"
	"sigs.k8s.io/controller-runtime/pkg/reconcile"
)

var _ = Describe("DynamoModel Controller", func() {
	const (
		timeout  = time.Second * 10
		interval = time.Millisecond * 250
	)

	var (
		reconciler *DynamoModelReconciler
		recorder   *record.FakeRecorder
	)

	BeforeEach(func() {
		recorder = record.NewFakeRecorder(100)
		reconciler = &DynamoModelReconciler{
			Client:         k8sClient,
			Recorder:       recorder,
			EndpointClient: modelendpoint.NewClient(),
		}
	})

	Context("When reconciling LoRA model", func() {
		It("Should discover endpoints and set conditions", func() {
			ctx := context.Background()
			namespace := defaultNamespace
			modelName := "test-lora-model"
			baseModelName := "base-model-lora"

			// Create the DynamoModel
			model := &v1alpha1.DynamoModel{
				ObjectMeta: metav1.ObjectMeta{
					Name:      modelName,
					Namespace: namespace,
				},
				Spec: v1alpha1.DynamoModelSpec{
					ModelName:     modelName,
					BaseModelName: baseModelName,
					ModelType:     "lora",
					Source: &v1alpha1.ModelSource{
						URI: "s3://bucket/model",
					},
				},
			}
			Expect(k8sClient.Create(ctx, model)).Should(Succeed())
			defer func() { _ = k8sClient.Delete(ctx, model) }()

			// Create EndpointSlice with ready Pod endpoints
			trueVal := true
			modelHash := dynamo.HashModelName(baseModelName)
			endpointSlice := &discoveryv1.EndpointSlice{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-lora-endpoints",
					Namespace: namespace,
					Labels: map[string]string{
						consts.KubeLabelDynamoBaseModelHash: modelHash,
						discoveryv1.LabelServiceName:        "test-service",
					},
				},
				AddressType: discoveryv1.AddressTypeIPv4,
				Endpoints: []discoveryv1.Endpoint{
					{
						Addresses: []string{"10.0.1.5"},
						Conditions: discoveryv1.EndpointConditions{
							Ready: &trueVal,
						},
						TargetRef: &corev1.ObjectReference{
							Kind: "Pod",
							Name: "worker-0",
						},
					},
					{
						Addresses: []string{"10.0.1.6"},
						Conditions: discoveryv1.EndpointConditions{
							Ready: &trueVal,
						},
						TargetRef: &corev1.ObjectReference{
							Kind: "Pod",
							Name: "worker-1",
						},
					},
				},
				Ports: []discoveryv1.EndpointPort{
					{
						Port: func() *int32 { p := int32(9090); return &p }(),
					},
				},
			}
			Expect(k8sClient.Create(ctx, endpointSlice)).Should(Succeed())
			defer func() { _ = k8sClient.Delete(ctx, endpointSlice) }()

			// Reconcile
			_, err := reconciler.Reconcile(ctx, reconcile.Request{
				NamespacedName: types.NamespacedName{
					Name:      modelName,
					Namespace: namespace,
				},
			})
			Expect(err).NotTo(HaveOccurred())

			// Verify endpoints were discovered
			Eventually(func() int {
				var updated v1alpha1.DynamoModel
				_ = k8sClient.Get(ctx, types.NamespacedName{Name: modelName, Namespace: namespace}, &updated)
				return updated.Status.TotalEndpoints
			}, timeout, interval).Should(Equal(2))

			// Verify condition is set (will be False since LoRA load will fail without real service)
			Eventually(func() bool {
				var updated v1alpha1.DynamoModel
				_ = k8sClient.Get(ctx, types.NamespacedName{Name: modelName, Namespace: namespace}, &updated)
				for _, cond := range updated.Status.Conditions {
					if cond.Type == ConditionTypeEndpointsReady {
						return true
					}
				}
				return false
			}, timeout, interval).Should(BeTrue())
		})
	})

	Context("When reconciling with non-Pod endpoints", func() {
		It("Should skip endpoints without Pod TargetRef", func() {
			ctx := context.Background()
			namespace := defaultNamespace
			modelName := "test-non-pod-model"
			baseModelName := "base-model-non-pod"

			// Create the DynamoModel
			model := &v1alpha1.DynamoModel{
				ObjectMeta: metav1.ObjectMeta{
					Name:      modelName,
					Namespace: namespace,
				},
				Spec: v1alpha1.DynamoModelSpec{
					ModelName:     modelName,
					BaseModelName: baseModelName,
					ModelType:     "base",
				},
			}
			Expect(k8sClient.Create(ctx, model)).Should(Succeed())
			defer func() { _ = k8sClient.Delete(ctx, model) }()

			// Create EndpointSlice with mixed endpoints (some Pod, some not)
			trueVal := true
			modelHash := dynamo.HashModelName(baseModelName)
			endpointSlice := &discoveryv1.EndpointSlice{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-mixed-endpoints",
					Namespace: namespace,
					Labels: map[string]string{
						consts.KubeLabelDynamoBaseModelHash: modelHash,
					},
				},
				AddressType: discoveryv1.AddressTypeIPv4,
				Endpoints: []discoveryv1.Endpoint{
					{
						Addresses: []string{"10.0.1.7"},
						Conditions: discoveryv1.EndpointConditions{
							Ready: &trueVal,
						},
						TargetRef: &corev1.ObjectReference{
							Kind: "Pod",
							Name: "worker-0",
						},
					},
					{
						Addresses: []string{"10.0.1.8"},
						Conditions: discoveryv1.EndpointConditions{
							Ready: &trueVal,
						},
						TargetRef: &corev1.ObjectReference{
							Kind: "Node", // Not a Pod - should be skipped
							Name: "node-1",
						},
					},
					{
						Addresses: []string{"10.0.1.9"},
						Conditions: discoveryv1.EndpointConditions{
							Ready: &trueVal,
						},
						TargetRef: nil, // Nil TargetRef - should be skipped
					},
				},
				Ports: []discoveryv1.EndpointPort{
					{
						Port: func() *int32 { p := int32(9090); return &p }(),
					},
				},
			}
			Expect(k8sClient.Create(ctx, endpointSlice)).Should(Succeed())
			defer func() { _ = k8sClient.Delete(ctx, endpointSlice) }()

			// Reconcile
			_, err := reconciler.Reconcile(ctx, reconcile.Request{
				NamespacedName: types.NamespacedName{
					Name:      modelName,
					Namespace: namespace,
				},
			})
			Expect(err).NotTo(HaveOccurred())

			// Should only discover 1 endpoint (the Pod), not the Node or nil TargetRef
			Eventually(func() int {
				var updated v1alpha1.DynamoModel
				_ = k8sClient.Get(ctx, types.NamespacedName{Name: modelName, Namespace: namespace}, &updated)
				return updated.Status.TotalEndpoints
			}, timeout, interval).Should(Equal(1))

			// Verify only the Pod endpoint was included
			Eventually(func() string {
				var updated v1alpha1.DynamoModel
				_ = k8sClient.Get(ctx, types.NamespacedName{Name: modelName, Namespace: namespace}, &updated)
				if len(updated.Status.Endpoints) > 0 {
					return updated.Status.Endpoints[0].PodName
				}
				return ""
			}, timeout, interval).Should(Equal("worker-0"))
		})
	})

	Context("When reconciling base model", func() {
		It("Should set EndpointsReady=True when endpoints exist", func() {
			ctx := context.Background()
			namespace := defaultNamespace
			modelName := "test-base-model"
			baseModelName := "base-model-base"

			// Create the DynamoModel
			model := &v1alpha1.DynamoModel{
				ObjectMeta: metav1.ObjectMeta{
					Name:      modelName,
					Namespace: namespace,
				},
				Spec: v1alpha1.DynamoModelSpec{
					ModelName:     modelName,
					BaseModelName: baseModelName,
					ModelType:     "base",
				},
			}
			Expect(k8sClient.Create(ctx, model)).Should(Succeed())
			defer func() { _ = k8sClient.Delete(ctx, model) }()

			// Create EndpointSlice
			trueVal := true
			modelHash := dynamo.HashModelName(baseModelName)
			endpointSlice := &discoveryv1.EndpointSlice{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-base-endpoints",
					Namespace: namespace,
					Labels: map[string]string{
						consts.KubeLabelDynamoBaseModelHash: modelHash,
					},
				},
				AddressType: discoveryv1.AddressTypeIPv4,
				Endpoints: []discoveryv1.Endpoint{
					{
						Addresses: []string{"10.0.1.10"},
						Conditions: discoveryv1.EndpointConditions{
							Ready: &trueVal,
						},
						TargetRef: &corev1.ObjectReference{
							Kind: "Pod",
							Name: "worker-0",
						},
					},
				},
				Ports: []discoveryv1.EndpointPort{
					{
						Port: func() *int32 { p := int32(9090); return &p }(),
					},
				},
			}
			Expect(k8sClient.Create(ctx, endpointSlice)).Should(Succeed())
			defer func() { _ = k8sClient.Delete(ctx, endpointSlice) }()

			// Reconcile
			_, err := reconciler.Reconcile(ctx, reconcile.Request{
				NamespacedName: types.NamespacedName{
					Name:      modelName,
					Namespace: namespace,
				},
			})
			Expect(err).NotTo(HaveOccurred())

			// For base models, EndpointsReady should be True when endpoints exist
			Eventually(func() bool {
				var updated v1alpha1.DynamoModel
				_ = k8sClient.Get(ctx, types.NamespacedName{Name: modelName, Namespace: namespace}, &updated)
				for _, cond := range updated.Status.Conditions {
					if cond.Type == ConditionTypeEndpointsReady {
						return cond.Status == metav1.ConditionTrue && cond.Reason == ReasonEndpointsDiscovered
					}
				}
				return false
			}, timeout, interval).Should(BeTrue())
		})

		It("Should set EndpointsReady=False when no endpoints exist", func() {
			ctx := context.Background()
			namespace := defaultNamespace
			modelName := "test-base-model-no-endpoints"
			baseModelName := "base-model-none"

			// Create the DynamoModel
			model := &v1alpha1.DynamoModel{
				ObjectMeta: metav1.ObjectMeta{
					Name:      modelName,
					Namespace: namespace,
				},
				Spec: v1alpha1.DynamoModelSpec{
					ModelName:     modelName,
					BaseModelName: baseModelName,
					ModelType:     "base",
				},
			}
			Expect(k8sClient.Create(ctx, model)).Should(Succeed())
			defer func() { _ = k8sClient.Delete(ctx, model) }()

			// Reconcile (no endpoints created)
			_, err := reconciler.Reconcile(ctx, reconcile.Request{
				NamespacedName: types.NamespacedName{
					Name:      modelName,
					Namespace: namespace,
				},
			})
			Expect(err).NotTo(HaveOccurred())

			// Should have condition set to False with NoEndpoints reason
			Eventually(func() bool {
				var updated v1alpha1.DynamoModel
				_ = k8sClient.Get(ctx, types.NamespacedName{Name: modelName, Namespace: namespace}, &updated)
				for _, cond := range updated.Status.Conditions {
					if cond.Type == ConditionTypeEndpointsReady {
						return cond.Status == metav1.ConditionFalse && cond.Reason == ReasonNoEndpoints
					}
				}
				return false
			}, timeout, interval).Should(BeTrue())
		})
	})
})
