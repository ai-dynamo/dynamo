/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
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
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	commoncontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

var _ = Describe("DisaggregatedSet sync", func() {
	It("creates and updates DisaggregatedSets through the Kubernetes API", func() {
		ns := &corev1.Namespace{ObjectMeta: metav1.ObjectMeta{GenerateName: "ds-sync-"}}
		Expect(k8sClient.Create(ctx, ns)).To(Succeed())
		DeferCleanup(func() {
			Expect(client.IgnoreNotFound(k8sClient.Delete(ctx, ns))).To(Succeed())
		})

		parent := &nvidiacomv1beta1.DynamoGraphDeployment{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "ds-envtest",
				Namespace: ns.Name,
			},
		}

		reconciler := &DynamoGraphDeploymentReconciler{Client: k8sClient}
		desired := envtestDisaggregatedSet(ns.Name, "ds-envtest", 0, 0)
		modified, synced, err := reconciler.syncDisaggregatedSet(ctx, parent, desired)
		Expect(err).NotTo(HaveOccurred())
		Expect(modified).To(BeTrue())
		Expect(synced.GetName()).To(Equal("ds-envtest"))

		fetched := newDisaggregatedSetObject()
		Expect(k8sClient.Get(ctx, types.NamespacedName{Name: "ds-envtest", Namespace: ns.Name}, fetched)).To(Succeed())
		roles, found, err := unstructured.NestedSlice(fetched.Object, "spec", "roles")
		Expect(err).NotTo(HaveOccurred())
		Expect(found).To(BeTrue())
		Expect(roles).To(HaveLen(2))
		Expect(fetched.GetAnnotations()).To(HaveKey(commoncontroller.NvidiaAnnotationHashKey))
		Expect(fetched.GetAnnotations()).To(HaveKey(commoncontroller.NvidiaAnnotationGenerationKey))

		modified, _, err = reconciler.syncDisaggregatedSet(ctx, parent, envtestDisaggregatedSet(ns.Name, "ds-envtest", 1, 1))
		Expect(err).NotTo(HaveOccurred())
		Expect(modified).To(BeTrue())

		invalid := envtestDisaggregatedSet(ns.Name, "ds-envtest-invalid", 0, 1)
		err = k8sClient.Create(ctx, invalid)
		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("replicas must be zero"))
	})
})

func envtestDisaggregatedSet(namespace, name string, prefillReplicas, decodeReplicas int64) *unstructured.Unstructured {
	ds := newDisaggregatedSetObject()
	ds.SetNamespace(namespace)
	ds.SetName(name)
	ds.SetLabels(map[string]string{
		consts.KubeLabelDynamoGraphDeploymentName: name,
	})
	ds.Object["spec"] = map[string]any{
		"roles": []any{
			envtestDisaggregatedSetRole("prefill", prefillReplicas),
			envtestDisaggregatedSetRole("decode", decodeReplicas),
		},
	}
	return ds
}

func envtestDisaggregatedSetRole(name string, replicas int64) map[string]any {
	return map[string]any{
		"name":     name,
		"metadata": map[string]any{"labels": map[string]any{"role": name}},
		"spec": map[string]any{
			"replicas": replicas,
			"leaderWorkerTemplate": map[string]any{
				"size": int64(2),
			},
		},
	}
}
