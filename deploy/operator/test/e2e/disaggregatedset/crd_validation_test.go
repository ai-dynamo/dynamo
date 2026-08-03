/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

package disaggregatedset

import (
	"strings"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/controller-runtime/pkg/client"
	disaggregatedsetv1 "sigs.k8s.io/lws/api/disaggregatedset/v1"
	leaderworkersetv1 "sigs.k8s.io/lws/api/leaderworkerset/v1"
)

var _ = Describe("LWS v0.9.0 API contract", Label("validation", "e2e", "k8s", "gpu_0"), func() {
	DescribeTable(
		"serves the expected namespaced v1 CRDs with status subresources",
		func(name, group, plural string) {
			crd := &apiextensionsv1.CustomResourceDefinition{}
			Expect(k8sClient.Get(ctx, client.ObjectKey{Name: name}, crd)).To(Succeed())
			Expect(crd.Spec.Group).To(Equal(group))
			Expect(crd.Spec.Scope).To(Equal(apiextensionsv1.NamespaceScoped))
			Expect(crd.Spec.Names.Plural).To(Equal(plural))

			storageVersions := 0
			for _, version := range crd.Spec.Versions {
				if version.Storage {
					storageVersions++
				}
				if version.Name != "v1" {
					continue
				}
				Expect(version.Served).To(BeTrue())
				Expect(version.Storage).To(BeTrue())
				Expect(version.Subresources).NotTo(BeNil())
				Expect(version.Subresources.Status).NotTo(BeNil())
			}
			Expect(storageVersions).To(Equal(1))
		},
		Entry(
			"DisaggregatedSet",
			"disaggregatedsets.disaggregatedset.x-k8s.io",
			"disaggregatedset.x-k8s.io",
			"disaggregatedsets",
		),
		Entry(
			"LeaderWorkerSet",
			"leaderworkersets.leaderworkerset.x-k8s.io",
			"leaderworkerset.x-k8s.io",
			"leaderworkersets",
		),
	)

	It("rejects a DisaggregatedSet with fewer than two roles", func() {
		invalid := &disaggregatedsetv1.DisaggregatedSet{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "ds-e2e-invalid-one-role",
				Namespace: flagNamespace,
			},
			Spec: disaggregatedsetv1.DisaggregatedSetSpec{
				Roles: []disaggregatedsetv1.DisaggregatedRoleSpec{{
					Name: "prefill",
					LeaderWorkerSetTemplateSpec: leaderworkersetv1.LeaderWorkerSetTemplateSpec{
						Spec: leaderworkersetv1.LeaderWorkerSetSpec{},
					},
				}},
			},
		}

		err := k8sClient.Create(ctx, invalid, client.DryRunAll)
		Expect(err).To(HaveOccurred())
		Expect(strings.ToLower(err.Error())).To(ContainSubstring("roles"))
	})
})
