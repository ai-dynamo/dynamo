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

package controller

import (
	"fmt"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
)

var _ = Describe("DynamoComponentDeployment API validation", func() {
	It("rejects an EPP component without configuration", func() {
		dcd := &nvidiacomv1beta1.DynamoComponentDeployment{
			ObjectMeta: metav1.ObjectMeta{
				Name:      fmt.Sprintf("invalid-epp-%d", GinkgoRandomSeed()),
				Namespace: envtestNamespace,
			},
			Spec: nvidiacomv1beta1.DynamoComponentDeploymentSpec{
				BackendFramework: "vllm",
				DynamoComponentDeploymentSharedSpec: nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{
					ComponentType: nvidiacomv1beta1.ComponentTypeEPP,
				},
			},
		}

		err := k8sClient.Create(ctx, dcd)
		Expect(err).To(MatchError(ContainSubstring("spec.eppConfig: Required value: is required for EPP components")))
	})

	It("converts a v1alpha1 DCD to v1beta1", func() {
		name := fmt.Sprintf("conversion-dcd-%d", GinkgoRandomSeed())
		alpha := &nvidiacomv1alpha1.DynamoComponentDeployment{
			ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: envtestNamespace},
			Spec: nvidiacomv1alpha1.DynamoComponentDeploymentSpec{
				BackendFramework: "vllm",
				DynamoComponentDeploymentSharedSpec: nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
					ServiceName:   "worker",
					ComponentType: consts.ComponentTypeWorker,
				},
			},
		}
		Expect(k8sClient.Create(ctx, alpha)).To(Succeed())
		DeferCleanup(func() {
			err := k8sClient.Delete(ctx, alpha)
			if err != nil && !apierrors.IsNotFound(err) {
				Expect(err).NotTo(HaveOccurred())
			}
		})

		var beta nvidiacomv1beta1.DynamoComponentDeployment
		key := types.NamespacedName{Name: name, Namespace: envtestNamespace}
		Expect(k8sClient.Get(ctx, key, &beta)).To(Succeed())

		Expect(beta.Spec.BackendFramework).To(Equal("vllm"))
		Expect(beta.Spec.ComponentName).To(Equal("worker"))
		Expect(beta.Spec.ComponentType).To(Equal(nvidiacomv1beta1.ComponentTypeWorker))
	})
})
