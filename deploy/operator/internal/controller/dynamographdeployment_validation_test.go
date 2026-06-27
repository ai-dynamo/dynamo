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

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/utils/ptr"
)

var _ = Describe("DynamoGraphDeployment API validation", func() {
	const restartOnCreateErr = "spec.restart must be unset on create"

	newDGD := func(name string) *nvidiacomv1beta1.DynamoGraphDeployment {
		return &nvidiacomv1beta1.DynamoGraphDeployment{
			ObjectMeta: metav1.ObjectMeta{
				Name:      name,
				Namespace: "default",
			},
			Spec: nvidiacomv1beta1.DynamoGraphDeploymentSpec{
				Components: []nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{
					{
						ComponentName: "frontend",
						Replicas:      ptr.To(int32(1)),
					},
				},
			},
		}
	}

	It("rejects spec.restart on create", func() {
		dgd := newDGD(fmt.Sprintf("restart-on-create-%d", GinkgoRandomSeed()))
		dgd.Spec.Restart = &nvidiacomv1beta1.Restart{ID: "restart-1"}

		err := k8sClient.Create(ctx, dgd)
		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring(restartOnCreateErr))
	})

	It("allows spec.restart to be set after create", func() {
		dgd := newDGD(fmt.Sprintf("restart-on-update-%d", GinkgoRandomSeed()))
		Expect(k8sClient.Create(ctx, dgd)).To(Succeed())
		DeferCleanup(func() {
			err := k8sClient.Delete(ctx, dgd)
			if err != nil && !apierrors.IsNotFound(err) {
				Expect(err).NotTo(HaveOccurred())
			}
		})

		dgd.Spec.Restart = &nvidiacomv1beta1.Restart{ID: "restart-1"}
		Expect(k8sClient.Update(ctx, dgd)).To(Succeed())
	})
})
