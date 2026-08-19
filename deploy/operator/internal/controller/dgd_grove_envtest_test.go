//go:build !clustertest

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
	"context"
	"testing"
	"time"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	commoncontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/testing/operatorenv"
	"github.com/onsi/gomega"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/utils/ptr"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

// groveEnv is an isolated envtest whose gates have Grove on.
//
// It cannot share sharedEnv: the defaulting webhook resolves a DGD's workload provider at
// Create from the gates its own manager was started with, and sharedEnv starts from
// features.Defaults(), where Grove is off. A DGD admitted there is stamped
// "workload-provider: component" no matter what gates the test's own controller carries,
// so the Grove pathway is unreachable. RunT gives this file its own API server with the
// gate flipped, without changing what every other test in the package sees.
var groveEnv = operatorenv.New(operatorenv.Options{
	Admission: operatorenv.AdmissionWebhooks{Mutating: true, Validating: true},
	RuntimeConfig: &commoncontroller.RuntimeConfig{
		Gate: features.Gates{Grove: true, GPUDiscovery: true},
	},
	SetupWebhooks: setupProductionWebhooks,
})

// TestGroveDGDRecreatesAnOwnedServiceAfterDeletion covers the DGD controller's
// Owns(&corev1.Service{}) watch on the Grove pathway, where the DGD owns the Services its
// stable-resources reconciler emits. Without the watch a deleted Service stays absent
// until an unrelated watched resource happens to trigger a reconcile.
func TestGroveDGDRecreatesAnOwnedServiceAfterDeletion(t *testing.T) {
	t.Log("Start the DynamoGraphDeployment controller on a Grove-enabled envtest")
	ctx := context.Background()
	g := gomega.NewGomegaWithT(t)
	env := groveEnv.RunT(t)
	kubeClient := env.Client()

	// Restricted mode: cluster-wide mode requires an RBAC manager this test does not run,
	// and the reconcile fails before it reaches any Service.
	operatorConfig := env.OperatorConfig()
	operatorConfig.Namespace.Restricted = env.Namespace()
	operatorConfig.GPU.DiscoveryEnabled = ptr.To(false)

	// The Grove scaler drives PodClique replicas through the scale subresource, so the
	// pathway does not reconcile at all without a scale client.
	scaleClient, err := env.ScaleClient()
	g.Expect(err).NotTo(gomega.HaveOccurred())

	env.StartManager(func(mgr ctrl.Manager) error {
		return SetupDynamoGraphDeployment(mgr, DynamoGraphDeploymentSetupOptions{
			SetupOptions: SetupOptions{
				Config:        operatorConfig,
				RuntimeConfig: env.RuntimeConfig(),
			},
			ScaleClient: scaleClient,
		})
	})

	t.Log("Create a DGD that renders through Grove")
	dgd := &v1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "grove-service-watch", Namespace: env.Namespace()},
		Spec: v1beta1.DynamoGraphDeploymentSpec{
			BackendFramework: "vllm",
			Components: []v1beta1.DynamoComponentDeploymentSharedSpec{
				{
					ComponentName: "Frontend",
					ComponentType: v1beta1.ComponentTypeFrontend,
					PodTemplate: &corev1.PodTemplateSpec{
						Spec: corev1.PodSpec{
							Containers: []corev1.Container{
								{Name: "main", Image: "nvcr.io/nvidia/ai-dynamo/vllm-runtime:0.1.0"},
							},
						},
					},
				},
			},
		},
	}
	g.Expect(kubeClient.Create(ctx, dgd)).To(gomega.Succeed())

	t.Log("Verify the DGD really took the Grove pathway, so the watch under test is the one exercised")
	g.Eventually(func() string {
		current := &v1beta1.DynamoGraphDeployment{}
		if err := kubeClient.Get(ctx, client.ObjectKeyFromObject(dgd), current); err != nil {
			return ""
		}
		return current.Annotations["nvidia.com/workload-provider"]
	}, 60*time.Second, 250*time.Millisecond).Should(gomega.Equal("grove"))

	serviceKey := types.NamespacedName{
		Name:      dynamo.NormalizeKubeResourceName(dynamo.GetDCDResourceName(dgd, "Frontend", "")),
		Namespace: env.Namespace(),
	}

	t.Log("Wait for the controller to create the owned Service")
	var originalUID types.UID
	g.Eventually(func() error {
		service := &corev1.Service{}
		if err := kubeClient.Get(ctx, serviceKey, service); err != nil {
			return err
		}
		originalUID = service.UID
		return nil
	}, 90*time.Second, 250*time.Millisecond).Should(gomega.Succeed())

	t.Log("Delete the owned Service out from under the controller")
	g.Expect(kubeClient.Delete(ctx, &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{Name: serviceKey.Name, Namespace: serviceKey.Namespace},
	})).To(gomega.Succeed())

	t.Log("Verify the watch re-enqueued the DGD and the Service came back as a new object")
	g.Eventually(func() (bool, error) {
		service := &corev1.Service{}
		if err := kubeClient.Get(ctx, serviceKey, service); err != nil {
			return false, client.IgnoreNotFound(err)
		}
		return service.UID != "" && service.UID != originalUID, nil
	}, 90*time.Second, 250*time.Millisecond).Should(gomega.BeTrue(),
		"the deleted Service was never recreated, so the owned-Service watch is not enqueuing its DGD")
}
