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

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	commoncontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/epp"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	"github.com/onsi/gomega"
	"github.com/stretchr/testify/require"
	networkingv1beta1 "istio.io/client-go/pkg/apis/networking/v1beta1"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/uuid"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
	gaiev1 "sigs.k8s.io/gateway-api-inference-extension/api/v1"
)

// TestDGDEPPReconciler_deleteOwnedLegacyEPPConfigMap covers the migration
// name-collision scenario: a legacy DGD's eppConfig.configMapRef can name a
// user-managed ConfigMap that happens to collide with the operator's
// deterministic legacy-EPP ConfigMap name (<dgd>-epp-config). Clearing
// eppConfig to migrate to native Rust EPP must delete only a ConfigMap this
// DGD actually generated and owns, never a same-named foreign object.
func TestDGDEPPReconciler_deleteOwnedLegacyEPPConfigMap(t *testing.T) {
	t.Run("deletes the ConfigMap this DGD owns", func(t *testing.T) {
		g := gomega.NewGomegaWithT(t)
		ctx := context.Background()
		scheme := newDynamoGraphDeploymentControllerTestScheme(t)
		dgd := &v1beta1.DynamoGraphDeployment{
			Name: "my-model", Namespace: "default", UID: uuid.NewUUID(),
		}
		owned := &corev1.ConfigMap{
			Name:      epp.GetConfigMapName(dgd.Name),
			Namespace: dgd.Namespace,
		}
		g.Expect(controllerutil.SetControllerReference(dgd, owned, scheme)).To(gomega.Succeed())

		fakeClient := fake.NewClientBuilder().WithScheme(scheme).WithObjects(dgd, owned).Build()
		reconciler := newDGDEPPReconciler(newDGDResourceSyncer(fakeClient, nil), nil, nil, nil)

		g.Expect(reconciler.deleteOwnedLegacyEPPConfigMap(ctx, dgd)).To(gomega.Succeed())

		err := fakeClient.Get(ctx, types.NamespacedName{Name: owned.Name, Namespace: owned.Namespace}, &corev1.ConfigMap{})
		g.Expect(apierrors.IsNotFound(err)).To(gomega.BeTrue(), "operator-owned ConfigMap must be deleted")
	})

	t.Run("leaves a foreign ConfigMap with a colliding name untouched", func(t *testing.T) {
		g := gomega.NewGomegaWithT(t)
		ctx := context.Background()
		scheme := newDynamoGraphDeploymentControllerTestScheme(t)
		dgd := &v1beta1.DynamoGraphDeployment{
			Name: "my-model", Namespace: "default", UID: uuid.NewUUID(),
		}
		// A user's own ConfigMap -- referenced via eppConfig.configMapRef
		// while this DGD was still on legacy Go EPP -- that happens to be
		// named exactly what the operator would have generated
		// (<dgd>-epp-config). It carries no owner reference to this DGD.
		foreign := &corev1.ConfigMap{
			Name:      epp.GetConfigMapName(dgd.Name),
			Namespace: dgd.Namespace,
			Data:      map[string]string{"do-not-delete-me": "user data"},
		}

		fakeClient := fake.NewClientBuilder().WithScheme(scheme).WithObjects(dgd, foreign).Build()
		reconciler := newDGDEPPReconciler(newDGDResourceSyncer(fakeClient, nil), nil, nil, nil)

		g.Expect(reconciler.deleteOwnedLegacyEPPConfigMap(ctx, dgd)).To(gomega.Succeed())

		var got corev1.ConfigMap
		g.Expect(fakeClient.Get(ctx, types.NamespacedName{Name: foreign.Name, Namespace: foreign.Namespace}, &got)).To(gomega.Succeed())
		g.Expect(got.Data).To(gomega.Equal(foreign.Data),
			"a ConfigMap this DGD does not own must never be deleted, even if its name collides with the operator's deterministic legacy-EPP name")
	})

	t.Run("no-op when the ConfigMap is already absent", func(t *testing.T) {
		g := gomega.NewGomegaWithT(t)
		ctx := context.Background()
		scheme := newDynamoGraphDeploymentControllerTestScheme(t)
		dgd := &v1beta1.DynamoGraphDeployment{
			Name: "my-model", Namespace: "default", UID: uuid.NewUUID(),
		}
		fakeClient := fake.NewClientBuilder().WithScheme(scheme).WithObjects(dgd).Build()
		reconciler := newDGDEPPReconciler(newDGDResourceSyncer(fakeClient, nil), nil, nil, nil)

		g.Expect(reconciler.deleteOwnedLegacyEPPConfigMap(ctx, dgd)).To(gomega.Succeed())
	})
}

func TestDGDEPPReconcilerRetainsLegacyChildServiceReference(t *testing.T) {
	t.Log("Seed an existing numeric EPP Service controlled by the graph's child DCD")
	scheme := newDynamoGraphDeploymentControllerTestScheme(t)
	require.NoError(t, gaiev1.Install(scheme))
	require.NoError(t, networkingv1beta1.AddToScheme(scheme))
	dgd := &v1beta1.DynamoGraphDeployment{
		Name: "7model", Namespace: "default", UID: "graph-uid",
		Spec: v1beta1.DynamoGraphDeploymentSpec{
			Components: []v1beta1.DynamoComponentDeploymentSharedSpec{{
				ComponentName: "epp", ComponentType: v1beta1.ComponentTypeEPP,
			}},
		},
	}
	name := dynamo.GetDCDResourceName(dgd, "epp", "")
	dcd := &v1beta1.DynamoComponentDeployment{Name: name, Namespace: dgd.Namespace, UID: "component-uid"}
	require.NoError(t, controllerutil.SetControllerReference(dgd, dcd, scheme))
	service := &corev1.Service{Name: name, Namespace: dgd.Namespace}
	require.NoError(t, controllerutil.SetControllerReference(dcd, service, scheme))
	kubeClient := fake.NewClientBuilder().WithScheme(scheme).WithObjects(dgd, dcd, service).Build()
	config := &configv1alpha1.OperatorConfiguration{
		ServiceMesh: configv1alpha1.ServiceMeshConfiguration{Provider: "istio", Istio: &configv1alpha1.IstioMeshConfiguration{}},
	}
	reconciler := newDGDEPPReconciler(newDGDResourceSyncer(kubeClient, nil), config,
		&commoncontroller.RuntimeConfig{Gate: features.Gates{Istio: true}}, nil)

	t.Log("Reconcile the graph-owned pool and DestinationRule without renaming the child Service")
	require.NoError(t, reconciler.Reconcile(t.Context(), dgd))
	pool := &gaiev1.InferencePool{}
	require.NoError(t, kubeClient.Get(t.Context(), types.NamespacedName{Name: epp.GetPoolName(dgd.Name), Namespace: dgd.Namespace}, pool))
	require.Equal(t, name, string(pool.Spec.EndpointPickerRef.Name))
	destinationRule := &networkingv1beta1.DestinationRule{}
	require.NoError(t, kubeClient.Get(t.Context(), types.NamespacedName{Name: name, Namespace: dgd.Namespace}, destinationRule))
	require.Equal(t, name+".default.svc.cluster.local", destinationRule.Spec.Host)
}
