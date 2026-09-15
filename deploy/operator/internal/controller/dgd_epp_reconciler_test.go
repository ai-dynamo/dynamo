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

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/epp"
	"github.com/onsi/gomega"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/uuid"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
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
			ObjectMeta: metav1.ObjectMeta{Name: "my-model", Namespace: "default", UID: uuid.NewUUID()},
		}
		owned := &corev1.ConfigMap{
			ObjectMeta: metav1.ObjectMeta{
				Name:      epp.GetConfigMapName(dgd.Name),
				Namespace: dgd.Namespace,
			},
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
			ObjectMeta: metav1.ObjectMeta{Name: "my-model", Namespace: "default", UID: uuid.NewUUID()},
		}
		// A user's own ConfigMap -- referenced via eppConfig.configMapRef
		// while this DGD was still on legacy Go EPP -- that happens to be
		// named exactly what the operator would have generated
		// (<dgd>-epp-config). It carries no owner reference to this DGD.
		foreign := &corev1.ConfigMap{
			ObjectMeta: metav1.ObjectMeta{
				Name:      epp.GetConfigMapName(dgd.Name),
				Namespace: dgd.Namespace,
			},
			Data: map[string]string{"do-not-delete-me": "user data"},
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
			ObjectMeta: metav1.ObjectMeta{Name: "my-model", Namespace: "default", UID: uuid.NewUUID()},
		}
		fakeClient := fake.NewClientBuilder().WithScheme(scheme).WithObjects(dgd).Build()
		reconciler := newDGDEPPReconciler(newDGDResourceSyncer(fakeClient, nil), nil, nil, nil)

		g.Expect(reconciler.deleteOwnedLegacyEPPConfigMap(ctx, dgd)).To(gomega.Succeed())
	})
}
