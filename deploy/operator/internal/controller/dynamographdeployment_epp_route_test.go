/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package controller

import (
	"context"
	"testing"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	commoncontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/epp"
	"github.com/onsi/gomega"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/tools/record"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	gaiev1 "sigs.k8s.io/gateway-api-inference-extension/api/v1"
	apixv1alpha1 "sigs.k8s.io/gateway-api-inference-extension/apix/config/v1alpha1"
	gatewayv1 "sigs.k8s.io/gateway-api/apis/v1"
)

// eppRouteTestScheme registers everything reconcileEPPResources touches: the
// DGD type (owner refs + GVK), plus the EPP ConfigMap / InferencePool / HTTPRoute
// child types it syncs.
func eppRouteTestScheme(t testing.TB) *runtime.Scheme {
	t.Helper()
	s := runtime.NewScheme()
	for _, add := range []func(*runtime.Scheme) error{
		corev1.AddToScheme,
		v1beta1.AddToScheme,
		gaiev1.Install,
		gatewayv1.Install,
	} {
		if err := add(s); err != nil {
			t.Fatalf("add to scheme: %v", err)
		}
	}
	return s
}

// eppDGD builds a DGD carrying an EPP component with an inline EndpointPickerConfig.
// annotations lets a test set the inference-gateway handoff annotation(s).
func eppDGD(annotations map[string]string) *v1beta1.DynamoGraphDeployment {
	return &v1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:        "graph",
			Namespace:   "ns",
			Annotations: annotations,
		},
		Spec: v1beta1.DynamoGraphDeploymentSpec{
			Components: []v1beta1.DynamoComponentDeploymentSharedSpec{
				{
					ComponentName: "Epp",
					ComponentType: v1beta1.ComponentTypeEPP,
					EPPConfig: &v1beta1.EPPConfig{
						Config: &apixv1alpha1.EndpointPickerConfig{},
					},
				},
			},
		},
	}
}

func eppRouteReconciler(t testing.TB, dgd *v1beta1.DynamoGraphDeployment) *DynamoGraphDeploymentReconciler {
	t.Helper()
	s := eppRouteTestScheme(t)
	return &DynamoGraphDeploymentReconciler{
		Client: fake.NewClientBuilder().
			WithScheme(s).
			WithObjects(dgd).
			Build(),
		// IstioAvailable defaults to false -> the DestinationRule branch is skipped;
		// Config is still dereferenced (ServiceMesh.IsEnabled), so it must be non-nil.
		Config:        &configv1alpha1.OperatorConfiguration{},
		RuntimeConfig: &commoncontroller.RuntimeConfig{},
		Recorder:      record.NewFakeRecorder(10),
	}
}

// When the inference-gateway annotation is set, the hook emits the HTTPRoute
// (alongside the InferencePool) and binds it to the named Gateway.
func TestReconcileEPPResources_EmitsHTTPRouteWhenAnnotationSet(t *testing.T) {
	g := gomega.NewWithT(t)
	ctx := context.Background()

	dgd := eppDGD(map[string]string{
		commonconsts.KubeAnnotationInferenceGatewayName:      "prod-gw",
		commonconsts.KubeAnnotationInferenceGatewayNamespace: "gw-ns",
	})
	r := eppRouteReconciler(t, dgd)

	g.Expect(r.reconcileEPPResources(ctx, dgd)).To(gomega.Succeed())

	// InferencePool is always created for an EPP DGD.
	pool := &gaiev1.InferencePool{}
	g.Expect(r.Get(ctx, types.NamespacedName{
		Name:      epp.GetPoolName(dgd.Name, nil),
		Namespace: "ns",
	}, pool)).To(gomega.Succeed())

	// HTTPRoute is emitted and bound to the annotated (cross-namespace) Gateway.
	route := &gatewayv1.HTTPRoute{}
	g.Expect(r.Get(ctx, types.NamespacedName{
		Name:      dgd.Name + "-route",
		Namespace: "ns",
	}, route)).To(gomega.Succeed())

	g.Expect(route.Spec.ParentRefs).To(gomega.HaveLen(1))
	parent := route.Spec.ParentRefs[0]
	g.Expect(string(parent.Name)).To(gomega.Equal("prod-gw"))
	g.Expect(parent.Namespace).NotTo(gomega.BeNil())
	g.Expect(string(*parent.Namespace)).To(gomega.Equal("gw-ns"))

	// The route is owned by the DGD (garbage-collected with it).
	g.Expect(route.OwnerReferences).To(gomega.HaveLen(1))
	g.Expect(route.OwnerReferences[0].Name).To(gomega.Equal(dgd.Name))
}

// Without the annotation, the InferencePool is still created but no HTTPRoute is
// emitted -- the safe path for hand-authored EPP DGDs that wire their own gateway.
func TestReconcileEPPResources_NoHTTPRouteWithoutAnnotation(t *testing.T) {
	g := gomega.NewWithT(t)
	ctx := context.Background()

	dgd := eppDGD(nil)
	r := eppRouteReconciler(t, dgd)

	g.Expect(r.reconcileEPPResources(ctx, dgd)).To(gomega.Succeed())

	pool := &gaiev1.InferencePool{}
	g.Expect(r.Get(ctx, types.NamespacedName{
		Name:      epp.GetPoolName(dgd.Name, nil),
		Namespace: "ns",
	}, pool)).To(gomega.Succeed())

	route := &gatewayv1.HTTPRoute{}
	err := r.Get(ctx, types.NamespacedName{Name: dgd.Name + "-route", Namespace: "ns"}, route)
	g.Expect(apierrors.IsNotFound(err)).To(gomega.BeTrue(), "no HTTPRoute should be emitted without the annotation")
}
