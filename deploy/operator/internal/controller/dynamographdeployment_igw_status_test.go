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
	"github.com/onsi/gomega"
	corev1 "k8s.io/api/core/v1"
	apimeta "k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/tools/record"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	gaiev1 "sigs.k8s.io/gateway-api-inference-extension/api/v1"
	apixv1alpha1 "sigs.k8s.io/gateway-api-inference-extension/apix/config/v1alpha1"
	gatewayv1 "sigs.k8s.io/gateway-api/apis/v1"
)

func igwScheme(t testing.TB) *runtime.Scheme {
	t.Helper()
	s := runtime.NewScheme()
	for _, add := range []func(*runtime.Scheme) error{
		corev1.AddToScheme, v1beta1.AddToScheme, gaiev1.Install, gatewayv1.Install,
	} {
		if err := add(s); err != nil {
			t.Fatalf("add to scheme: %v", err)
		}
	}
	return s
}

func igwDGD() *v1beta1.DynamoGraphDeployment {
	return &v1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "graph",
			Namespace: "ns",
			Annotations: map[string]string{
				commonconsts.KubeAnnotationInferenceGatewayName:      "inference-gateway",
				commonconsts.KubeAnnotationInferenceGatewayNamespace: "gw-ns",
			},
		},
		Spec: v1beta1.DynamoGraphDeploymentSpec{
			Components: []v1beta1.DynamoComponentDeploymentSharedSpec{{
				ComponentName: "Epp",
				ComponentType: v1beta1.ComponentTypeEPP,
				EPPConfig:     &v1beta1.EPPConfig{Config: &apixv1alpha1.EndpointPickerConfig{}},
			}},
		},
	}
}

func igwReconciler(t testing.TB) (*DynamoGraphDeploymentReconciler, *v1beta1.DynamoGraphDeployment) {
	t.Helper()
	dgd := igwDGD()
	r := &DynamoGraphDeploymentReconciler{
		Client:        fake.NewClientBuilder().WithScheme(igwScheme(t)).WithObjects(dgd).Build(),
		Config:        &configv1alpha1.OperatorConfiguration{},
		RuntimeConfig: &commoncontroller.RuntimeConfig{},
		Recorder:      record.NewFakeRecorder(10),
	}
	return r, dgd
}

func conditionReason(dgd *v1beta1.DynamoGraphDeployment) (string, metav1.ConditionStatus) {
	c := apimeta.FindStatusCondition(dgd.Status.Conditions, commonconsts.ConditionTypeInferenceGatewayReady)
	if c == nil {
		return "", ""
	}
	return c.Reason, c.Status
}

// No Gateway exists -> condition surfaces GatewayNotFound (not a silent 404).
func TestReconcileEPP_SetsConditionGatewayNotFound(t *testing.T) {
	g := gomega.NewWithT(t)
	r, dgd := igwReconciler(t)
	g.Expect(r.reconcileEPPResources(context.Background(), dgd)).To(gomega.Succeed())
	reason, status := conditionReason(dgd)
	g.Expect(status).To(gomega.Equal(metav1.ConditionFalse))
	g.Expect(reason).To(gomega.Equal(commonconsts.ReasonInferenceGatewayNotFound))
}

// A programmed, allowedRoutes=All gateway in its own ns -> Ready (cross-namespace).
func TestReconcileEPP_SetsConditionReady(t *testing.T) {
	g := gomega.NewWithT(t)
	dgd := igwDGD()
	gw := &gatewayv1.Gateway{
		ObjectMeta: metav1.ObjectMeta{Name: "inference-gateway", Namespace: "gw-ns"},
		Spec: gatewayv1.GatewaySpec{Listeners: []gatewayv1.Listener{{
			Name: "http", Port: 80, Protocol: gatewayv1.HTTPProtocolType,
			AllowedRoutes: &gatewayv1.AllowedRoutes{
				Namespaces: &gatewayv1.RouteNamespaces{From: ptr.To(gatewayv1.NamespacesFromAll)},
			},
		}}},
		Status: gatewayv1.GatewayStatus{Conditions: []metav1.Condition{{
			Type: "Programmed", Status: metav1.ConditionTrue,
			Reason: "Programmed", LastTransitionTime: metav1.Now(),
		}}},
	}
	r := &DynamoGraphDeploymentReconciler{
		Client:        fake.NewClientBuilder().WithScheme(igwScheme(t)).WithObjects(dgd, gw).Build(),
		Config:        &configv1alpha1.OperatorConfiguration{},
		RuntimeConfig: &commoncontroller.RuntimeConfig{},
		Recorder:      record.NewFakeRecorder(10),
	}
	g.Expect(r.reconcileEPPResources(context.Background(), dgd)).To(gomega.Succeed())
	reason, status := conditionReason(dgd)
	g.Expect(status).To(gomega.Equal(metav1.ConditionTrue))
	g.Expect(reason).To(gomega.Equal(commonconsts.ReasonInferenceGatewayReady))
}
