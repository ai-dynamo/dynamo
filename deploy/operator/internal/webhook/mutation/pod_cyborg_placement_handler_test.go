/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package mutation

import (
	"cmp"
	"context"
	"encoding/json"
	"slices"
	"testing"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	commoncontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	dynamolpx "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx"
	lpxv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/scheduler/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	"github.com/stretchr/testify/require"
	admissionv1 "k8s.io/api/admission/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"sigs.k8s.io/controller-runtime/pkg/webhook/admission"
)

// testExcludedNamespaces lists the namespaces that another operator instance owns.
type testExcludedNamespaces []string

func (n testExcludedNamespaces) Contains(namespace string) bool { return slices.Contains(n, namespace) }

func TestPodCyborgPlacementMutator(t *testing.T) {
	scheme := runtime.NewScheme()
	require.NoError(t, corev1.AddToScheme(scheme))

	const namespace, ephemeral = "serving", "ephemeral-serving"
	active := features.Gates{LPX: true, Grove: true}
	placed := map[string]string{dynamolpx.CyborgPlacementLabel: dynamolpx.CyborgPlacementPerFabricPod}
	cyborg := map[string]string{lpxv1alpha1.PodRoleAnnotation: lpxv1alpha1.PodRoleCyborgWorker, dynamolpx.DeploymentNameAnnotation: "llama"}
	podClique := &metav1.OwnerReference{APIVersion: "grove.io/v1beta1", Kind: "PodClique", Name: "pcs-0-lpx-0-cond", UID: "clique-uid", Controller: new(true)}
	groveGate := corev1.PodSchedulingGate{Name: "grove.io/podgang-pending-creation"}
	placementGate := corev1.PodSchedulingGate{Name: dynamolpx.CyborgPlacementGate}

	for _, tc := range []struct {
		name       string
		operation  admissionv1.Operation
		enabled    features.Gates
		restricted string
		namespace  string
		excluded   testExcludedNamespaces
		labels     map[string]string
		identity   map[string]string
		owner      *metav1.OwnerReference
		gates      []corev1.PodSchedulingGate
		want       []corev1.PodSchedulingGate
	}{
		{
			name:      "a placed Cyborg Pod keeps the Grove gate and gets the placement gate",
			operation: admissionv1.Create, enabled: active, labels: placed, identity: cyborg, owner: podClique,
			gates: []corev1.PodSchedulingGate{groveGate}, want: []corev1.PodSchedulingGate{groveGate, placementGate},
		},
		{
			name:      "a placed Cyborg Pod without a gate gets the placement gate",
			operation: admissionv1.Create, enabled: active, labels: placed, identity: cyborg, owner: podClique,
			want: []corev1.PodSchedulingGate{placementGate},
		},
		{
			name:      "a second admission of the same Pod adds nothing",
			operation: admissionv1.Create, enabled: active, labels: placed, identity: cyborg, owner: podClique,
			gates: []corev1.PodSchedulingGate{groveGate, placementGate}, want: []corev1.PodSchedulingGate{groveGate, placementGate},
		},
		{
			name:      "a Pod update cannot add a gate",
			operation: admissionv1.Update, enabled: active, labels: placed, identity: cyborg, owner: podClique,
			gates: []corev1.PodSchedulingGate{groveGate}, want: []corev1.PodSchedulingGate{groveGate},
		},
		{
			name:      "a Pod that the LPX controller does not place is not gated",
			operation: admissionv1.Create, enabled: active, identity: cyborg, owner: podClique,
			gates: []corev1.PodSchedulingGate{groveGate}, want: []corev1.PodSchedulingGate{groveGate},
		},
		{
			name:      "a placement mode that the LPX controller does not know is not gated",
			operation: admissionv1.Create, enabled: active, identity: cyborg, owner: podClique,
			labels: map[string]string{dynamolpx.CyborgPlacementLabel: "per-rack"},
			gates:  []corev1.PodSchedulingGate{groveGate}, want: []corev1.PodSchedulingGate{groveGate},
		},
		{
			name:      "a Pod of another component that has the label is not gated",
			operation: admissionv1.Create, enabled: active, labels: placed, owner: podClique,
			gates: []corev1.PodSchedulingGate{groveGate}, want: []corev1.PodSchedulingGate{groveGate},
		},
		{
			name:      "an Agent Pod of an LPX deployment is not gated",
			operation: admissionv1.Create, enabled: active, labels: placed, owner: podClique,
			identity: map[string]string{lpxv1alpha1.PodRoleAnnotation: lpxv1alpha1.PodRoleAgent, dynamolpx.DeploymentNameAnnotation: "llama"},
			gates:    []corev1.PodSchedulingGate{groveGate}, want: []corev1.PodSchedulingGate{groveGate},
		},
		{
			name:      "a Cyborg Pod that names no LPX deployment is not gated",
			operation: admissionv1.Create, enabled: active, labels: placed, owner: podClique,
			identity: map[string]string{lpxv1alpha1.PodRoleAnnotation: lpxv1alpha1.PodRoleCyborgWorker},
			gates:    []corev1.PodSchedulingGate{groveGate}, want: []corev1.PodSchedulingGate{groveGate},
		},
		{
			name:      "a copy of a placed Pod without an owner is not gated",
			operation: admissionv1.Create, enabled: active, labels: placed, identity: cyborg,
		},
		{
			name:      "a copy of a placed Pod with another controller is not gated",
			operation: admissionv1.Create, enabled: active, labels: placed, identity: cyborg,
			owner: &metav1.OwnerReference{APIVersion: "apps/v1", Kind: "ReplicaSet", Name: "copy", UID: "rs-uid", Controller: new(true)},
		},
		{
			name:      "a PodClique that is an owner but not the controller does not gate the Pod",
			operation: admissionv1.Create, enabled: active, labels: placed, identity: cyborg,
			owner: &metav1.OwnerReference{APIVersion: "grove.io/v1beta1", Kind: "PodClique", Name: "pcs-0-lpx-0-cond", UID: "clique-uid"},
		},
		{
			name:      "a PodClique kind of another API group does not gate the Pod",
			operation: admissionv1.Create, enabled: active, labels: placed, identity: cyborg,
			owner: &metav1.OwnerReference{APIVersion: "example.com/v1", Kind: "PodClique", Name: "pcs-0-lpx-0-cond", UID: "clique-uid", Controller: new(true)},
		},
		{
			name:      "without the LPX controller no Pod is gated",
			operation: admissionv1.Create, enabled: features.Gates{Grove: true}, labels: placed, identity: cyborg, owner: podClique,
			gates: []corev1.PodSchedulingGate{groveGate}, want: []corev1.PodSchedulingGate{groveGate},
		},
		{
			name:      "without Grove the LPX controller places no Pod",
			operation: admissionv1.Create, enabled: features.Gates{LPX: true}, labels: placed, identity: cyborg, owner: podClique,
			gates: []corev1.PodSchedulingGate{groveGate}, want: []corev1.PodSchedulingGate{groveGate},
		},
		{
			name:      "a namespace-restricted operator places no Pod",
			operation: admissionv1.Create, enabled: active, restricted: namespace, labels: placed, identity: cyborg, owner: podClique,
			gates: []corev1.PodSchedulingGate{groveGate}, want: []corev1.PodSchedulingGate{groveGate},
		},
		{
			name:      "a namespace of another operator instance is not gated",
			operation: admissionv1.Create, enabled: active, excluded: testExcludedNamespaces{namespace}, labels: placed, identity: cyborg, owner: podClique,
			gates: []corev1.PodSchedulingGate{groveGate}, want: []corev1.PodSchedulingGate{groveGate},
		},
		{
			name:      "an ephemeral namespace, which the LPX controller does not reconcile, is not gated",
			operation: admissionv1.Create, enabled: active, namespace: ephemeral, labels: placed, identity: cyborg, owner: podClique,
			gates: []corev1.PodSchedulingGate{groveGate}, want: []corev1.PodSchedulingGate{groveGate},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Log("Given a Pod create or update request as the API server sends it")
			podNamespace := cmp.Or(tc.namespace, namespace)
			pod := &corev1.Pod{
				Name: "pcs-0-lpx-0-cond-x7k2p", Namespace: podNamespace, Labels: tc.labels, Annotations: tc.identity,
				Spec: corev1.PodSpec{SchedulingGates: tc.gates, Containers: []corev1.Container{{Name: "main", Image: "cyborg"}}},
			}
			if tc.owner != nil {
				pod.OwnerReferences = []metav1.OwnerReference{*tc.owner}
			}
			original := mustMarshalPod(t, pod)
			runtimeConfig := &commoncontroller.RuntimeConfig{Gate: tc.enabled}
			if tc.excluded != nil {
				runtimeConfig.ExcludedNamespaces = tc.excluded
			}
			mutator := NewPodCyborgPlacementMutator(&configv1alpha1.OperatorConfiguration{
				Namespace: configv1alpha1.NamespaceConfiguration{Restricted: tc.restricted},
			}, runtimeConfig)
			mutator.scheme = scheme

			t.Log("When the Dynamo admission webhook handles it")
			response := mutator.Handle(features.WithGate(context.Background(), tc.enabled), admission.Request{
				Operation: tc.operation, Namespace: podNamespace, Object: runtime.RawExtension{Raw: original},
			})

			t.Log("Then the Pod is admitted with the expected scheduling gates")
			require.True(t, response.Allowed)
			require.Equal(t, tc.want, applyAdmissionPatches(t, original, response).Spec.SchedulingGates)
		})
	}
}

func TestPodCyborgPlacementMutatorKeepsUnknownFields(t *testing.T) {
	scheme := runtime.NewScheme()
	require.NoError(t, corev1.AddToScheme(scheme))

	t.Log("Given a placed Cyborg Pod from an API server with a Pod field that this operator does not know")
	original := []byte(`{
		"apiVersion": "v1", "kind": "Pod",
		"metadata": {
			"name": "pcs-0-lpx-0-cond-x7k2p", "namespace": "serving",
			"labels": {"` + dynamolpx.CyborgPlacementLabel + `": "` + dynamolpx.CyborgPlacementPerFabricPod + `"},
			"annotations": {"` + lpxv1alpha1.PodRoleAnnotation + `": "` + lpxv1alpha1.PodRoleCyborgWorker + `", "` + dynamolpx.DeploymentNameAnnotation + `": "llama"},
			"ownerReferences": [{"apiVersion": "grove.io/v1beta1", "kind": "PodClique", "name": "pcs-0-lpx-0-cond", "uid": "clique-uid", "controller": true}]
		},
		"spec": {
			"containers": [{"name": "main", "image": "cyborg", "futureContainerField": {"limit": 9007199254740993}}],
			"futurePodField": "kept",
			"schedulingGates": [{"name": "grove.io/podgang-pending-creation"}]
		}
	}`)
	mutator := NewPodCyborgPlacementMutator(&configv1alpha1.OperatorConfiguration{}, &commoncontroller.RuntimeConfig{})
	mutator.scheme = scheme

	t.Log("When the Dynamo admission webhook adds the placement gate")
	response := mutator.Handle(features.WithGate(context.Background(), features.Gates{LPX: true, Grove: true}), admission.Request{
		Operation: admissionv1.Create, Namespace: "serving", Object: runtime.RawExtension{Raw: original},
	})

	t.Log("Then the patch is one operation that appends the gate")
	require.True(t, response.Allowed)
	patches, err := json.Marshal(response.Patches)
	require.NoError(t, err)
	require.JSONEq(t, `[{"op": "add", "path": "/spec/schedulingGates/-", "value": {"name": "`+dynamolpx.CyborgPlacementGate+`"}}]`, string(patches))
}
