/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"testing"

	dynamov1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/webhook/defaulting"
	"github.com/stretchr/testify/require"
	admissionv1 "k8s.io/api/admission/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/webhook/admission"
)

func TestLPXSchedulerSelectionUsesComponents(t *testing.T) {
	t.Log("Select components by their declared type, independently of build or templates")
	canonicalLPX := []dynamov1beta1.DynamoComponentDeploymentSharedSpec{{ComponentType: dynamov1beta1.ComponentTypeLPX}}

	t.Log("Define default-selection scenarios")
	tests := []struct {
		name        string
		annotations map[string]string
		components  []dynamov1beta1.DynamoComponentDeploymentSharedSpec
		want        bool
	}{
		{name: "canonical LPX", components: canonicalLPX, want: true},
		{
			name: "component provider",
			annotations: map[string]string{
				commonconsts.KubeAnnotationWorkloadProvider: commonconsts.WorkloadProviderComponent,
			},
			components: canonicalLPX,
			want:       true,
		},
		{
			name: "Grove opt-out",
			annotations: map[string]string{
				commonconsts.KubeAnnotationEnableGrove: "FALSE",
			},
			components: canonicalLPX,
			want:       true,
		},
		{name: "non-LPX", components: []dynamov1beta1.DynamoComponentDeploymentSharedSpec{{ComponentName: dynamov1beta1.ComponentRoleLPXAgent}}},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Construct the DGD selection input")
			dgd := &dynamov1beta1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{Annotations: test.annotations},
				Spec: dynamov1beta1.DynamoGraphDeploymentSpec{
					Components: test.components,
					Scheduling: &dynamov1beta1.SchedulingSpec{},
				},
			}

			t.Log("Resolve and verify scheduler selection")
			selected := dgd.HasLPXComponent()
			require.Equal(t, test.want, selected)
		})
	}
}

func TestSelectedModelNames(t *testing.T) {
	t.Log("Construct a SpecDecode selection with draft fanout")
	draftReplicas := int32(3)
	dgd := &dynamov1beta1.DynamoGraphDeployment{Spec: dynamov1beta1.DynamoGraphDeploymentSpec{
		Components: []dynamov1beta1.DynamoComponentDeploymentSharedSpec{
			testLPXComponent("small", "draft-build", dynamov1beta1.ComponentRoleSpec{Name: dynamov1beta1.ComponentRoleLPXAgent, PodTemplate: testLPXPodTemplate("draft")}),
			testLPXComponent("large", "target-build", dynamov1beta1.ComponentRoleSpec{Name: dynamov1beta1.ComponentRoleLPXConductor}, dynamov1beta1.ComponentRoleSpec{Name: dynamov1beta1.ComponentRoleLPXAgent, PodTemplate: testLPXPodTemplate("target")}),
		},
	}}
	dgd.Spec.Components[0].Replicas = &draftReplicas

	t.Log("Expand the configured draft replica count into logical model names")
	names, err := SelectedModelNames(dgd)
	require.NoError(t, err)
	require.Equal(t, []string{"draft0", "draft1", "draft2", "target"}, names)

	t.Log("Default an omitted draft replica count to one logical draft")
	dgd.Spec.Components[0].Replicas = nil
	names, err = SelectedModelNames(dgd)
	require.NoError(t, err)
	require.Equal(t, []string{"draft0", "target"}, names)

	for _, invalid := range []int32{0, 9} {
		t.Logf("Reject invalid draft replica count %d", invalid)
		dgd.Spec.Components[0].Replicas = &invalid
		_, err = SelectedModelNames(dgd)
		require.ErrorIs(t, err, ErrUnsupportedRuntime)
	}
	t.Log("Derive the existing default runtime identity with an implicit conductor")
	dgd.Spec.Components = dgd.Spec.Components[1:]
	dgd.Spec.Components[0].Roles = dgd.Spec.Components[0].Roles[1:]
	names, err = SelectedModelNames(dgd)
	require.NoError(t, err)
	require.Equal(t, []string{"default"}, names)
	require.Same(t, &dgd.Spec.Components[0], ServingComponent(dgd))
}

func TestSelectedModelNamesRequiresSharedConductorOwner(t *testing.T) {
	t.Log("Read two agent-only components without a shared conductor owner")
	dgd := newSelectedTestDGD(t, "missing-conductor",
		testLPXComponent("small", "build", dynamov1beta1.ComponentRoleSpec{Name: dynamov1beta1.ComponentRoleLPXAgent, PodTemplate: testLPXPodTemplate("agent")}),
		testLPXComponent("large", "build", dynamov1beta1.ComponentRoleSpec{Name: dynamov1beta1.ComponentRoleLPXAgent, PodTemplate: testLPXPodTemplate("agent")}),
	)
	before := dgd.DeepCopy()

	t.Log("Reject the incomplete runtime without inventing a conductor")
	names, err := SelectedModelNames(dgd)
	require.ErrorIs(t, err, ErrUnsupportedRuntime)
	require.Nil(t, names)
	require.Nil(t, ServingComponent(dgd))
	require.Equal(t, before, dgd)
}

func TestSelectedMinAvailableOwnership(t *testing.T) {
	t.Parallel()

	for _, test := range []struct {
		name              string
		draftMin          *int32
		targetMin         *int32
		singleton         bool
		implicitConductor bool
		wantErr           bool
	}{
		{name: "omitted draft"},
		{name: "default draft", draftMin: ptr.To(int32(1))},
		{name: "nondefault draft", draftMin: ptr.To(int32(2)), wantErr: true},
		{name: "target", targetMin: ptr.To(int32(1))},
		{name: "singleton", targetMin: ptr.To(int32(2)), singleton: true},
		{name: "implicit singleton conductor", targetMin: ptr.To(int32(2)), singleton: true, implicitConductor: true},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Put a non-LPX component before the target and reverse the authored LPX order")
			target := testLPXComponent("target", "target-build",
				dynamov1beta1.ComponentRoleSpec{Name: dynamov1beta1.ComponentRoleLPXConductor},
				dynamov1beta1.ComponentRoleSpec{Name: dynamov1beta1.ComponentRoleLPXAgent, PodTemplate: testLPXPodTemplate("target")})
			target.MinAvailable = test.targetMin
			if test.singleton {
				target.Replicas = ptr.To(int32(2))
			}
			if test.implicitConductor {
				target.Roles = target.Roles[1:]
			}
			dgd := newSelectedTestDGD(t, "min-available",
				dynamov1beta1.DynamoComponentDeploymentSharedSpec{ComponentName: "frontend", MinAvailable: ptr.To(int32(1))}, target)
			if !test.singleton {
				draft := testLPXComponent("draft", "draft-build", dynamov1beta1.ComponentRoleSpec{Name: dynamov1beta1.ComponentRoleLPXAgent, PodTemplate: testLPXPodTemplate("draft")})
				draft.Replicas = ptr.To(int32(2))
				draft.MinAvailable = test.draftMin
				dgd.Spec.Components = append(dgd.Spec.Components, draft)
			}

			for _, phase := range []string{"authored", "defaulted"} {
				t.Run(phase, func(t *testing.T) {
					if phase == "defaulted" {
						t.Log("Apply the real Grove CREATE defaulter before selected-LPX validation")
						ctx := features.WithGate(t.Context(), features.Gates{Grove: true})
						ctx = admission.NewContextWithRequest(ctx, admission.Request{AdmissionRequest: admissionv1.AdmissionRequest{
							Operation: admissionv1.Create,
							Kind:      metav1.GroupVersionKind(dynamov1beta1.DynamoGraphDeploymentGVK),
						}})
						require.NoError(t, defaulting.NewDGDDefaulter("test").Default(ctx, dgd))
						if !test.singleton {
							require.Equal(t, ptr.To(ptr.Deref(test.draftMin, 1)), dgd.Spec.Components[2].MinAvailable)
						}
					}
					before := dgd.DeepCopy()

					t.Log("Reject only nondefault draft availability at its authored index without mutation")
					errs := ValidateSelectedIntent(dgd)
					require.Equal(t, before, dgd)
					if test.wantErr {
						require.Len(t, errs, 1)
						require.Equal(t, field.ErrorTypeForbidden, errs[0].Type)
						require.Equal(t, "spec.components[2].minAvailable", errs[0].Field)
						require.Contains(t, errs[0].Detail, "must be omitted or 1")
						return
					}
					require.Empty(t, errs)
				})
			}
		})
	}
}
