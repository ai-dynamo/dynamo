// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package v1alpha1

import (
	"testing"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/stretchr/testify/require"
)

func TestRetuneNativeConversion(t *testing.T) {
	for _, test := range []struct {
		name string
		spec *v1beta1.RetuneSpec
	}{
		{name: "absent"},
		{name: "disabled", spec: &v1beta1.RetuneSpec{}},
		{name: "enabled", spec: &v1beta1.RetuneSpec{Enabled: true, TraceStorage: &v1beta1.RetuneTraceStorageSpec{ClaimName: "request-traces"}}},
		{name: "router startup inputs including explicit zero", spec: &v1beta1.RetuneSpec{Enabled: true,
			TraceStorage: &v1beta1.RetuneTraceStorageSpec{ClaimName: "request-traces"},
			Router:       &v1beta1.RetuneRouterSpec{OverlapScoreCredit: new(float64(0)), PrefillLoadScale: new(float64(2)), Temperature: new(float64(0.5))},
		}},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Own native Planner Retune intent and a producer capability in both resources")
			status := &v1beta1.RetuneStatus{ConfigHash: "contract-binding"}
			dgd := &v1beta1.DynamoGraphDeployment{
				Spec: v1beta1.DynamoGraphDeploymentSpec{Components: []v1beta1.DynamoComponentDeploymentSharedSpec{{
					ComponentName: "optimizer", ComponentType: v1beta1.ComponentTypePlanner, Retune: test.spec,
				}}},
				Status: v1beta1.DynamoGraphDeploymentStatus{ObservedGeneration: 7, Components: map[string]v1beta1.ComponentReplicaStatus{
					"optimizer": {Retune: status},
				}},
			}
			dcd := &v1beta1.DynamoComponentDeployment{Spec: v1beta1.DynamoComponentDeploymentSpec{
				DynamoComponentDeploymentSharedSpec: v1beta1.DynamoComponentDeploymentSharedSpec{
					ComponentName: "optimizer", ComponentType: v1beta1.ComponentTypePlanner, Retune: test.spec,
				},
			}, Status: v1beta1.DynamoComponentDeploymentStatus{ObservedGeneration: 7, Component: &v1beta1.ComponentReplicaStatus{Retune: status}}}
			originalDGD, originalDCD := dgd.DeepCopy(), dcd.DeepCopy()

			t.Log("Convert natively to alpha without hiding Retune fields in annotations")
			alphaDGD, alphaDCD := &DynamoGraphDeployment{}, &DynamoComponentDeployment{}
			require.NoError(t, alphaDGD.ConvertFrom(dgd))
			require.NoError(t, alphaDCD.ConvertFrom(dcd))
			require.Equal(t, test.spec, alphaDGD.Spec.Services["optimizer"].Retune)
			require.Equal(t, test.spec, alphaDCD.Spec.Retune)
			require.Equal(t, status, alphaDGD.Status.Services["optimizer"].Retune)
			require.Equal(t, status, alphaDCD.Status.Service.Retune)
			require.Empty(t, alphaDGD.Annotations)
			require.Empty(t, alphaDCD.Annotations)
			require.Equal(t, originalDGD, dgd)
			require.Equal(t, originalDCD, dcd)

			t.Log("Round-trip every represented value and status without changing the source")
			gotDGD, gotDCD := &v1beta1.DynamoGraphDeployment{}, &v1beta1.DynamoComponentDeployment{}
			require.NoError(t, alphaDGD.ConvertTo(gotDGD))
			require.NoError(t, alphaDCD.ConvertTo(gotDCD))
			require.Equal(t, test.spec, gotDGD.Spec.Components[0].Retune)
			require.Equal(t, test.spec, gotDCD.Spec.Retune)
			require.Equal(t, dgd.Status, gotDGD.Status)
			require.Equal(t, dcd.Status, gotDCD.Status)

			t.Log("Live alpha removal clears both intent and capability, including a reused destination")
			alphaDGD.Spec.Services["optimizer"].Retune = nil
			service := alphaDGD.Status.Services["optimizer"]
			service.Retune = nil
			alphaDGD.Status.Services["optimizer"] = service
			alphaDCD.Spec.Retune = nil
			alphaDCD.Status.Service.Retune = nil
			require.NoError(t, alphaDGD.ConvertTo(gotDGD))
			require.NoError(t, alphaDCD.ConvertTo(gotDCD))
			require.Nil(t, gotDGD.Spec.Components[0].Retune)
			require.Nil(t, gotDCD.Spec.Retune)
			require.Nil(t, gotDGD.Status.Components["optimizer"].Retune)
			require.Nil(t, gotDCD.Status.Component.Retune)
		})
	}
}
