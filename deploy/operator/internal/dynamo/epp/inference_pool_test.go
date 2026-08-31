/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package epp

import (
	"testing"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestGenerateInferencePoolSelectsWorkerClass(t *testing.T) {
	dgd := &v1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "graph", Namespace: "ns"},
		Spec: v1beta1.DynamoGraphDeploymentSpec{
			Components: []v1beta1.DynamoComponentDeploymentSharedSpec{
				{ComponentName: "epp", ComponentType: v1beta1.ComponentTypeEPP},
				{ComponentName: "prefill", ComponentType: v1beta1.ComponentTypePrefill},
				{ComponentName: "decode", ComponentType: v1beta1.ComponentTypeDecode},
			},
		},
	}

	pool, err := GenerateInferencePool(dgd, "epp", "graph-epp", nil)
	if err != nil {
		t.Fatalf("GenerateInferencePool() error = %v", err)
	}

	selector := pool.Spec.Selector.MatchLabels
	if got := selector[consts.KubeLabelDynamoComponentClass]; got != consts.ComponentClassWorker {
		t.Fatalf("worker class selector = %q, want %q", got, consts.ComponentClassWorker)
	}
	if _, hasComponentType := selector[consts.KubeLabelDynamoComponentType]; hasComponentType {
		t.Fatalf("selector still filters by component type: %#v", selector)
	}
}
