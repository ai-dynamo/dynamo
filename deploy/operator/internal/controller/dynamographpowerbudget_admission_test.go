//go:build !clustertest

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package controller

import (
	"context"
	"testing"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
)

func TestDGPBCopyOnceCRDImmutability(t *testing.T) {
	t.Log("Start an API server with the generated DGPB CRD")
	env := sharedEnv.RunT(t)
	kubeClient := env.Client()
	ctx := context.Background()
	key := types.NamespacedName{Namespace: env.Namespace(), Name: "immutable-power-policy"}
	dgpb := &nvidiacomv1beta1.DynamoGraphPowerBudget{
		ObjectMeta: metav1.ObjectMeta{Namespace: key.Namespace, Name: key.Name},
		Spec: nvidiacomv1beta1.DynamoGraphPowerBudgetSpec{
			BudgetWatts: 2400,
			Policy:      nvidiacomv1beta1.DynamoGraphPowerBudgetPolicy{MinEndpoint: 1},
		},
	}
	if err := kubeClient.Create(ctx, dgpb); err != nil {
		t.Fatalf("create DGPB: %v", err)
	}

	t.Log("Reject a DGPB spec mutation through CRD transition validation")
	dgpb.Spec.BudgetWatts = 2500
	dgpb.Labels = map[string]string{"attempted": "spec-mutation"}
	err := kubeClient.Update(ctx, dgpb)
	if err == nil || !apierrors.IsInvalid(err) {
		t.Fatalf("mutate DGPB spec error = %v, want API-server Invalid", err)
	}

	t.Log("Keep the stored policy unchanged after the rejected update")
	stored := &nvidiacomv1beta1.DynamoGraphPowerBudget{}
	if err := kubeClient.Get(ctx, key, stored); err != nil {
		t.Fatalf("read DGPB after rejected update: %v", err)
	}
	if stored.Spec.BudgetWatts != 2400 || stored.Spec.Policy.MinEndpoint != 1 {
		t.Fatalf("stored immutable spec = %#v, want original policy", stored.Spec)
	}

	t.Log("Allow metadata-only updates without a DGPB webhook")
	stored.Labels = map[string]string{"metadata": "allowed"}
	if err := kubeClient.Update(ctx, stored); err != nil {
		t.Fatalf("metadata-only DGPB update: %v", err)
	}
}
