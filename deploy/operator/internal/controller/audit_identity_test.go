/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package controller

import (
	"context"
	"testing"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/discovery"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

const (
	testAuditDGDName = "test-dgd"
	testAuditNS      = "dynamo-system-0"
	testAuditRoleArn = "arn:aws:iam::123456789012:role/my-audit-role"
	testAuditOldRole = "arn:aws:iam::123456789012:role/old-role"
	testAuditNewRole = "arn:aws:iam::123456789012:role/new-role"
)

func TestReconcileAuditIdentity_AddsIrsaAnnotation(t *testing.T) {
	ctx := context.Background()
	scheme := newDynamoGraphDeploymentControllerTestScheme(t)

	saName := discovery.GetK8sDiscoveryServiceAccountName(testAuditDGDName)

	sa := &corev1.ServiceAccount{
		ObjectMeta: metav1.ObjectMeta{
			Name:      saName,
			Namespace: testAuditNS,
		},
	}

	dgd := &v1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      testAuditDGDName,
			Namespace: testAuditNS,
		},
		Spec: v1beta1.DynamoGraphDeploymentSpec{
			Audit: &v1beta1.AuditSpec{
				AwsS3: &v1beta1.AuditAwsS3Spec{
					IrsaRoleArn: testAuditRoleArn,
				},
			},
		},
	}

	reconciler := &DynamoGraphDeploymentReconciler{
		Client: fake.NewClientBuilder().
			WithScheme(scheme).
			WithObjects(sa).
			Build(),
	}

	err := reconciler.reconcileAuditIdentity(ctx, dgd)
	require.NoError(t, err)

	updated := &corev1.ServiceAccount{}
	err = reconciler.Get(ctx, types.NamespacedName{Name: saName, Namespace: testAuditNS}, updated)
	require.NoError(t, err)
	assert.Equal(t, testAuditRoleArn, updated.Annotations["eks.amazonaws.com/role-arn"])
}

func TestReconcileAuditIdentity_RemovesAnnotationWhenSpecCleared(t *testing.T) {
	ctx := context.Background()
	scheme := newDynamoGraphDeploymentControllerTestScheme(t)

	saName := discovery.GetK8sDiscoveryServiceAccountName(testAuditDGDName)

	sa := &corev1.ServiceAccount{
		ObjectMeta: metav1.ObjectMeta{
			Name:      saName,
			Namespace: testAuditNS,
			Annotations: map[string]string{
				"eks.amazonaws.com/role-arn":   testAuditOldRole,
				"nvidia.com/last-applied-hash": "abc123",
			},
		},
	}

	dgd := &v1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      testAuditDGDName,
			Namespace: testAuditNS,
		},
		Spec: v1beta1.DynamoGraphDeploymentSpec{},
	}

	reconciler := &DynamoGraphDeploymentReconciler{
		Client: fake.NewClientBuilder().
			WithScheme(scheme).
			WithObjects(sa).
			Build(),
	}

	err := reconciler.reconcileAuditIdentity(ctx, dgd)
	require.NoError(t, err)

	updated := &corev1.ServiceAccount{}
	err = reconciler.Get(ctx, types.NamespacedName{Name: saName, Namespace: testAuditNS}, updated)
	require.NoError(t, err)

	_, hasIrsa := updated.Annotations["eks.amazonaws.com/role-arn"]
	assert.False(t, hasIrsa, "IRSA annotation should be removed when spec.audit is nil")
	assert.Equal(t, "abc123", updated.Annotations["nvidia.com/last-applied-hash"])
}

func TestReconcileAuditIdentity_UpdatesInPlace(t *testing.T) {
	ctx := context.Background()
	scheme := newDynamoGraphDeploymentControllerTestScheme(t)

	saName := discovery.GetK8sDiscoveryServiceAccountName(testAuditDGDName)

	sa := &corev1.ServiceAccount{
		ObjectMeta: metav1.ObjectMeta{
			Name:      saName,
			Namespace: testAuditNS,
			Annotations: map[string]string{
				"eks.amazonaws.com/role-arn": testAuditOldRole,
			},
		},
	}

	dgd := &v1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      testAuditDGDName,
			Namespace: testAuditNS,
		},
		Spec: v1beta1.DynamoGraphDeploymentSpec{
			Audit: &v1beta1.AuditSpec{
				AwsS3: &v1beta1.AuditAwsS3Spec{
					IrsaRoleArn: testAuditNewRole,
				},
			},
		},
	}

	reconciler := &DynamoGraphDeploymentReconciler{
		Client: fake.NewClientBuilder().
			WithScheme(scheme).
			WithObjects(sa).
			Build(),
	}

	err := reconciler.reconcileAuditIdentity(ctx, dgd)
	require.NoError(t, err)

	updated := &corev1.ServiceAccount{}
	err = reconciler.Get(ctx, types.NamespacedName{Name: saName, Namespace: testAuditNS}, updated)
	require.NoError(t, err)
	assert.Equal(t, testAuditNewRole, updated.Annotations["eks.amazonaws.com/role-arn"])
}

func TestReconcileAuditIdentity_NoOpWhenSAMissing(t *testing.T) {
	ctx := context.Background()
	scheme := newDynamoGraphDeploymentControllerTestScheme(t)

	dgd := &v1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      testAuditDGDName,
			Namespace: testAuditNS,
		},
		Spec: v1beta1.DynamoGraphDeploymentSpec{
			Audit: &v1beta1.AuditSpec{
				AwsS3: &v1beta1.AuditAwsS3Spec{
					IrsaRoleArn: testAuditRoleArn,
				},
			},
		},
	}

	reconciler := &DynamoGraphDeploymentReconciler{
		Client: fake.NewClientBuilder().
			WithScheme(scheme).
			Build(),
	}

	err := reconciler.reconcileAuditIdentity(ctx, dgd)
	assert.NoError(t, err)
}
