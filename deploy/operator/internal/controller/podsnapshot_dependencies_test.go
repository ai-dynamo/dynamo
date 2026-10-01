/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package controller

import (
	"context"
	"testing"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	snapshotv1alpha1 "github.com/ai-dynamo/snapshot/api/v1alpha1"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/utils/ptr"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

func TestPodSnapshotDependencyIndexes(t *testing.T) {
	t.Run("indexes explicit graph references once", func(t *testing.T) {
		t.Log("Given two graph components that reference the same PodSnapshot")
		dgd := &nvidiacomv1beta1.DynamoGraphDeployment{
			Spec: nvidiacomv1beta1.DynamoGraphDeploymentSpec{
				Components: []nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{
					{Experimental: nativeCheckpointExperimental("snapshot-a")},
					{Experimental: nativeCheckpointExperimental("snapshot-a")},
				},
			},
		}

		t.Log("When the field index extracts native dependencies")
		refs := dgdPodSnapshotRefIndexValues(dgd)

		t.Log("Then one Snapshot event is sufficient to enqueue the graph")
		assert.Equal(t, []string{"snapshot-a"}, refs)
	})

	t.Run("indexes component references", func(t *testing.T) {
		t.Log("Given a DCD that references a PodSnapshot")
		dcd := &nvidiacomv1beta1.DynamoComponentDeployment{
			Spec: nvidiacomv1beta1.DynamoComponentDeploymentSpec{
				DynamoComponentDeploymentSharedSpec: nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{
					Experimental: nativeCheckpointExperimental("checkpoint-a"),
				},
			},
		}

		t.Log("When the dependency index is evaluated")
		refs := dcdPodSnapshotRefIndexValues(dcd)

		t.Log("Then the referenced PodSnapshot is indexed")
		assert.Equal(t, []string{"checkpoint-a"}, refs)
	})
}

func TestMapPodSnapshotToDGDRequests(t *testing.T) {
	t.Log("Given two graphs where only one references the changed PodSnapshot")
	scheme := runtime.NewScheme()
	require.NoError(t, nvidiacomv1beta1.AddToScheme(scheme))
	referenced := nativeSnapshotDependencyDGD("referenced", "snapshot-a")
	unrelated := nativeSnapshotDependencyDGD("unrelated", "snapshot-b")
	kubeClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(referenced, unrelated).
		WithIndex(&nvidiacomv1beta1.DynamoGraphDeployment{}, dgdPodSnapshotRefIndex, dgdPodSnapshotRefIndexValues).
		Build()
	reconciler := &DynamoGraphDeploymentReconciler{Client: kubeClient}
	snapshot := &snapshotv1alpha1.PodSnapshot{ObjectMeta: metav1.ObjectMeta{Name: "snapshot-a", Namespace: "default"}}

	t.Log("When the PodSnapshot watch maps the event")
	requests := reconciler.mapPodSnapshotToDGDRequests(context.Background(), snapshot)

	t.Log("Then only the dependent graph is reconciled")
	assert.Equal(t, []ctrl.Request{{NamespacedName: types.NamespacedName{
		Name:      referenced.Name,
		Namespace: referenced.Namespace,
	}}}, requests)
}

func TestMapAutomaticPodSnapshotToDGDRequests(t *testing.T) {
	t.Log("Given an automatic PodSnapshot whose DGD has no explicit reference")
	scheme := runtime.NewScheme()
	require.NoError(t, nvidiacomv1beta1.AddToScheme(scheme))
	dgd := &nvidiacomv1beta1.DynamoGraphDeployment{ObjectMeta: metav1.ObjectMeta{
		Name:      "automatic",
		Namespace: "default",
	}}
	kubeClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(dgd).
		WithIndex(&nvidiacomv1beta1.DynamoGraphDeployment{}, dgdPodSnapshotRefIndex, dgdPodSnapshotRefIndexValues).
		Build()
	reconciler := &DynamoGraphDeploymentReconciler{Client: kubeClient}
	snapshot := &snapshotv1alpha1.PodSnapshot{ObjectMeta: metav1.ObjectMeta{
		Name:      "automatic-snapshot",
		Namespace: "default",
		Labels: map[string]string{
			consts.KubeLabelDynamoGraphDeploymentName: dgd.Name,
		},
		Annotations: map[string]string{
			consts.CheckpointAutoAnnotation: consts.KubeLabelValueTrue,
		},
	}}

	t.Log("When the generated PodSnapshot changes")
	requests := reconciler.mapPodSnapshotToDGDRequests(context.Background(), snapshot)

	t.Log("Then lifecycle metadata maps the event back to the graph")
	assert.Equal(t, []ctrl.Request{{NamespacedName: types.NamespacedName{
		Name:      dgd.Name,
		Namespace: dgd.Namespace,
	}}}, requests)
}

func nativeCheckpointExperimental(name string) *nvidiacomv1beta1.ExperimentalSpec {
	return &nvidiacomv1beta1.ExperimentalSpec{
		Checkpoint: &nvidiacomv1beta1.ComponentCheckpointConfig{
			Enabled:       true,
			CheckpointRef: ptr.To(name),
		},
	}
}

func nativeSnapshotDependencyDGD(name string, snapshotName string) *nvidiacomv1beta1.DynamoGraphDeployment {
	return &nvidiacomv1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: "default"},
		Spec: nvidiacomv1beta1.DynamoGraphDeploymentSpec{
			Components: []nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{{
				Experimental: nativeCheckpointExperimental(snapshotName),
			}},
		},
	}
}
