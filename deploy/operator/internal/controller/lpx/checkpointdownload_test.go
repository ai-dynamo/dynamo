/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"context"
	"fmt"
	"testing"
	"time"

	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx"
)

const (
	checkpointTestModel    = "openai/gpt-oss-20b"
	checkpointTestRevision = "6cee5e81ee83917806bbde320786a8fb61efebee"
	checkpointTestKey      = checkpointTestModel + "@" + checkpointTestRevision
	checkpointTestOtherKey = "openai/gpt-oss-120b@0000000000000000000000000000000000000000"
)

func TestEnsureCheckpointsDownloaded(t *testing.T) {
	tests := []struct {
		name           string
		ready          map[string]bool
		err            map[string]error
		existing       []string
		wantErr        string
		wantCalls      []string
		wantDownloaded []string
		wantPending    []string
	}{
		{
			name:           "downloads every checkpoint",
			ready:          map[string]bool{checkpointTestKey: true, checkpointTestOtherKey: true},
			wantCalls:      []string{checkpointTestOtherKey, checkpointTestKey},
			wantDownloaded: []string{checkpointTestOtherKey, checkpointTestKey},
		},
		{
			name:           "waits for a checkpoint download in progress",
			ready:          map[string]bool{checkpointTestOtherKey: true},
			wantCalls:      []string{checkpointTestOtherKey, checkpointTestKey},
			wantDownloaded: []string{checkpointTestOtherKey},
			wantPending:    []string{checkpointTestKey},
		},
		{
			name:           "reports a download error with the checkpoint identity",
			ready:          map[string]bool{checkpointTestOtherKey: true},
			err:            map[string]error{checkpointTestKey: fmt.Errorf("repository not found")},
			wantErr:        `ensure checkpoint "` + checkpointTestKey + `" is downloaded: repository not found`,
			wantCalls:      []string{checkpointTestOtherKey, checkpointTestKey},
			wantDownloaded: []string{checkpointTestOtherKey},
		},
		{
			name:           "reuses fresh checkpoint downloads",
			existing:       []string{checkpointTestOtherKey, checkpointTestKey},
			wantDownloaded: []string{checkpointTestOtherKey, checkpointTestKey},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Log("Select two checkpoints in key order")
			checkpoints := []*v1beta1.LPXCheckpoint{
				{Provider: v1beta1.LPXCheckpointProviderHuggingFace, Model: "openai/gpt-oss-120b", Revision: "0000000000000000000000000000000000000000"},
				{Provider: v1beta1.LPXCheckpointProviderHuggingFace, Model: checkpointTestModel, Revision: checkpointTestRevision},
			}
			registry := newModelDownloadRegistry(t, tt.ready, tt.err)
			started := time.Now()

			t.Log("Check every uncached checkpoint within one shared budget")
			downloaded, pending, err := ensureCheckpointsDownloaded(t.Context(), checkpoints, registry, tt.existing)
			if tt.wantErr != "" {
				require.EqualError(t, err, tt.wantErr)
			} else {
				require.NoError(t, err)
			}
			require.Equal(t, tt.wantCalls, registry.checkpointCalls)
			require.Equal(t, tt.wantDownloaded, downloaded)
			require.Equal(t, tt.wantPending, pending)
			for _, deadline := range registry.deadlines {
				require.WithinDuration(t, started.Add(modelDownloadCheckTimeout/2), deadline, time.Second)
			}
		})
	}
}

func TestHybridCheckpointDownloadGatesRendering(t *testing.T) {
	t.Log("Select a checkpoint for a hybrid workload whose download is still in progress")
	_, dgd, baseRegistry := newLPXTestDGD(t, lpx.PipelineLPX)
	dgd.Spec.Components[0].LPX.Checkpoint = &v1beta1.LPXCheckpoint{
		Provider: v1beta1.LPXCheckpointProviderHuggingFace, Model: checkpointTestModel, Revision: checkpointTestRevision,
	}
	child := newLPXTestDeployment(t, dgd)
	registry := &fakeModelDownloadRegistry{ModelRegistry: baseRegistry}
	reconciler := newLPXTestReconciler(t, registry, child, dgd)
	request := ctrl.Request{NamespacedName: client.ObjectKeyFromObject(child)}

	t.Log("Hold the workload and name the pending checkpoint in the Ready condition")
	result, err := reconciler.Reconcile(t.Context(), request)
	require.NoError(t, err)
	require.Equal(t, modelDownloadRequeueAfter, result.RequeueAfter)
	require.Equal(t, []string{checkpointTestKey}, registry.checkpointCalls)
	require.NoError(t, reconciler.Get(t.Context(), request.NamespacedName, child))
	ready := meta.FindStatusCondition(child.Status.Conditions, v1alpha1.LPXReadyCondition)
	require.NotNil(t, ready)
	require.Equal(t, metav1.ConditionFalse, ready.Status)
	require.Equal(t, checkpointDownloadPendingMessage+": "+checkpointTestKey, ready.Message)
	require.NotNil(t, child.Status.CheckpointDownload)
	require.Empty(t, child.Status.CheckpointDownload.Checkpoints)
	sets := &grovev1alpha1.PodCliqueSetList{}
	require.NoError(t, reconciler.List(t.Context(), sets))
	require.Empty(t, sets.Items)

	t.Log("Render the workload and record the checkpoint once Model Express completes it")
	registry.ready = map[string]bool{checkpointTestKey: true}
	_, err = reconciler.Reconcile(t.Context(), request)
	require.NoError(t, err)
	require.NoError(t, reconciler.Get(t.Context(), request.NamespacedName, child))
	require.Equal(t, []string{checkpointTestKey}, child.Status.CheckpointDownload.Checkpoints)
	require.NotNil(t, child.Status.CheckpointDownload.LastCheckedAt)
	require.NoError(t, reconciler.List(t.Context(), sets))
	require.Len(t, sets.Items, 1)
}

func TestLPUOnlyCheckpointIsRejectedBeforeDownload(t *testing.T) {
	t.Log("Select a checkpoint for an LPU-only workload, which has no Cyborg conductor")
	_, dgd, baseRegistry := newLPXTestDGD(t, lpx.PipelineSingle)
	dgd.Spec.Components[0].LPX.Checkpoint = &v1beta1.LPXCheckpoint{
		Provider: v1beta1.LPXCheckpointProviderHuggingFace, Model: checkpointTestModel, Revision: checkpointTestRevision,
	}
	child := newLPXTestDeployment(t, dgd)
	registry := &fakeModelDownloadRegistry{ModelRegistry: baseRegistry, ready: map[string]bool{checkpointTestKey: true}}
	reconciler := newLPXTestReconciler(t, registry, child, dgd)

	t.Log("Reject the checkpoint during resolution without asking Model Express for it")
	_, err := reconciler.Reconcile(t.Context(), ctrl.Request{NamespacedName: client.ObjectKeyFromObject(child)})
	require.ErrorContains(t, err, "checkpoint requires a hybrid build with a Cyborg conductor")
	require.Empty(t, registry.checkpointCalls)
}

func (r *fakeModelDownloadRegistry) EnsureCheckpointDownloaded(ctx context.Context, checkpoint *v1beta1.LPXCheckpoint) (bool, error) {
	key := lpx.CheckpointKey(checkpoint)
	r.checkpointCalls = append(r.checkpointCalls, key)
	deadline, _ := ctx.Deadline()
	r.deadlines = append(r.deadlines, deadline)
	return r.ready[key], r.err[key]
}
