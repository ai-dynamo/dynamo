// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package controller

import (
	"context"
	"errors"
	"testing"

	"github.com/go-logr/logr"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/meta"
	k8sfake "k8s.io/client-go/kubernetes/fake"
	"k8s.io/client-go/tools/cache"
	"sigs.k8s.io/controller-runtime/pkg/client"
	crfake "sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/controller-runtime/pkg/client/interceptor"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	snapshottypes "github.com/ai-dynamo/dynamo/deploy/snapshot/internal/types"
	snapshotprotocol "github.com/ai-dynamo/dynamo/deploy/snapshot/protocol"
)

// makeNodeControllerWithInterceptor mirrors makeNodeController but threads interceptor.Funcs so a
// test can inject API errors on specific code paths.
func makeNodeControllerWithInterceptor(t *testing.T, fc *fakeCheckpointer, funcs interceptor.Funcs, objs ...client.Object) *NodeController {
	t.Helper()
	s := contentScheme(t)
	idx := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{podRefIndex: podRefIndexFunc})
	for _, o := range objs {
		if sc, ok := o.(*nvidiacomv1alpha1.PodSnapshotContent); ok {
			require.NoError(t, idx.Add(mustUnstructured(t, sc)))
		}
	}
	w := &NodeController{
		config:    &snapshottypes.AgentConfig{NodeName: "node-a", Storage: snapshottypes.StorageSpec{Type: "pvc", BasePath: t.TempDir()}},
		clientset: k8sfake.NewClientset(),
		client: crfake.NewClientBuilder().WithScheme(s).WithObjects(objs...).
			WithStatusSubresource(&nvidiacomv1alpha1.PodSnapshotContent{}).
			WithInterceptorFuncs(funcs).Build(),
		runtime:        &fakeRuntime{},
		log:            logr.Discard(),
		holderID:       "snapshot-agent/test",
		inFlight:       make(map[string]struct{}),
		contentIndexer: idx,
	}
	w.checkpointFn = fc.fn
	return w
}

func TestReconcilePodSnapshotContent_ContentGetErrorReturns(t *testing.T) {
	content := makeWorkOrder("podsnapshotcontent-x", "node-a", "x")
	pod := makeSourcePod("x")
	funcs := interceptor.Funcs{
		Get: func(ctx context.Context, c client.WithWatch, key client.ObjectKey, obj client.Object, opts ...client.GetOption) error {
			if _, ok := obj.(*nvidiacomv1alpha1.PodSnapshotContent); ok {
				return errors.New("apiserver unavailable")
			}
			return c.Get(ctx, key, obj, opts...)
		},
	}
	w := makeNodeControllerWithInterceptor(t, &fakeCheckpointer{}, funcs, content, pod)

	w.reconcilePodSnapshotContent(context.Background(), content.Name)

	// The gate could not read the work order, so it must not have promoted the pod.
	_, labeled := getPod(t, w, "inference", "worker-0").Labels[snapshotprotocol.CaptureEligibleLabel]
	assert.False(t, labeled)
}

func TestReconcilePodSnapshotContent_SourcePodGetErrorReturns(t *testing.T) {
	content := makeWorkOrder("podsnapshotcontent-x", "node-a", "x")
	pod := makeSourcePod("x")
	funcs := interceptor.Funcs{
		Get: func(ctx context.Context, c client.WithWatch, key client.ObjectKey, obj client.Object, opts ...client.GetOption) error {
			if _, ok := obj.(*corev1.Pod); ok {
				return errors.New("apiserver unavailable")
			}
			return c.Get(ctx, key, obj, opts...)
		},
	}
	w := makeNodeControllerWithInterceptor(t, &fakeCheckpointer{}, funcs, content, pod)

	w.reconcilePodSnapshotContent(context.Background(), content.Name)

	// A transient pod-Get error must be retried, not written as a terminal failure.
	assert.Empty(t, getContent(t, w, content.Name).Status.Conditions)
}

func TestReconcilePodSnapshotContent_LabelErrorLeavesPodUnlabeled(t *testing.T) {
	content := makeWorkOrder("podsnapshotcontent-x", "node-a", "x")
	pod := makeSourcePod("x")
	funcs := interceptor.Funcs{
		Patch: func(ctx context.Context, c client.WithWatch, obj client.Object, patch client.Patch, opts ...client.PatchOption) error {
			return errors.New("patch rejected")
		},
	}
	w := makeNodeControllerWithInterceptor(t, &fakeCheckpointer{}, funcs, content, pod)

	w.reconcilePodSnapshotContent(context.Background(), content.Name)

	// Validation passed but the promotion patch failed: logged best-effort, pod stays unlabeled.
	_, labeled := getPod(t, w, "inference", "worker-0").Labels[snapshotprotocol.CaptureEligibleLabel]
	assert.False(t, labeled)
}

func TestReconcileSourcePod_ContentGetErrorReturns(t *testing.T) {
	content := makeWorkOrder("podsnapshotcontent-x", "node-a", "x")
	pod := makeSourcePod("x")
	pod.Labels[snapshotprotocol.CaptureEligibleLabel] = "true"
	fc := &fakeCheckpointer{}
	funcs := interceptor.Funcs{
		Get: func(ctx context.Context, c client.WithWatch, key client.ObjectKey, obj client.Object, opts ...client.GetOption) error {
			if _, ok := obj.(*nvidiacomv1alpha1.PodSnapshotContent); ok {
				return errors.New("apiserver unavailable")
			}
			return c.Get(ctx, key, obj, opts...)
		},
	}
	w := makeNodeControllerWithInterceptor(t, fc, funcs, content, pod)

	require.NoError(t, w.reconcileSourcePod(context.Background(), pod))

	assert.False(t, fc.called, "a content Get error must abort before the dump")
}

func TestLabelCaptureEligible_AlreadyLabeledNoOp(t *testing.T) {
	pod := makeSourcePod("x")
	pod.Labels[snapshotprotocol.CaptureEligibleLabel] = "true"
	funcs := interceptor.Funcs{
		Patch: func(ctx context.Context, c client.WithWatch, obj client.Object, patch client.Patch, opts ...client.PatchOption) error {
			return errors.New("patch must not be called")
		},
	}
	w := makeNodeControllerWithInterceptor(t, &fakeCheckpointer{}, funcs, pod)

	// Already labeled → early return, no patch (so the erroring interceptor is never hit).
	require.NoError(t, w.labelCaptureEligible(context.Background(), pod))
}

func TestLabelCaptureEligible_PatchErrorReturned(t *testing.T) {
	pod := makeSourcePod("x")
	funcs := interceptor.Funcs{
		Patch: func(ctx context.Context, c client.WithWatch, obj client.Object, patch client.Patch, opts ...client.PatchOption) error {
			return errors.New("patch rejected")
		},
	}
	w := makeNodeControllerWithInterceptor(t, &fakeCheckpointer{}, funcs, pod)

	err := w.labelCaptureEligible(context.Background(), pod)
	require.Error(t, err)
}

func TestRemoveCaptureEligibleLabel_AbsentNoOp(t *testing.T) {
	pod := makeSourcePod("x") // no CaptureEligibleLabel
	funcs := interceptor.Funcs{
		Patch: func(ctx context.Context, c client.WithWatch, obj client.Object, patch client.Patch, opts ...client.PatchOption) error {
			return errors.New("patch must not be called")
		},
	}
	w := makeNodeControllerWithInterceptor(t, &fakeCheckpointer{}, funcs, pod)

	// No label → early return, no patch attempted (best-effort, void); must not panic.
	w.removeCaptureEligibleLabel(context.Background(), pod)
	_, labeled := getPod(t, w, "inference", "worker-0").Labels[snapshotprotocol.CaptureEligibleLabel]
	assert.False(t, labeled)
}

func TestRemoveCaptureEligibleLabel_PatchErrorLeavesLabel(t *testing.T) {
	pod := makeSourcePod("x")
	pod.Labels[snapshotprotocol.CaptureEligibleLabel] = "true"
	funcs := interceptor.Funcs{
		Patch: func(ctx context.Context, c client.WithWatch, obj client.Object, patch client.Patch, opts ...client.PatchOption) error {
			return errors.New("patch rejected")
		},
	}
	w := makeNodeControllerWithInterceptor(t, &fakeCheckpointer{}, funcs, pod)

	// Patch fails → logged best-effort; the stored pod keeps the label.
	w.removeCaptureEligibleLabel(context.Background(), pod)
	_, labeled := getPod(t, w, "inference", "worker-0").Labels[snapshotprotocol.CaptureEligibleLabel]
	assert.True(t, labeled)
}

func TestSetSnapshotContentSucceeded_StatusPatchErrorLeavesNoCondition(t *testing.T) {
	content := makeWorkOrder("podsnapshotcontent-x", "node-a", "x")
	funcs := interceptor.Funcs{
		SubResourcePatch: func(ctx context.Context, c client.Client, sub string, obj client.Object, patch client.Patch, opts ...client.SubResourcePatchOption) error {
			return errors.New("status patch rejected")
		},
	}
	w := makeNodeControllerWithInterceptor(t, &fakeCheckpointer{}, funcs, content)

	w.setSnapshotContentSucceeded(context.Background(), content)

	assert.Nil(t, meta.FindStatusCondition(getContent(t, w, content.Name).Status.Conditions, nvidiacomv1alpha1.PodSnapshotConditionReady))
}

func TestSetSnapshotContentFailed_StatusPatchErrorLeavesNoCondition(t *testing.T) {
	content := makeWorkOrder("podsnapshotcontent-x", "node-a", "x")
	funcs := interceptor.Funcs{
		SubResourcePatch: func(ctx context.Context, c client.Client, sub string, obj client.Object, patch client.Patch, opts ...client.SubResourcePatchOption) error {
			return errors.New("status patch rejected")
		},
	}
	w := makeNodeControllerWithInterceptor(t, &fakeCheckpointer{}, funcs, content)

	w.setSnapshotContentFailed(context.Background(), content, "SomeReason", errors.New("boom"))

	assert.Nil(t, meta.FindStatusCondition(getContent(t, w, content.Name).Status.Conditions, nvidiacomv1alpha1.PodSnapshotConditionFailed))
}
