// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package controller

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"github.com/go-logr/logr"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	k8sfake "k8s.io/client-go/kubernetes/fake"
	"k8s.io/client-go/tools/cache"
	"sigs.k8s.io/controller-runtime/pkg/client"
	crfake "sigs.k8s.io/controller-runtime/pkg/client/fake"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	snapshottypes "github.com/ai-dynamo/dynamo/deploy/snapshot/internal/types"
	snapshotprotocol "github.com/ai-dynamo/dynamo/deploy/snapshot/protocol"
)

// fakeCheckpointer records calls behind the checkpointFn seam and returns a configured error.
type fakeCheckpointer struct {
	mu     sync.Mutex
	called bool
	params CheckpointParams
	err    error
}

// fn is the checkpointFn seam the NodeController invokes for the dump.
func (fc *fakeCheckpointer) fn(_ context.Context, params CheckpointParams) error {
	fc.mu.Lock()
	defer fc.mu.Unlock()
	fc.called = true
	fc.params = params
	return fc.err
}

// wasCalled reports whether the seam was invoked.
func (fc *fakeCheckpointer) wasCalled() bool {
	fc.mu.Lock()
	defer fc.mu.Unlock()
	return fc.called
}

// lastParams returns the params from the most recent seam invocation.
func (fc *fakeCheckpointer) lastParams() CheckpointParams {
	fc.mu.Lock()
	defer fc.mu.Unlock()
	return fc.params
}

// contentScheme builds a scheme with the PodSnapshotContent and core types registered.
func contentScheme(t *testing.T) *runtime.Scheme {
	t.Helper()
	s := runtime.NewScheme()
	require.NoError(t, nvidiacomv1alpha1.AddToScheme(s))
	require.NoError(t, corev1.AddToScheme(s))
	return s
}

// makeNodeController builds a NodeController wired to a fake typed client, runtime, and seam. Any
// PodSnapshotContent in objs is also added to the podRef index (mirroring the content informer's
// cache) so the pod-driven reconcileSourcePod can resolve it; tests that need a different index
// state override w.contentIndexer after construction.
func makeNodeController(t *testing.T, fc *fakeCheckpointer, objs ...client.Object) *NodeController {
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
			WithStatusSubresource(&nvidiacomv1alpha1.PodSnapshotContent{}).Build(),
		runtime:        &fakeRuntime{},
		log:            logr.Discard(),
		holderID:       "snapshot-agent/test",
		inFlight:       make(map[string]struct{}),
		contentIndexer: idx,
	}
	w.checkpointFn = fc.fn
	return w
}

// makeWorkOrder builds a PodSnapshotContent work order pinned to a node and checkpoint id.
// Capture parameters now live on the source pod, so the work order carries only the node
// label and spec.
func makeWorkOrder(name, node, checkpointID string) *nvidiacomv1alpha1.PodSnapshotContent {
	return &nvidiacomv1alpha1.PodSnapshotContent{
		ObjectMeta: metav1.ObjectMeta{
			Name:   name,
			Labels: map[string]string{snapshotprotocol.SnapshotNodeLabel: node},
		},
		Spec: nvidiacomv1alpha1.PodSnapshotContentSpec{
			PodSnapshotRef: nvidiacomv1alpha1.PodSnapshotReference{Namespace: "inference", Name: "podsnapshot-" + checkpointID},
			Source:         nvidiacomv1alpha1.PodSnapshotContentSource{PodRef: nvidiacomv1alpha1.PodReference{Name: "worker-0", UID: types.UID("pod-uid")}, NodeName: node},
		},
	}
}

// makeSourcePod builds a ready source pod that carries the capture parameters the agent reads:
// the checkpoint-id label, the target-container annotation, and the storage/version annotations
// checkpointLocationsFromPod needs.
func makeSourcePod(checkpointID string) *corev1.Pod {
	return &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "worker-0",
			Namespace: "inference",
			UID:       types.UID("pod-uid"),
			Labels:    map[string]string{snapshotprotocol.CheckpointIDLabel: checkpointID},
			Annotations: map[string]string{
				snapshotprotocol.TargetContainersAnnotation:          "main",
				snapshotprotocol.CheckpointArtifactVersionAnnotation: "1",
			},
		},
		Spec: corev1.PodSpec{NodeName: "node-a"},
		Status: corev1.PodStatus{
			Phase: corev1.PodRunning,
			ContainerStatuses: []corev1.ContainerStatus{
				{Name: "main", Ready: true, ContainerID: "containerd://abc123"},
			},
		},
	}
}

// getContent reads a PodSnapshotContent back from the fake client.
func getContent(t *testing.T, w *NodeController, name string) *nvidiacomv1alpha1.PodSnapshotContent {
	t.Helper()
	c := &nvidiacomv1alpha1.PodSnapshotContent{}
	require.NoError(t, w.client.Get(context.Background(), types.NamespacedName{Name: name}, c))
	return c
}

// getPod reads a Pod back from the fake client (used to assert CaptureEligibleLabel changes).
func getPod(t *testing.T, w *NodeController, namespace, name string) *corev1.Pod {
	t.Helper()
	p := &corev1.Pod{}
	require.NoError(t, w.client.Get(context.Background(), types.NamespacedName{Namespace: namespace, Name: name}, p))
	return p
}

func TestReconcileSnapshotContent_IgnoresOtherNode(t *testing.T) {
	content := makeWorkOrder("podsnapshotcontent-x", "node-b", "x")
	fc := &fakeCheckpointer{}
	w := makeNodeController(t, fc, content)

	w.reconcilePodSnapshotContent(context.Background(), content.Name)
	assert.False(t, fc.wasCalled())
	got := getContent(t, w, content.Name)
	assert.Empty(t, got.Status.Conditions)
}

func TestReconcileSnapshotContent_GateLabelsPodOnSuccess(t *testing.T) {
	content := makeWorkOrder("podsnapshotcontent-x", "node-a", "x")
	pod := makeSourcePod("x")
	fc := &fakeCheckpointer{}
	w := makeNodeController(t, fc, content, pod)

	// The gate promotes a valid pod by labeling it; it must NOT run the capture flow itself.
	w.reconcilePodSnapshotContent(context.Background(), content.Name)

	assert.False(t, fc.wasCalled(), "gate must not invoke the dump directly")
	assert.Equal(t, "true", getPod(t, w, "inference", "worker-0").Labels[snapshotprotocol.CaptureEligibleLabel])
	assert.Empty(t, getContent(t, w, content.Name).Status.Conditions)
}

func TestReconcileSourcePod_InFlightGuard(t *testing.T) {
	// Content name differs from ID to prove the guard is keyed on the ID, not the content name.
	content := makeWorkOrder("podsnapshotcontent-mywork", "node-a", "x")
	pod := makeSourcePod("x")
	pod.Labels[snapshotprotocol.CaptureEligibleLabel] = "true"
	w := makeNodeController(t, &fakeCheckpointer{}, content, pod)
	// Seed the guard using the checkpoint ID ("x"), not the content name.
	w.inFlight["x"] = struct{}{}

	require.NoError(t, w.reconcileSourcePod(context.Background(), pod))
	got := getContent(t, w, content.Name)
	assert.Empty(t, got.Status.Conditions, "in-flight guard must not write any status")
}

func TestReconcileSourcePod_MissingCheckpointIDFails(t *testing.T) {
	content := makeWorkOrder("podsnapshotcontent-x", "node-a", "x")
	pod := makeSourcePod("x")
	delete(pod.Labels, snapshotprotocol.CheckpointIDLabel)
	w := makeNodeController(t, &fakeCheckpointer{}, content, pod)

	require.NoError(t, w.reconcileSourcePod(context.Background(), pod))
	got := getContent(t, w, content.Name)
	cond := meta.FindStatusCondition(got.Status.Conditions, nvidiacomv1alpha1.PodSnapshotConditionFailed)
	require.NotNil(t, cond)
	assert.Equal(t, "MissingCheckpointID", cond.Reason)
}

func TestReconcileSourcePod_ProvenanceInvalidFailsAndUnlabels(t *testing.T) {
	content := makeWorkOrder("podsnapshotcontent-x", "node-a", "x")
	pod := makeSourcePod("x")
	pod.UID = types.UID("stale-uid") // UID mismatch vs the work order's pinned source UID
	pod.Labels[snapshotprotocol.CaptureEligibleLabel] = "true"
	w := makeNodeController(t, &fakeCheckpointer{}, content, pod)

	require.NoError(t, w.reconcileSourcePod(context.Background(), pod))

	cond := meta.FindStatusCondition(getContent(t, w, content.Name).Status.Conditions, nvidiacomv1alpha1.PodSnapshotConditionFailed)
	require.NotNil(t, cond)
	assert.Equal(t, "StalePodReference", cond.Reason)
	_, labeled := getPod(t, w, "inference", "worker-0").Labels[snapshotprotocol.CaptureEligibleLabel]
	assert.False(t, labeled, "eligible label must be removed on cancellation")
}

func TestReconcileSourcePod_InFlightShortCircuits(t *testing.T) {
	// Content name differs from ID to prove the guard is keyed on the ID, not the content name.
	content := makeWorkOrder("podsnapshotcontent-mywork", "node-a", "x")
	pod := makeSourcePod("x")
	pod.Labels[snapshotprotocol.CaptureEligibleLabel] = "true"
	w := makeNodeController(t, &fakeCheckpointer{}, content, pod)
	// A dump is already in flight: tryAcquire short-circuits before any further work, so a second
	// reconcile does nothing — no status write, no relabel. Guard keyed on ID "x".
	w.inFlight["x"] = struct{}{}

	require.NoError(t, w.reconcileSourcePod(context.Background(), pod))

	got := getPod(t, w, "inference", "worker-0")
	assert.Empty(t, getContent(t, w, content.Name).Status.Conditions, "in-flight dump must not be touched")
	assert.Equal(t, "true", got.Labels[snapshotprotocol.CaptureEligibleLabel], "in-flight dump must not be unlabeled")
}

func TestReconcileSourcePod_GuardSurvivesWorkOrderRecreation(t *testing.T) {
	// A PodSnapshot delete+recreate yields a new content name but the same checkpoint ID.
	// The in-flight guard is keyed on the ID, so the new work order must not start a second dump.
	content := makeWorkOrder("podsnapshotcontent-recreated", "node-a", "x")
	pod := makeSourcePod("x")
	pod.Labels[snapshotprotocol.CaptureEligibleLabel] = "true"
	w := makeNodeController(t, &fakeCheckpointer{}, content, pod)
	// Seed guard with ID "x" — simulates the original work order's dump still running.
	w.inFlight["x"] = struct{}{}

	require.NoError(t, w.reconcileSourcePod(context.Background(), pod))

	assert.Empty(t, getContent(t, w, content.Name).Status.Conditions, "second work order must not start a dump or write status")
}

func TestReconcileSourcePod_InvalidCheckpointIDFails(t *testing.T) {
	content := makeWorkOrder("podsnapshotcontent-x", "node-a", "x")
	pod := makeSourcePod("x")
	// Replace the valid ID with one that contains an uppercase letter — not a valid DNS-1123 label.
	pod.Labels[snapshotprotocol.CheckpointIDLabel] = "Bad_ID"
	w := makeNodeController(t, &fakeCheckpointer{}, content, pod)

	require.NoError(t, w.reconcileSourcePod(context.Background(), pod))

	got := getContent(t, w, content.Name)
	cond := meta.FindStatusCondition(got.Status.Conditions, nvidiacomv1alpha1.PodSnapshotConditionFailed)
	require.NotNil(t, cond)
	assert.Equal(t, "InvalidCheckpointID", cond.Reason)
}

func TestReconcileSnapshotContent_FailedContainerUnsticksAndFails(t *testing.T) {
	content := makeWorkOrder("podsnapshotcontent-abc", "node-a", "abc")
	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:        "worker-0",
			Namespace:   "inference",
			UID:         types.UID("pod-uid"),
			Labels:      map[string]string{snapshotprotocol.CheckpointIDLabel: "abc"},
			Annotations: map[string]string{snapshotprotocol.TargetContainersAnnotation: "main"},
		},
		Spec: corev1.PodSpec{NodeName: "node-a"},
		Status: corev1.PodStatus{
			Phase: corev1.PodRunning,
			ContainerStatuses: []corev1.ContainerStatus{
				{Name: "main", State: corev1.ContainerState{Running: &corev1.ContainerStateRunning{}}, ContainerID: "containerd://main-id"},
				{Name: "helper", State: corev1.ContainerState{Terminated: &corev1.ContainerStateTerminated{ExitCode: 1, Reason: "Error"}}, ContainerID: "containerd://helper-id"},
			},
		},
	}
	fc := &fakeCheckpointer{}
	rt := &fakeRuntime{} // PID 0 → ResolveContainer errors → SendSignalToPID skipped (no real signal sent)
	w := makeNodeController(t, fc, content, pod)
	w.runtime = rt

	require.NoError(t, w.reconcileSourcePod(context.Background(), pod))

	got := getContent(t, w, content.Name)
	cond := meta.FindStatusCondition(got.Status.Conditions, nvidiacomv1alpha1.PodSnapshotConditionFailed)
	require.NotNil(t, cond)
	assert.Equal(t, "CheckpointContainerFailed", cond.Reason)
	assert.Contains(t, cond.Message, "helper")
	assert.True(t, sawEventReason(w.clientset.(*k8sfake.Clientset), "CheckpointFailed"))
	// Only the still-running sibling is resolved for the SIGKILL; the dead container is skipped.
	assert.Equal(t, []string{"main-id"}, rt.resolvedContainerIDs)
	assert.False(t, fc.wasCalled())
	assert.Empty(t, w.inFlight)
}

func TestFailCheckpointOnContainerExit_IgnoresCleanExit(t *testing.T) {
	w := makeNodeController(t, &fakeCheckpointer{})
	pod := &corev1.Pod{Status: corev1.PodStatus{ContainerStatuses: []corev1.ContainerStatus{
		{Name: "main", State: corev1.ContainerState{Running: &corev1.ContainerStateRunning{}}},
		{Name: "helper", State: corev1.ContainerState{Terminated: &corev1.ContainerStateTerminated{ExitCode: 0}}},
	}}}

	handled := w.failCheckpointOnContainerExit(context.Background(), &nvidiacomv1alpha1.PodSnapshotContent{}, pod)
	assert.False(t, handled)
}

func TestReconcileSnapshotContent_OpaqueNameUsesPodLabel(t *testing.T) {
	// The work order name does not encode the pod's checkpoint id: the name is opaque and the
	// pod label is the sole source of truth. Capture must proceed using the pod label ("abc").
	content := makeWorkOrder("podsnapshotcontent-unrelated-name", "node-a", "abc")
	pod := makeSourcePod("abc")
	fc := &fakeCheckpointer{}
	w := makeNodeController(t, fc, content, pod)
	w.runtime = &fakeRuntime{resolveContainerPID: 7}

	require.NoError(t, w.reconcileSourcePod(context.Background(), pod))
	require.Eventually(t, fc.wasCalled, time.Second, 5*time.Millisecond)

	// The checkpoint id and destination come from the pod label, not the work order name.
	params := fc.lastParams()
	assert.Equal(t, "abc", params.CheckpointID)
	assert.Equal(t, filepath.Join(w.config.Storage.BasePath, "abc", "versions", "1"), params.HostPath)

	// setSnapshotContentSucceeded runs after checkpointFn returns, so poll for the Ready condition rather than reading once.
	require.Eventually(t, func() bool {
		c := &nvidiacomv1alpha1.PodSnapshotContent{}
		if err := w.client.Get(context.Background(), types.NamespacedName{Name: content.Name}, c); err != nil {
			return false
		}
		return meta.FindStatusCondition(c.Status.Conditions, nvidiacomv1alpha1.PodSnapshotConditionReady) != nil
	}, time.Second, 5*time.Millisecond)
}

func TestReconcileSnapshotContent_ResumeWritesReady(t *testing.T) {
	content := makeWorkOrder("podsnapshotcontent-abc", "node-a", "abc")
	pod := makeSourcePod("abc")
	fc := &fakeCheckpointer{}
	w := makeNodeController(t, fc, content, pod)
	w.runtime = &fakeRuntime{resolveContainerPID: 4242}
	// Pre-create the artifact directory at the resolved destination so the resume check fires.
	dest := filepath.Join(w.config.Storage.BasePath, "abc", "versions", "1")
	require.NoError(t, os.MkdirAll(dest, 0o755))

	require.NoError(t, w.reconcileSourcePod(context.Background(), pod))
	assert.False(t, fc.wasCalled())
	got := getContent(t, w, content.Name)
	cond := meta.FindStatusCondition(got.Status.Conditions, nvidiacomv1alpha1.PodSnapshotConditionReady)
	require.NotNil(t, cond)
}

func TestReconcileSnapshotContent_PodMountResolvesContainerPID(t *testing.T) {
	content := makeWorkOrder("podsnapshotcontent-abc", "node-a", "abc")
	pod := makeSourcePod("abc")
	fc := &fakeCheckpointer{}
	w := makeNodeController(t, fc, content, pod)
	w.config.Storage.AccessMode = snapshottypes.StorageAccessModePodMount
	rt := &fakeRuntime{resolveContainerPID: 4242}
	w.runtime = rt

	require.NoError(t, w.reconcileSourcePod(context.Background(), pod))

	// podMount mode resolves the container PID and feeds it through checkpointLocationsFromPod
	// (a zero PID would fail there with a different reason). The subsequent live-PID validation
	// fails in a unit test because /host/proc/<pid> does not exist, which proves the non-zero
	// PID flowed through to validatePodMountContainerPID.
	assert.Contains(t, rt.resolvedContainerIDs, "abc123")
	assert.False(t, fc.wasCalled())
	got := getContent(t, w, content.Name)
	cond := meta.FindStatusCondition(got.Status.Conditions, nvidiacomv1alpha1.PodSnapshotConditionFailed)
	require.NotNil(t, cond)
	assert.Equal(t, "ContainerChanged", cond.Reason)
}

func TestReconcileSnapshotContent_PodNotFoundFails(t *testing.T) {
	content := makeWorkOrder("podsnapshotcontent-x", "node-a", "x")
	w := makeNodeController(t, &fakeCheckpointer{}, content) // no pod

	w.reconcilePodSnapshotContent(context.Background(), content.Name)
	got := getContent(t, w, content.Name)
	cond := meta.FindStatusCondition(got.Status.Conditions, nvidiacomv1alpha1.PodSnapshotConditionFailed)
	require.NotNil(t, cond)
	assert.Equal(t, "SourcePodNotFound", cond.Reason)
}

func TestClassifySourcePod(t *testing.T) {
	content := makeWorkOrder("podsnapshotcontent-x", "node-a", "x") // PodRef Name worker-0, UID pod-uid
	running := func(uid string, phase corev1.PodPhase, deleting bool) *corev1.Pod {
		p := &corev1.Pod{
			ObjectMeta: metav1.ObjectMeta{Name: "worker-0", Namespace: "inference", UID: types.UID(uid)},
			Status:     corev1.PodStatus{Phase: phase},
		}
		if deleting {
			now := metav1.Now()
			p.DeletionTimestamp = &now
		}
		return p
	}

	reason, _ := classifySourcePod(content, running("pod-uid", corev1.PodRunning, false))
	assert.Equal(t, "", reason)

	reason, _ = classifySourcePod(content, running("other-uid", corev1.PodRunning, false))
	assert.Equal(t, "StalePodReference", reason)

	for _, phase := range []corev1.PodPhase{corev1.PodFailed, corev1.PodSucceeded} {
		reason, _ = classifySourcePod(content, running("pod-uid", phase, false))
		assert.Equal(t, "SourcePodGone", reason)
	}

	reason, _ = classifySourcePod(content, running("pod-uid", corev1.PodRunning, true))
	assert.Equal(t, "SourcePodGone", reason)
}

func TestReconcileSnapshotContent_StalePodUIDFails(t *testing.T) {
	content := makeWorkOrder("podsnapshotcontent-x", "node-a", "x")
	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "worker-0", Namespace: "inference", UID: types.UID("different-uid")},
		Spec:       corev1.PodSpec{NodeName: "node-a"},
		Status:     corev1.PodStatus{Phase: corev1.PodRunning},
	}
	w := makeNodeController(t, &fakeCheckpointer{}, content, pod)

	w.reconcilePodSnapshotContent(context.Background(), content.Name)
	got := getContent(t, w, content.Name)
	cond := meta.FindStatusCondition(got.Status.Conditions, nvidiacomv1alpha1.PodSnapshotConditionFailed)
	require.NotNil(t, cond)
	assert.Equal(t, "StalePodReference", cond.Reason)
}

func TestReconcileSnapshotContent_PodFailedFails(t *testing.T) {
	content := makeWorkOrder("podsnapshotcontent-x", "node-a", "x")
	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "worker-0", Namespace: "inference", UID: types.UID("pod-uid")},
		Spec:       corev1.PodSpec{NodeName: "node-a"},
		Status:     corev1.PodStatus{Phase: corev1.PodFailed},
	}
	w := makeNodeController(t, &fakeCheckpointer{}, content, pod)

	w.reconcilePodSnapshotContent(context.Background(), content.Name)
	got := getContent(t, w, content.Name)
	cond := meta.FindStatusCondition(got.Status.Conditions, nvidiacomv1alpha1.PodSnapshotConditionFailed)
	require.NotNil(t, cond)
	assert.Equal(t, "SourcePodGone", cond.Reason)
}

func TestReconcileSnapshotContent_NotReadyQuiesceNoOp(t *testing.T) {
	content := makeWorkOrder("podsnapshotcontent-x", "node-a", "x")
	pod := makeSourcePod("x")
	pod.Status.ContainerStatuses[0].Ready = false
	fc := &fakeCheckpointer{}
	w := makeNodeController(t, fc, content, pod)

	require.NoError(t, w.reconcileSourcePod(context.Background(), pod))
	assert.False(t, fc.wasCalled())
	got := getContent(t, w, content.Name)
	assert.Empty(t, got.Status.Conditions)
}

func TestReconcileSnapshotContent_CapturesFromPod(t *testing.T) {
	content := makeWorkOrder("podsnapshotcontent-abc", "node-a", "abc")
	pod := makeSourcePod("abc")
	fc := &fakeCheckpointer{}
	w := makeNodeController(t, fc, content, pod)
	w.runtime = &fakeRuntime{resolveContainerPID: 7}

	require.NoError(t, w.reconcileSourcePod(context.Background(), pod))

	// acquireLease runs synchronously in reconcileSourcePod before the goroutine is launched, so
	// the Lease exists immediately after the function returns.
	_, err := w.clientset.CoordinationV1().Leases("inference").Get(context.Background(), "checkpoint-lease-abc", metav1.GetOptions{})
	assert.NoError(t, err, "Lease checkpoint-lease-abc must exist in namespace inference")

	require.Eventually(t, fc.wasCalled, time.Second, 5*time.Millisecond)

	// Capture parameters are read from the source pod, not from PodSnapshotContent metadata.
	params := fc.lastParams()
	assert.Equal(t, "abc", params.CheckpointID)
	assert.Equal(t, "main", params.ContainerName)
	assert.Equal(t, "abc123", params.ContainerID)
	assert.Equal(t, 7, params.ContainerPID)
	// agentMount: HostPath == ContainerPath == resolved destination.
	dest := filepath.Join(w.config.Storage.BasePath, "abc", "versions", "1")
	assert.Equal(t, dest, params.HostPath)
	assert.Equal(t, dest, params.ContainerPath)

	// setSnapshotContentSucceeded runs after checkpointFn returns, so poll for the Ready condition rather than reading once.
	require.Eventually(t, func() bool {
		c := &nvidiacomv1alpha1.PodSnapshotContent{}
		if err := w.client.Get(context.Background(), types.NamespacedName{Name: content.Name}, c); err != nil {
			return false
		}
		return meta.FindStatusCondition(c.Status.Conditions, nvidiacomv1alpha1.PodSnapshotConditionReady) != nil
	}, time.Second, 5*time.Millisecond)
}

func TestRunCheckpoint_WritesReadyOnSuccess(t *testing.T) {
	content := makeWorkOrder("podsnapshotcontent-abc", "node-a", "abc")
	fc := &fakeCheckpointer{}
	w := makeNodeController(t, fc, content)
	pod := &corev1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "worker-0", Namespace: "inference", UID: types.UID("pod-uid")}}
	leaseKey := client.ObjectKey{Namespace: "inference", Name: "checkpoint-lease-abc"}
	loc := checkpointLocations{
		HostPath:      filepath.Join(w.config.Storage.BasePath, "abc", "versions", "1"),
		ContainerPath: filepath.Join(w.config.Storage.BasePath, "abc", "versions", "1"),
	}

	w.runCheckpoint(context.Background(), content, pod, "main", "abc123", 7, "abc", loc, leaseKey, "abc")

	assert.True(t, fc.wasCalled())
	got := getContent(t, w, content.Name)
	cond := meta.FindStatusCondition(got.Status.Conditions, nvidiacomv1alpha1.PodSnapshotConditionReady)
	require.NotNil(t, cond)
}

func TestRunCheckpoint_WritesFailedOnError(t *testing.T) {
	content := makeWorkOrder("podsnapshotcontent-abc", "node-a", "abc")
	fc := &fakeCheckpointer{err: errors.New("criu boom")}
	w := makeNodeController(t, fc, content)
	pod := &corev1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "worker-0", Namespace: "inference", UID: types.UID("pod-uid")}}
	leaseKey := client.ObjectKey{Namespace: "inference", Name: "checkpoint-lease-abc"}
	loc := checkpointLocations{
		HostPath:      filepath.Join(w.config.Storage.BasePath, "abc", "versions", "1"),
		ContainerPath: filepath.Join(w.config.Storage.BasePath, "abc", "versions", "1"),
	}

	w.runCheckpoint(context.Background(), content, pod, "main", "abc123", 7, "abc", loc, leaseKey, "abc")

	got := getContent(t, w, content.Name)
	cond := meta.FindStatusCondition(got.Status.Conditions, nvidiacomv1alpha1.PodSnapshotConditionFailed)
	require.NotNil(t, cond)
	assert.Equal(t, "CheckpointFailed", cond.Reason)
}

// mustUnstructured converts a typed object to the *unstructured.Unstructured the dynamic informer
// (and thus the podRef index) stores.
func mustUnstructured(t *testing.T, obj runtime.Object) *unstructured.Unstructured {
	t.Helper()
	m, err := runtime.DefaultUnstructuredConverter.ToUnstructured(obj)
	require.NoError(t, err)
	return &unstructured.Unstructured{Object: m}
}

// contentForWorker0 builds a PodSnapshotContent referencing pod inference/worker-0 with a given
// creation time, optionally carrying a terminal condition (PodSnapshotConditionReady/Failed).
func contentForWorker0(name string, created metav1.Time, terminal string) *nvidiacomv1alpha1.PodSnapshotContent {
	c := &nvidiacomv1alpha1.PodSnapshotContent{
		ObjectMeta: metav1.ObjectMeta{Name: name, CreationTimestamp: created},
		Spec: nvidiacomv1alpha1.PodSnapshotContentSpec{
			PodSnapshotRef: nvidiacomv1alpha1.PodSnapshotReference{Namespace: "inference", Name: "snapshot-" + name},
			Source:         nvidiacomv1alpha1.PodSnapshotContentSource{PodRef: nvidiacomv1alpha1.PodReference{Name: "worker-0", UID: types.UID("pod-uid")}, NodeName: "node-a"},
		},
	}
	if terminal != "" {
		meta.SetStatusCondition(&c.Status.Conditions, metav1.Condition{Type: terminal, Status: metav1.ConditionTrue, Reason: "Done"})
	}
	return c
}

func TestPodRefIndexFunc(t *testing.T) {
	keys, err := podRefIndexFunc(mustUnstructured(t, contentForWorker0("podsnapshotcontent-abc", metav1.Unix(1000, 0), "")))
	require.NoError(t, err)
	assert.Equal(t, []string{"inference/worker-0"}, keys)
}

func TestPodRefIndexFunc_MissingFieldsOrWrongType(t *testing.T) {
	keys, err := podRefIndexFunc(&unstructured.Unstructured{Object: map[string]interface{}{"spec": map[string]interface{}{}}})
	require.NoError(t, err)
	assert.Nil(t, keys)

	keys, err = podRefIndexFunc("not-unstructured")
	require.NoError(t, err)
	assert.Nil(t, keys)
}

func TestContentFromInformerObj(t *testing.T) {
	u := mustUnstructured(t, contentForWorker0("podsnapshotcontent-abc", metav1.Unix(1000, 0), ""))

	c, ok := contentFromInformerObj(u)
	require.True(t, ok)
	assert.Equal(t, "podsnapshotcontent-abc", c.Name)

	c, ok = contentFromInformerObj(cache.DeletedFinalStateUnknown{Key: "k", Obj: u})
	require.True(t, ok)
	assert.Equal(t, "podsnapshotcontent-abc", c.Name)

	_, ok = contentFromInformerObj(cache.DeletedFinalStateUnknown{Key: "k", Obj: "bad"})
	assert.False(t, ok)
	_, ok = contentFromInformerObj("bad")
	assert.False(t, ok)
}

func TestChooseActiveContent_OldestNonTerminalWins(t *testing.T) {
	// "podsnapshotcontent-a" sorts first by name but is newer; oldest-by-CreationTimestamp must win.
	newer := mustUnstructured(t, contentForWorker0("podsnapshotcontent-a", metav1.Unix(2000, 0), ""))
	older := mustUnstructured(t, contentForWorker0("podsnapshotcontent-b", metav1.Unix(1000, 0), ""))
	assert.Equal(t, "podsnapshotcontent-b", chooseActiveContent([]interface{}{newer, older}))
}

func TestChooseActiveContent_SkipsTerminalAndTieBreaksByName(t *testing.T) {
	terminal := mustUnstructured(t, contentForWorker0("podsnapshotcontent-old", metav1.Unix(1000, 0), nvidiacomv1alpha1.PodSnapshotConditionReady))
	tieA := mustUnstructured(t, contentForWorker0("podsnapshotcontent-a", metav1.Unix(2000, 0), ""))
	tieB := mustUnstructured(t, contentForWorker0("podsnapshotcontent-b", metav1.Unix(2000, 0), ""))
	assert.Equal(t, "podsnapshotcontent-a", chooseActiveContent([]interface{}{terminal, tieB, tieA}))
}

func TestChooseActiveContent_AllTerminalReturnsEmpty(t *testing.T) {
	ready := mustUnstructured(t, contentForWorker0("podsnapshotcontent-a", metav1.Unix(1000, 0), nvidiacomv1alpha1.PodSnapshotConditionReady))
	failed := mustUnstructured(t, contentForWorker0("podsnapshotcontent-b", metav1.Unix(2000, 0), nvidiacomv1alpha1.PodSnapshotConditionFailed))
	assert.Equal(t, "", chooseActiveContent([]interface{}{ready, failed}))
}

// podWithFailedSibling builds the inference/worker-0 source pod with the target Running and a
// sibling Terminated non-zero, so a reconcile triggers the unstick.
func podWithFailedSibling() *corev1.Pod {
	return &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:        "worker-0",
			Namespace:   "inference",
			UID:         types.UID("pod-uid"),
			Labels:      map[string]string{snapshotprotocol.CheckpointIDLabel: "abc"},
			Annotations: map[string]string{snapshotprotocol.TargetContainersAnnotation: "main"},
		},
		Spec: corev1.PodSpec{NodeName: "node-a"},
		Status: corev1.PodStatus{
			Phase: corev1.PodRunning,
			ContainerStatuses: []corev1.ContainerStatus{
				{Name: "main", State: corev1.ContainerState{Running: &corev1.ContainerStateRunning{}}, ContainerID: "containerd://main-id"},
				{Name: "helper", State: corev1.ContainerState{Terminated: &corev1.ContainerStateTerminated{ExitCode: 1, Reason: "Error"}}, ContainerID: "containerd://helper-id"},
			},
		},
	}
}

func seedIndex(t *testing.T, contents ...*nvidiacomv1alpha1.PodSnapshotContent) cache.Indexer {
	t.Helper()
	idx := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{podRefIndex: podRefIndexFunc})
	for _, c := range contents {
		require.NoError(t, idx.Add(mustUnstructured(t, c)))
	}
	return idx
}

func TestReconcileSourcePod_TriggersUnstick(t *testing.T) {
	content := makeWorkOrder("podsnapshotcontent-abc", "node-a", "abc")
	content.CreationTimestamp = metav1.Unix(1000, 0)
	pod := podWithFailedSibling()
	fc := &fakeCheckpointer{}
	rt := &fakeRuntime{}
	w := makeNodeController(t, fc, content, pod)
	w.runtime = rt

	require.NoError(t, w.reconcileSourcePod(context.Background(), pod))

	got := getContent(t, w, content.Name)
	cond := meta.FindStatusCondition(got.Status.Conditions, nvidiacomv1alpha1.PodSnapshotConditionFailed)
	require.NotNil(t, cond)
	assert.Equal(t, "CheckpointContainerFailed", cond.Reason)
	assert.Equal(t, []string{"main-id"}, rt.resolvedContainerIDs)
	assert.False(t, fc.wasCalled())
}

func TestReconcileSourcePod_PodNotIndexedNoOp(t *testing.T) {
	content := makeWorkOrder("podsnapshotcontent-abc", "node-a", "abc")
	pod := podWithFailedSibling()
	w := makeNodeController(t, &fakeCheckpointer{}, content, pod)
	w.contentIndexer = seedIndex(t) // override: empty index

	require.NoError(t, w.reconcileSourcePod(context.Background(), pod))
	assert.Empty(t, getContent(t, w, content.Name).Status.Conditions)
}

func TestReconcileSourcePod_IndexErrorReturned(t *testing.T) {
	content := makeWorkOrder("podsnapshotcontent-abc", "node-a", "abc")
	pod := podWithFailedSibling()
	w := makeNodeController(t, &fakeCheckpointer{}, content, pod)
	// Indexer without podRefIndex registered → ByIndex returns an error; reconcile surfaces it
	// (the informer handler logs it) and writes no status.
	w.contentIndexer = cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{})

	require.Error(t, w.reconcileSourcePod(context.Background(), pod))
	assert.Empty(t, getContent(t, w, content.Name).Status.Conditions)
}
