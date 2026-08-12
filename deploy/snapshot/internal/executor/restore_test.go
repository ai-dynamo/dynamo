package executor

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"sync"
	"testing"

	"github.com/go-logr/logr/testr"
	specs "github.com/opencontainers/runtime-spec/specs-go"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/ai-dynamo/dynamo/deploy/snapshot/internal/nsmount"
	"github.com/ai-dynamo/dynamo/deploy/snapshot/internal/types"
)

const testPlaceholderPID = 123

type restoreFakeRuntime struct {
	resolvedID      string
	resolveByPodHit bool
}

func (r *restoreFakeRuntime) ResolveContainer(_ context.Context, id string) (int, *specs.Spec, error) {
	r.resolvedID = id
	return testPlaceholderPID, &specs.Spec{}, nil
}

func (r *restoreFakeRuntime) ResolveContainerIDByPod(_ context.Context, _, _, _ string) (string, error) {
	return "", errors.New("pod lookup should not be used")
}

func (r *restoreFakeRuntime) ResolveContainerByPod(_ context.Context, _, _, _ string) (int, *specs.Spec, error) {
	r.resolveByPodHit = true
	return 0, nil, errors.New("pod lookup should not be used")
}

func (r *restoreFakeRuntime) Close() error { return nil }

// fakeMountPoint stands in for a mounted bundle or artifact. It tracks
// Unmount calls so tests can assert that cleanup always runs, regardless of
// whether the restore succeeded or failed.
type fakeMountPoint struct {
	mu            sync.Mutex
	closed        int
	closeErr      error
	strictUnmount bool
}

func (mp *fakeMountPoint) Unmount(_ context.Context, strict bool) error {
	mp.mu.Lock()
	defer mp.mu.Unlock()
	mp.closed++
	mp.strictUnmount = strict
	return mp.closeErr
}

func (mp *fakeMountPoint) Path(name string) (string, error) {
	return filepath.Join("/fake", name), nil
}

func (mp *fakeMountPoint) NsFd() *os.File { return nil }

func (mp *fakeMountPoint) closeCount() int {
	mp.mu.Lock()
	defer mp.mu.Unlock()
	return mp.closed
}

func (mp *fakeMountPoint) wasStrict() bool {
	mp.mu.Lock()
	defer mp.mu.Unlock()
	return mp.strictUnmount
}

// fakeMounter stands in for one policy's mounter and records what it was asked
// to mount. Restore takes two, so a test builds one per policy.
type fakeMounter struct {
	mp       *fakeMountPoint
	mountErr error
	events   *[]string
	label    string

	gotPID int
	gotSrc string
	gotDst string
}

func (m *fakeMounter) Mount(_ context.Context, pid int, src, dst string) (nsmount.MountPoint, error) {
	m.gotPID, m.gotSrc, m.gotDst = pid, src, dst
	if m.events != nil {
		*m.events = append(*m.events, m.label)
	}
	if m.mountErr != nil {
		return nil, m.mountErr
	}
	return m.mp, nil
}

// newMounters builds the pair Restore expects, sharing one event log so tests
// can assert the order the two mounts happen in.
func newMounters(bundle, artifact *fakeMounter) Mounters {
	return Mounters{Bundle: bundle, Artifact: artifact}
}

// writeArtifact lays out basePath/<id>/versions/<version> with a manifest, the
// shape OpenArtifact expects.
func writeArtifact(t *testing.T, checkpointID, version string) string {
	t.Helper()
	basePath := t.TempDir()
	dir := filepath.Join(basePath, checkpointID, "versions", version)
	require.NoError(t, os.MkdirAll(dir, 0o755))
	manifest := types.NewCheckpointManifest(
		checkpointID,
		types.CRIUDumpManifest{},
		types.NewSourcePodManifest("source-id", 456, "node-1", "source-pod", "default", "10.0.0.11", nil),
		types.OverlayManifest{},
	)
	require.NoError(t, types.WriteManifest(dir, manifest))
	return basePath
}

func TestRestore_AlwaysUnmountsBoth(t *testing.T) {
	t.Parallel()

	// nsrestore shells out to nsenter and cannot succeed under a unit test, so
	// this exercises the failure path: both mounts must be released regardless.
	basePath := writeArtifact(t, "checkpoint-123", "1")
	bundleMP := &fakeMountPoint{}
	artifactMP := &fakeMountPoint{}
	bundle := &fakeMounter{mp: bundleMP, label: "bundle"}
	artifact := &fakeMounter{mp: artifactMP, label: "artifact"}
	mounts := newMounters(bundle, artifact)

	_, err := Restore(context.Background(), &restoreFakeRuntime{}, testr.New(t), RestoreRequest{
		CheckpointID:    "checkpoint-123",
		ArtifactVersion: "1",
		BasePath:        basePath,
		ContainerID:     "placeholder-id",
		ContainerName:   "main",
	}, mounts)

	require.Error(t, err)
	assert.Equal(t, 1, bundleMP.closeCount(), "bundle mount must be released even when the restore fails")
	assert.Equal(t, 1, artifactMP.closeCount(), "artifact mount must be released even when the restore fails")
	// A restore that leaves either mount behind in the target has to say so,
	// which a lazy detach would hide.
	assert.True(t, artifactMP.wasStrict(), "the artifact must be unmounted strictly")
	assert.True(t, bundleMP.wasStrict(), "the bundle must be unmounted strictly")
	assert.Equal(t, testPlaceholderPID, bundle.gotPID)
	// Only the exact version directory is exposed, never the checkpoint root
	// above it, so the target cannot reach sibling versions.
	assert.Equal(t, filepath.Join(basePath, "checkpoint-123", "versions", "1"), artifact.gotSrc)
}

func TestRestore_PublishesInProgressOnlyAfterArtifactValidation(t *testing.T) {
	t.Parallel()

	basePath := writeArtifact(t, "checkpoint-123", "1")
	events := []string{}
	mountErr := errors.New("stop after artifact-ready callback")
	bundle := &fakeMounter{mountErr: mountErr, events: &events, label: "bundle"}
	mounts := newMounters(bundle, &fakeMounter{mp: &fakeMountPoint{}, events: &events, label: "artifact"})

	pid, err := Restore(context.Background(), &restoreFakeRuntime{}, testr.New(t), RestoreRequest{
		CheckpointID:    "checkpoint-123",
		ArtifactVersion: "1",
		BasePath:        basePath,
		ContainerID:     "placeholder-id",
		ContainerName:   "main",
		OnArtifactReady: func() error {
			events = append(events, "in-progress")
			return nil
		},
	}, mounts)

	require.ErrorIs(t, err, mountErr)
	assert.Equal(t, testPlaceholderPID, pid, "post-inspection failures must preserve the PID needed for termination")
	// in_progress is published before anything touches the target namespace, and
	// the bundle is the first mount attempted.
	assert.Equal(t, []string{"in-progress", "bundle"}, events)
}

func TestRestore_DoesNotPublishInProgressForInvalidManifest(t *testing.T) {
	t.Parallel()

	basePath := writeArtifact(t, "checkpoint-123", "1")
	require.NoError(t, os.WriteFile(
		filepath.Join(basePath, "checkpoint-123", "versions", "1", types.ManifestFilename),
		[]byte("checkpointId: other-checkpoint\n"),
		0o600,
	))
	called := false
	bundle := &fakeMounter{mp: &fakeMountPoint{}, label: "bundle"}
	artifact := &fakeMounter{mp: &fakeMountPoint{}, label: "artifact"}
	mounts := newMounters(bundle, artifact)

	_, err := Restore(context.Background(), &restoreFakeRuntime{}, testr.New(t), RestoreRequest{
		CheckpointID:    "checkpoint-123",
		ArtifactVersion: "1",
		BasePath:        basePath,
		ContainerID:     "placeholder-id",
		ContainerName:   "main",
		OnArtifactReady: func() error {
			called = true
			return nil
		},
	}, mounts)

	require.ErrorContains(t, err, "does not match requested ID")
	assert.False(t, called)
	assert.Equal(t, 0, bundle.gotPID, "invalid artifacts must be rejected before mounting")
}

func TestRestore_DoesNotMountWhenPublishingInProgressFails(t *testing.T) {
	t.Parallel()

	basePath := writeArtifact(t, "checkpoint-123", "1")
	publishErr := errors.New("in-progress status rejected")
	bundle := &fakeMounter{mp: &fakeMountPoint{}, label: "bundle"}
	artifact := &fakeMounter{mp: &fakeMountPoint{}, label: "artifact"}
	mounts := newMounters(bundle, artifact)

	pid, err := Restore(context.Background(), &restoreFakeRuntime{}, testr.New(t), RestoreRequest{
		CheckpointID:    "checkpoint-123",
		ArtifactVersion: "1",
		BasePath:        basePath,
		ContainerID:     "placeholder-id",
		ContainerName:   "main",
		OnArtifactReady: func() error {
			return publishErr
		},
	}, mounts)

	require.ErrorIs(t, err, publishErr)
	assert.Equal(t, testPlaceholderPID, pid, "the controller needs the resolved PID to terminate this failed target")
	assert.Equal(t, 0, bundle.gotPID, "no namespace may be modified before in-progress is persisted")
}

func TestRestore_RejectsManifestForADifferentCheckpoint(t *testing.T) {
	t.Parallel()

	basePath := writeArtifact(t, "checkpoint-123", "1")
	// Ask for a directory whose manifest names a different checkpoint by
	// renaming the tree the request points at.
	require.NoError(t, os.Rename(
		filepath.Join(basePath, "checkpoint-123"),
		filepath.Join(basePath, "checkpoint-999"),
	))

	bundle := &fakeMounter{mp: &fakeMountPoint{}, label: "bundle"}
	artifact := &fakeMounter{mp: &fakeMountPoint{}, label: "artifact"}
	mounts := newMounters(bundle, artifact)
	_, err := Restore(context.Background(), &restoreFakeRuntime{}, testr.New(t), RestoreRequest{
		CheckpointID:    "checkpoint-999",
		ArtifactVersion: "1",
		BasePath:        basePath,
		ContainerID:     "placeholder-id",
		ContainerName:   "main",
	}, mounts)

	require.ErrorContains(t, err, "does not match requested ID")
	assert.Equal(t, 0, bundle.gotPID, "nothing should be mounted for a mismatched manifest")
}

func TestRestore_RejectsUnusableArtifactCoordinates(t *testing.T) {
	t.Parallel()

	basePath := writeArtifact(t, "checkpoint-123", "1")

	tests := []struct {
		name            string
		checkpointID    string
		artifactVersion string
	}{
		{name: "checkpoint ID escapes the base path", checkpointID: "../etc", artifactVersion: "1"},
		{name: "version escapes the checkpoint", checkpointID: "checkpoint-123", artifactVersion: ".."},
		{name: "checkpoint ID carries a separator", checkpointID: "a/b", artifactVersion: "1"},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			bundle := &fakeMounter{mp: &fakeMountPoint{}, label: "bundle"}
			artifact := &fakeMounter{mp: &fakeMountPoint{}, label: "artifact"}
			mounts := newMounters(bundle, artifact)
			_, err := Restore(context.Background(), &restoreFakeRuntime{}, testr.New(t), RestoreRequest{
				CheckpointID:    tc.checkpointID,
				ArtifactVersion: tc.artifactVersion,
				BasePath:        basePath,
				ContainerID:     "placeholder-id",
				ContainerName:   "main",
			}, mounts)

			require.Error(t, err)
			assert.Equal(t, 0, bundle.gotPID)
		})
	}
}

func TestInspectRestore_UsesContainerIDAndTheFixedMountPoint(t *testing.T) {
	t.Parallel()

	manifest := types.NewCheckpointManifest(
		"checkpoint-123",
		types.CRIUDumpManifest{},
		types.NewSourcePodManifest("source-id", 456, "node-1", "source-pod", "default", "10.0.0.11", nil),
		types.OverlayManifest{},
	)

	rt := &restoreFakeRuntime{}
	snap, err := inspectRestore(context.Background(), rt, testr.New(t), RestoreRequest{
		CheckpointID:  "checkpoint-123",
		ContainerID:   "placeholder-id",
		PodName:       "virtual-pod-name",
		PodNamespace:  "default",
		ContainerName: "main",
	}, manifest)
	require.NoError(t, err)

	assert.Equal(t, "placeholder-id", rt.resolvedID)
	assert.False(t, rt.resolveByPodHit, "ResolveContainerByPod is a fallback, not the primary path")
	assert.Equal(t, testPlaceholderPID, snap.PlaceholderPID)
}
