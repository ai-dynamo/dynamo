/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"errors"
	"testing"

	deploymentcapnpv1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/manifest/deployment/v1"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
)

func TestApplyCyborgWeightsPathPrecedesAuthoredReferences(t *testing.T) {
	t.Parallel()

	t.Log("Define authored bindings that depend on the projected checkpoint location")
	projection := &ModelProjection{checkpoint: &Checkpoint{Model: "openai/gpt-oss-20b",
		Revision: "6cee5e81ee83917806bbde320786a8fb61efebee",
	}}
	authored := []corev1.EnvVar{
		{Name: "CYBORG_ARGS", Value: "--weights-path $(CYBORG_WEIGHTS_PATH)"},
		{Name: "OTHER", ValueFrom: &corev1.EnvVarSource{FieldRef: &corev1.ObjectFieldSelector{FieldPath: "metadata.name"}}},
	}
	want := append([]corev1.EnvVar{{
		Name:  cyborgWeightsPathEnv,
		Value: "/nfs/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee",
	}}, authored...)
	for _, test := range []struct {
		name string
		env  []corev1.EnvVar
	}{
		{name: "missing binding", env: authored},
		{name: "stale binding after reference", env: append(append([]corev1.EnvVar(nil), authored...), corev1.EnvVar{
			Name: cyborgWeightsPathEnv, Value: "/nfs/huggingface/hub/models--openai--gpt-oss-20b/snapshots/stale",
		})},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Publish the snapshot path before references while preserving other bindings")
			container := corev1.Container{Env: test.env}
			applyCyborgWeightsPath(&container, projection, "/nfs")
			require.Equal(t, want, container.Env)

			t.Log("Repeat rendering without duplicating or reordering environment bindings")
			applyCyborgWeightsPath(&container, projection, "/nfs")
			require.Equal(t, want, container.Env)
		})
	}
}

func TestApplyCyborgWeightsPathWithoutCheckpointKeepsAuthoredValue(t *testing.T) {
	t.Parallel()

	t.Log("Leave an authored weights path in place when no checkpoint is selected")
	authored := []corev1.EnvVar{{Name: cyborgWeightsPathEnv, Value: "/nfs/authored"}}
	container := corev1.Container{Env: append([]corev1.EnvVar(nil), authored...)}
	applyCyborgWeightsPath(&container, &ModelProjection{}, "/nfs")
	require.Equal(t, authored, container.Env)
}

func TestCheckpointSnapshotPath(t *testing.T) {
	t.Parallel()

	t.Log("Map repository IDs to the Hugging Face cache layout under model storage")
	for _, test := range []struct {
		model string
		want  string
	}{
		{model: "openai/gpt-oss-20b", want: "/mnt/models/models--openai--gpt-oss-20b/snapshots/0123456789abcdef0123456789abcdef01234567"},
		{model: "gpt2", want: "/mnt/models/models--gpt2/snapshots/0123456789abcdef0123456789abcdef01234567"},
	} {
		checkpoint := &Checkpoint{Model: test.model, Revision: "0123456789abcdef0123456789abcdef01234567"}
		require.Equal(t, test.want, checkpointSnapshotPath(checkpoint, "/mnt/models"))
	}
}

var errInvalidTestCheckpoint = errors.New(`deployment.v1.capnp.bin checkpoint.revision "main" is not a full commit SHA`)

func TestProjectModelCheckpoint(t *testing.T) {
	t.Parallel()
	manifestCheckpoint := &Checkpoint{Model: "openai/gpt-oss-20b", Revision: "6cee5e81ee83917806bbde320786a8fb61efebee"}
	for _, test := range []struct {
		name               string
		pipeline           Pipeline
		manifestCheckpoint *Checkpoint
		checkpointErr      error
		want               *Checkpoint
		wantErr            string
	}{
		{name: "hybrid build uses its manifest checkpoint", pipeline: PipelineHybrid, manifestCheckpoint: manifestCheckpoint, want: manifestCheckpoint},
		{name: "hybrid build without a checkpoint projects none", pipeline: PipelineHybrid},
		{name: "LPU-only build ignores its manifest checkpoint", pipeline: PipelineSingle, manifestCheckpoint: manifestCheckpoint},
		{name: "hybrid build rejects an invalid checkpoint", pipeline: PipelineHybrid, checkpointErr: errInvalidTestCheckpoint, wantErr: errInvalidTestCheckpoint.Error()},
		{name: "LPU-only build ignores an invalid checkpoint", pipeline: PipelineSingle, checkpointErr: errInvalidTestCheckpoint},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()

			t.Log("Project a build whose deployment contract may name a checkpoint")
			normalized := acquireTestSnapshot(t, writeV2CompilerFixture(t))
			normalized.checkpoint, normalized.checkpointErr = test.manifestCheckpoint, test.checkpointErr
			projections, err := appendModelProjections(nil, ModelProjectionInput{
				Pipeline: test.pipeline, Models: []string{"default"}, Build: normalized,
			})
			if test.wantErr != "" {
				require.EqualError(t, err, test.wantErr)
				return
			}
			require.NoError(t, err)

			t.Log("Project the selected checkpoint without changing the workload digest")
			require.Same(t, test.want, projections[0].checkpoint)
			baseline := projectTestBuild(t, normalized, test.pipeline)
			require.Equal(t, baseline.Digest(), projections[0].Digest())
		})
	}
}

func TestWithoutBuildCheckpoints(t *testing.T) {
	t.Parallel()

	t.Log("Acquire a build that records a checkpoint through the stripping registry")
	build := acquireTestSnapshot(t, writeV2CompilerFixture(t))
	build.checkpoint = &Checkpoint{Model: "openai/gpt-oss-20b", Revision: "6cee5e81ee83917806bbde320786a8fb61efebee"}
	build.checkpointErr = errInvalidTestCheckpoint
	stripped, err := WithoutBuildCheckpoints(staticModelRegistry{"build": build}).AcquireBuild(t.Context(), "build")
	require.NoError(t, err)

	t.Log("Return the build without its checkpoint or checkpoint error and leave the registry's build unchanged")
	require.Nil(t, stripped.checkpoint)
	require.NoError(t, stripped.checkpointErr)
	require.NotNil(t, build.checkpoint)
	require.Error(t, build.checkpointErr)
	stripped.checkpoint, stripped.checkpointErr = build.checkpoint, build.checkpointErr
	require.Equal(t, build, stripped)
}

func TestCheckpointFromDeploymentV1(t *testing.T) {
	const revision = "6cee5e81ee83917806bbde320786a8fb61efebee"
	for _, test := range []struct {
		name     string
		absent   bool
		provider deploymentcapnpv1.CheckpointProvider
		model    string
		revision string
		want     *Checkpoint
		wantErr  string
	}{
		{name: "absent checkpoint", absent: true},
		{
			name: "pinned Hugging Face checkpoint", model: "openai/gpt-oss-20b", revision: revision,
			want: &Checkpoint{Model: "openai/gpt-oss-20b", Revision: revision},
		},
		{name: "unknown provider", provider: 1, model: "openai/gpt-oss-20b", revision: revision, wantErr: "checkpoint.provider 1 is not supported"},
		{name: "path traversal", model: "openai/../gpt-oss-20b", revision: revision, wantErr: `checkpoint.model "openai/../gpt-oss-20b" is not a valid Hugging Face repository ID`},
		{name: "cache layout alias", model: "openai/gpt--oss", revision: revision, wantErr: `checkpoint.model "openai/gpt--oss" is not a valid Hugging Face repository ID`},
		{name: "git suffix", model: "openai/gpt-oss.git", revision: revision, wantErr: `checkpoint.model "openai/gpt-oss.git" is not a valid Hugging Face repository ID`},
		{name: "branch revision", model: "openai/gpt-oss-20b", revision: "main", wantErr: `checkpoint.revision "main" is not a full commit SHA`},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Encode a deployment contract with the selected checkpoint")
			deployment := newDeploymentV1ContractFixture(t)
			if !test.absent {
				checkpoint, err := deployment.NewCheckpoint()
				require.NoError(t, err)
				checkpoint.SetProvider(test.provider)
				require.NoError(t, checkpoint.SetModel(test.model))
				require.NoError(t, checkpoint.SetRevision(test.revision))
			}

			t.Log("Accept the build, recording only a pinned Hugging Face repository snapshot")
			build, err := buildFromGbuildDeploymentV1("gs://models/build", deployment)
			require.NoError(t, err)
			require.Equal(t, test.want, build.checkpoint)
			if test.wantErr != "" {
				require.ErrorContains(t, build.checkpointErr, test.wantErr)
				return
			}
			require.NoError(t, build.checkpointErr)
		})
	}
}
