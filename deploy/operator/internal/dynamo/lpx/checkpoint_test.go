/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"testing"

	deploymentcapnpv1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/manifest/deployment/v1"
	"github.com/stretchr/testify/require"
)

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
		require.Equal(t, test.want, checkpoint.SnapshotPath("/mnt/models"))
	}
}

func TestProjectModelCheckpoint(t *testing.T) {
	t.Parallel()
	manifestCheckpoint := &Checkpoint{Model: "openai/gpt-oss-20b", Revision: "6cee5e81ee83917806bbde320786a8fb61efebee"}
	for _, test := range []struct {
		name               string
		pipeline           Pipeline
		manifestCheckpoint *Checkpoint
		want               *Checkpoint
	}{
		{name: "hybrid build uses its manifest checkpoint", pipeline: PipelineHybrid, manifestCheckpoint: manifestCheckpoint, want: manifestCheckpoint},
		{name: "hybrid build without a checkpoint projects none", pipeline: PipelineHybrid},
		{name: "LPU-only build ignores its manifest checkpoint", pipeline: PipelineSingle, manifestCheckpoint: manifestCheckpoint},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()

			t.Log("Project a build whose deployment contract may name a checkpoint")
			normalized := acquireTestSnapshot(t, writeV2CompilerFixture(t))
			normalized.checkpoint = test.manifestCheckpoint
			projections, err := projectModels(ModelProjectionInput{
				Pipeline: test.pipeline, Models: []string{"default"}, Build: normalized,
			})
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
	stripped, err := WithoutBuildCheckpoints(staticModelRegistry{"build": build}).AcquireBuild(t.Context(), "build")
	require.NoError(t, err)

	t.Log("Return the build without its checkpoint and leave the registry's build unchanged")
	require.Nil(t, stripped.checkpoint)
	require.NotNil(t, build.checkpoint)
	stripped.checkpoint = build.checkpoint
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

			t.Log("Accept only a build with no checkpoint or a pinned Hugging Face repository snapshot")
			build, err := buildFromGbuildDeploymentV1("gs://models/build", deployment)
			if test.wantErr != "" {
				require.ErrorContains(t, err, test.wantErr)
				return
			}
			require.NoError(t, err)
			require.Equal(t, test.want, build.checkpoint)
		})
	}
}
