// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package controller

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	config "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	api "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	common "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	grove "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	"github.com/google/go-cmp/cmp"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/client-go/tools/events"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/controller-runtime/pkg/client/interceptor"
)

const upgradeReplayRoot = "testdata/upgrade"

type upgradeReplayCatalog struct {
	CandidateMinor string
	Releases       map[string]struct {
		Revision        string
		NewWorkerSuffix bool
	}
	Profiles []upgradeReplayProfile
}

type upgradeReplayProfile struct{ Name, Release, Image string }

type upgradeReplaySnapshot struct {
	Generated, Revision string
	DGD                 api.DynamoGraphDeployment
	PCS                 json.RawMessage
}

// readUpgradeReplayJSON rejects unrecognized fixture fields instead of silently losing evidence.
func readUpgradeReplayJSON(t *testing.T, path string, target any) []byte {
	t.Helper()
	data, err := os.ReadFile(path)
	require.NoError(t, err)
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	require.NoError(t, decoder.Decode(target))
	return data
}

func loadUpgradeReplayCatalog(t *testing.T) upgradeReplayCatalog {
	t.Helper()
	var lock struct {
		Generated         string
		Inputs, Snapshots map[string]string
	}
	readUpgradeReplayJSON(t, filepath.Join(upgradeReplayRoot, "snapshots.lock.json"), &lock)
	require.NotEmpty(t, lock.Inputs)
	require.NotEmpty(t, lock.Snapshots)
	for path, want := range lock.Inputs {
		data, err := os.ReadFile(filepath.Join("../..", path))
		require.NoError(t, err)
		sum := sha256.Sum256(data)
		require.Equal(t, want, hex.EncodeToString(sum[:]), "baseline input changed: %s; regenerate only from pinned releases", path)
	}
	for path, want := range lock.Snapshots {
		require.Equal(t, filepath.Base(path), path)
		data, err := os.ReadFile(filepath.Join(upgradeReplayRoot, path))
		require.NoError(t, err)
		sum := sha256.Sum256(data)
		require.Equal(t, want, hex.EncodeToString(sum[:]), "historical snapshot changed: %s", path)
	}
	var catalog upgradeReplayCatalog
	readUpgradeReplayJSON(t, filepath.Join(upgradeReplayRoot, "catalog.json"), &catalog)
	require.Len(t, catalog.Profiles, len(lock.Snapshots), "catalog and snapshots must cover the same inventory")
	seen := map[string]bool{}
	for _, profile := range catalog.Profiles {
		require.False(t, seen[profile.Name], "duplicate profile")
		seen[profile.Name] = true
		require.Contains(t, lock.Snapshots, profile.Name+".json")
		require.Contains(t, catalog.Releases, profile.Release)
	}
	return catalog
}

// loadUpgradeReplaySnapshot checks release and runtime identity while retaining raw PCS evidence.
func loadUpgradeReplaySnapshot(t *testing.T, catalog upgradeReplayCatalog, profile upgradeReplayProfile) (upgradeReplaySnapshot, *grove.PodCliqueSet) {
	t.Helper()
	var snapshot upgradeReplaySnapshot
	readUpgradeReplayJSON(t, filepath.Join(upgradeReplayRoot, profile.Name+".json"), &snapshot)
	require.Equal(t, catalog.Releases[profile.Release].Revision, snapshot.Revision)
	require.NotEmpty(t, snapshot.DGD.Annotations[consts.AnnotationCurrentWorkerHashV2])
	for _, component := range snapshot.DGD.Spec.Components {
		require.Equal(t, profile.Image, component.PodTemplate.Spec.Containers[0].Image)
	}
	var pcs grove.PodCliqueSet
	require.NoError(t, json.Unmarshal(snapshot.PCS, &pcs))
	return snapshot, &pcs
}

// The fake client exercises resource writes, not real Grove controllers or Pod lifecycles.
// dgd and pcs must be non-nil.
func newUpgradeReplayReconciler(t *testing.T, dgd *api.DynamoGraphDeployment, pcs *grove.PodCliqueSet) (*groveWorkloadsReconciler, client.Client) {
	t.Helper()
	c := fake.NewClientBuilder().WithScheme(upgradeScheme).WithObjects(dgd, pcs).
		WithRESTMapper(groveScaleRESTMapper()).
		WithStatusSubresource(&api.DynamoGraphDeployment{}).WithInterceptorFuncs(groveScaleInterceptor(interceptor.Funcs{}, nil)).Build()
	recorder := events.NewFakeRecorder(100)
	rollout := newDGDWorkerRolloutReconciler(c, recorder)
	secrets := &mockDockerSecretRetriever{GetSecretsFunc: func(string, string) ([]string, error) { return nil, nil }}
	r := newGroveWorkloadsReconciler(c, recorder, rollout, &config.OperatorConfiguration{}, &common.RuntimeConfig{Gate: features.Gates{Grove: true}}, secrets)
	return r, c
}

// upgradeReplayTemplates inspects raw JSON so a newer API type cannot discard an old Pod field.
func upgradeReplayTemplates(t *testing.T, data []byte) map[string]any {
	t.Helper()
	var object map[string]any
	require.NoError(t, json.Unmarshal(data, &object))
	cliques, found, err := unstructured.NestedSlice(object, "spec", "template", "cliques")
	require.NoError(t, err)
	require.True(t, found)
	templates := map[string]any{}
	for _, value := range cliques {
		clique, ok := value.(map[string]any)
		require.True(t, ok)
		name, ok := clique["name"].(string)
		require.True(t, ok)
		spec, ok := clique["spec"].(map[string]any)
		require.True(t, ok)
		templates[name] = map[string]any{"labels": clique["labels"], "annotations": clique["annotations"], "podSpec": spec["podSpec"], "replicas": spec["replicas"]}
	}
	return templates
}

func writeUpgradeReplayArtifact(t *testing.T, name string, value any) []byte {
	t.Helper()
	data, err := json.MarshalIndent(value, "", "  ")
	require.NoError(t, err)
	root := os.Getenv("DYNAMO_UPGRADE_REPLAY_OUTPUT_DIR")
	if root == "" {
		root = t.TempDir()
	}
	path := filepath.Join(root, strings.ReplaceAll(t.Name(), "/", "_"), name+".json")
	require.NoError(t, os.MkdirAll(filepath.Dir(path), 0755))
	require.NoError(t, os.WriteFile(path, append(data, '\n'), 0644))
	t.Logf("replay evidence: %s", path)
	return data
}

// TestOperatorUpgradeReplay requires unchanged workloads to retain their historical worker templates.
func TestOperatorUpgradeReplay(t *testing.T) {
	t.Log("Validate the frozen release catalog and source-generated snapshots")
	catalog := loadUpgradeReplayCatalog(t)
	for _, profile := range catalog.Profiles {
		t.Run(profile.Name, func(t *testing.T) {
			t.Logf("Operator %s -> candidate %s; runtime image unchanged: %s", profile.Release, catalog.CandidateMinor, profile.Image)
			t.Log("Seed the candidate with the released DGD and PCS, including their worker hash")
			snapshot, pcs := loadUpgradeReplaySnapshot(t, catalog, profile)
			originalSpec := snapshot.DGD.Spec.DeepCopy()
			before := upgradeReplayTemplates(t, snapshot.PCS)
			r, c := newUpgradeReplayReconciler(t, &snapshot.DGD, pcs)
			desiredHash, err := dynamo.ComputeDGDWorkersSpecHash(&snapshot.DGD)
			require.NoError(t, err)
			writeUpgradeReplayArtifact(t, "baseline", snapshot)
			writeUpgradeReplayArtifact(t, "hashes", map[string]string{"release": profile.Release, "revision": snapshot.Revision, "current": snapshot.DGD.Annotations[consts.AnnotationCurrentWorkerHashV2], "desired": desiredHash})

			t.Log("Run actual transition planning, rendering, PCS synchronization and hash projection repeatedly")
			for round := 1; round <= 3; round++ {
				var current api.DynamoGraphDeployment
				require.NoError(t, c.Get(t.Context(), client.ObjectKeyFromObject(&snapshot.DGD), &current))
				_, err = r.Reconcile(t.Context(), groveReconcileRequest{DGD: &current}, nil, nil)
				require.NoError(t, err)
				var after grove.PodCliqueSet
				require.NoError(t, c.Get(t.Context(), client.ObjectKeyFromObject(pcs), &after))
				var allPCS grove.PodCliqueSetList
				require.NoError(t, c.List(t.Context(), &allPCS))
				require.Len(t, allPCS.Items, 1, "takeover must not create a second PCS")
				require.Equal(t, pcs.UID, after.UID, "takeover must preserve PCS identity")
				data := writeUpgradeReplayArtifact(t, fmt.Sprintf("round-%d-pcs", round), &after)
				require.Equal(t, originalSpec, &current.Spec, "operator-only upgrade changed user intent")
				if diff := cmp.Diff(before, upgradeReplayTemplates(t, data)); diff != "" {
					t.Errorf("round %d: unchanged DGD and runtime changed worker templates (-released +candidate):\n%s", round, diff)
				}
			}
		})
	}
}

// A real worker edit must still advance the generation after historical state is adopted.
func TestOperatorUpgradeReplayIntentionalChange(t *testing.T) {
	t.Log("Load a released suffixed generation for a real worker edit")
	catalog := loadUpgradeReplayCatalog(t)
	var profile upgradeReplayProfile
	for _, candidate := range catalog.Profiles {
		if candidate.Name == "operator-1.5-to-candidate_sglang-runtime-1.5-fixed" {
			profile = candidate
		}
	}
	require.NotEmpty(t, profile.Name)
	snapshot, pcs := loadUpgradeReplaySnapshot(t, catalog, profile)
	oldHash := snapshot.DGD.Annotations[consts.AnnotationCurrentWorkerHashV2]
	snapshot.DGD.Spec.Components[0].PodTemplate.Spec.Containers[0].Env = append(snapshot.DGD.Spec.Components[0].PodTemplate.Spec.Containers[0].Env, corev1.EnvVar{Name: "WORKER_REVISION", Value: "2"})
	newHash, err := dynamo.ComputeDGDWorkersSpecHash(&snapshot.DGD)
	require.NoError(t, err)
	require.NotEqual(t, oldHash, newHash)
	r, c := newUpgradeReplayReconciler(t, &snapshot.DGD, pcs)

	t.Log("Reconcile the intentional edit repeatedly and verify convergence with the runtime image unchanged")
	var current api.DynamoGraphDeployment
	var stable map[string]any
	for round := 0; round < 3; round++ {
		require.NoError(t, c.Get(t.Context(), client.ObjectKeyFromObject(&snapshot.DGD), &current))
		_, err = r.Reconcile(t.Context(), groveReconcileRequest{DGD: &current}, nil, nil)
		require.NoError(t, err)
		var after grove.PodCliqueSet
		require.NoError(t, c.Get(t.Context(), client.ObjectKeyFromObject(pcs), &after))
		for _, clique := range after.Spec.Template.Cliques {
			require.Equal(t, newHash, clique.Labels[consts.KubeLabelDynamoWorkerHash])
			require.Equal(t, profile.Image, clique.Spec.PodSpec.Containers[0].Image)
			if clique.Labels[consts.KubeLabelDynamoComponent] == "SglangPrefill" {
				require.Contains(t, clique.Spec.PodSpec.Containers[0].Env, corev1.EnvVar{Name: "WORKER_REVISION", Value: "2"})
			}
		}
		data, err := json.Marshal(&after)
		require.NoError(t, err)
		templates := upgradeReplayTemplates(t, data)
		if stable != nil {
			require.Equal(t, stable, templates)
		}
		stable = templates
	}
	require.NoError(t, c.Get(t.Context(), client.ObjectKeyFromObject(&snapshot.DGD), &current))
	require.Equal(t, newHash, current.Annotations[consts.AnnotationCurrentWorkerHashV2])
}
