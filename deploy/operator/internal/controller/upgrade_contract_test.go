// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package controller

import (
	"crypto/sha256"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"

	api "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	common "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/testing/golden"
	"github.com/google/go-cmp/cmp"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
)

// Keep fixture kind names independent of helpers added after the release cut.
const (
	upgradePodCliqueSetKind          = "PodCliqueSet"
	upgradeDynamoGraphDeploymentKind = "DynamoGraphDeployment"
)

// Output-only metadata can be reviewed in goldens; generation selection and the
// effective pod templates are checked independently against historical state.
func upgradeSafetyError(generationChanged bool, before, after []unstructured.Unstructured) error {
	if generationChanged {
		return fmt.Errorf("operator-only takeover selected a new worker generation")
	}
	index := map[string]unstructured.Unstructured{}
	for _, object := range after {
		index[object.GetKind()+"/"+object.GetName()] = object
	}
	oldWorkloads, newWorkloads := 0, 0
	for _, object := range before {
		if isUpgradeWorkload(object.GetKind()) {
			oldWorkloads++
		}
	}
	for _, object := range after {
		if isUpgradeWorkload(object.GetKind()) {
			newWorkloads++
		}
	}
	if oldWorkloads != newWorkloads {
		return fmt.Errorf("active workload inventory changed")
	}
	for _, old := range before {
		if !isUpgradeWorkload(old.GetKind()) && old.GetKind() != upgradeDynamoGraphDeploymentKind {
			continue
		}
		current, ok := index[old.GetKind()+"/"+old.GetName()]
		if !ok || old.GetUID() != current.GetUID() {
			return fmt.Errorf("active identity changed: %s/%s", old.GetKind(), old.GetName())
		}
		switch old.GetKind() {
		case upgradeDynamoGraphDeploymentKind:
			const origin = "nvidia.com/dynamo-operator-origin-version"
			oldOrigin, oldExists := old.GetAnnotations()[origin]
			newOrigin, newExists := current.GetAnnotations()[origin]
			if oldOrigin != newOrigin || oldExists != newExists {
				return fmt.Errorf("historical origin changed")
			}
			// A golden refresh cannot approve a newly requested rollout or restart.
			for _, key := range []string{"rollingUpdate", "restart"} {
				a, _, err := unstructured.NestedFieldCopy(old.Object, "status", key)
				if err != nil {
					return err
				}
				b, _, err := unstructured.NestedFieldCopy(current.Object, "status", key)
				if err != nil {
					return err
				}
				if diff := cmp.Diff(a, b); diff != "" {
					return fmt.Errorf("DGD rollout intent changed: %s", diff)
				}
			}
		case "Deployment", "LeaderWorkerSet", "DisaggregatedSet":
			if err := upgradeComponentWorkloadSafetyError(old, current); err != nil {
				return err
			}
		case upgradePodCliqueSetKind:
			a, err := upgradeWorkerTemplates(old)
			if err != nil {
				return err
			}
			b, err := upgradeWorkerTemplates(current)
			if err != nil {
				return err
			}
			if diff := cmp.Diff(a, b); diff != "" {
				return fmt.Errorf("effective worker template or identity changed: %s", diff)
			}
			// Spec-level restart requests also matter even when the pod template is equal.
			for _, key := range []string{"updateStrategy"} {
				a, _, _ := unstructured.NestedFieldCopy(old.Object, "spec", key)
				b, _, _ := unstructured.NestedFieldCopy(current.Object, "spec", key)
				if diff := cmp.Diff(a, b); diff != "" {
					return fmt.Errorf("Grove rollout policy changed: %s", diff)
				}
			}
		}
	}
	return nil
}

// Component workloads preserve both their generation and complete effective spec.
func upgradeComponentWorkloadSafetyError(old, current unstructured.Unstructured) error {
	if old.GetGeneration() != current.GetGeneration() {
		return fmt.Errorf("workload generation changed: %s/%s", old.GetKind(), old.GetName())
	}
	a, _, err := unstructured.NestedMap(old.Object, "spec")
	if err != nil {
		return err
	}
	b, _, err := unstructured.NestedMap(current.Object, "spec")
	if err != nil {
		return err
	}
	if diff := cmp.Diff(a, b); diff != "" {
		return fmt.Errorf("effective worker template or rollout intent changed: %s", diff)
	}
	return nil
}

func isUpgradeWorkload(kind string) bool {
	return kind == upgradePodCliqueSetKind || kind == "Deployment" || kind == "LeaderWorkerSet" || kind == "DisaggregatedSet"
}

func upgradeWorkerTemplates(object unstructured.Unstructured) (map[string]any, error) {
	cliques, found, err := unstructured.NestedSlice(object.Object, "spec", "template", "cliques")
	if err != nil || !found {
		return nil, fmt.Errorf("missing Grove clique templates: %v", err)
	}
	templates := map[string]any{}
	for _, value := range cliques {
		clique, ok := value.(map[string]any)
		if !ok {
			return nil, fmt.Errorf("invalid clique")
		}
		name, ok := clique["name"].(string)
		if !ok {
			return nil, fmt.Errorf("missing clique name")
		}
		spec, ok := clique["spec"].(map[string]any)
		if !ok {
			return nil, fmt.Errorf("missing clique spec")
		}
		templates[name] = map[string]any{"labels": clique["labels"], "annotations": clique["annotations"], "podSpec": spec["podSpec"]}
	}
	return templates, nil
}

func TestUpgradeSafetyIndependentOfGolden(t *testing.T) {
	t.Log("A reviewed resource-metadata default is allowed without weakening the rollout contract")
	old := unstructured.Unstructured{Object: map[string]any{
		"apiVersion": "grove.io/v1alpha1", "kind": upgradePodCliqueSetKind,
		"metadata": map[string]any{"name": "workers", "uid": "preserved"},
		"spec": map[string]any{"template": map[string]any{"cliques": []any{map[string]any{
			"name": "decode", "spec": map[string]any{"podSpec": map[string]any{"hostname": "worker"}},
		}}}},
	}}
	before := []unstructured.Unstructured{old}
	metadata := old.DeepCopy()
	metadata.SetAnnotations(map[string]string{"new-default": "true"})
	require.NoError(t, upgradeSafetyError(false, before, []unstructured.Unstructured{*metadata}))
	t.Log("A changed transition decision fails even when output is byte-for-byte equal")
	require.ErrorContains(t, upgradeSafetyError(true, before, before), "new worker generation")
	t.Log("Accepting new output cannot bless an effective template mutation or identity replacement")
	changed := old.DeepCopy()
	cliques, _, err := unstructured.NestedSlice(changed.Object, "spec", "template", "cliques")
	require.NoError(t, err)
	cliques[0].(map[string]any)["spec"].(map[string]any)["podSpec"].(map[string]any)["hostname"] = "new-worker"
	require.NoError(t, unstructured.SetNestedSlice(changed.Object, cliques, "spec", "template", "cliques"))
	require.ErrorContains(t, upgradeSafetyError(false, before, []unstructured.Unstructured{*changed}), "effective worker template")
	changed = old.DeepCopy()
	changed.SetUID("replacement")
	require.ErrorContains(t, upgradeSafetyError(false, before, []unstructured.Unstructured{*changed}), "active identity")
}

// Namespace is fixed before admission so generated names and hashes belong to the captured state.
func upgradeScenarioNamespace(relative string) string {
	name := "upgrade-" + regexp.MustCompile(`[^a-z0-9]+`).ReplaceAllString(strings.ToLower(filepath.ToSlash(relative)), "-")
	if len(name) > 63 {
		sum := sha256.Sum256([]byte(relative))
		name = strings.TrimRight(name[:50], "-") + fmt.Sprintf("-%x", sum[:6])
	}
	return name
}

func TestUpgradeScenarioNamespaces(t *testing.T) {
	t.Log("Derive separate stable namespaces for complete latest paths and each promoted version")
	paths := []string{
		"latest/sglang-1.4",
		"history/1.6.0/sglang/aggregated/single/grove",
		"history/1.7.0/sglang/aggregated/single/grove",
		"latest/sglang/aggregated/single/grove",
		"latest/sglang/disaggregated/multi/disaggregatedset",
		"latest/" + strings.Repeat("long/", 20) + "grove",
		"latest/" + strings.Repeat("long/", 20) + "component",
	}
	seen := map[string]bool{}
	for _, path := range paths {
		name := upgradeScenarioNamespace(path)
		require.LessOrEqual(t, len(name), 63)
		require.Regexp(t, `^[a-z0-9]([-a-z0-9]*[a-z0-9])?$`, name)
		require.Equal(t, name, upgradeScenarioNamespace(path))
		require.NotContains(t, seen, name)
		seen[name] = true
	}
	require.Equal(t, "upgrade-history-1-6-0-sglang-aggregated-single-grove", upgradeScenarioNamespace(paths[1]))
}

// Hash expectations come from the frozen golden spec, never from candidate output.
func upgradeLatestVariables(t *testing.T, path, version string) golden.Variables {
	t.Helper()
	variables := golden.Variables{"operator-version": version, "worker-hash": "pending", "pcs-spec-hash": "pending", "ds-revision": "pending"}
	raw, err := golden.ReadManifests(path)
	require.NoError(t, err)
	for _, object := range raw {
		upgradeWorkerStringVariables(object.Object, "pending", variables)
	}
	objects, err := golden.ReadManifestsWithVariables(path, variables)
	require.NoError(t, err)

	// Bind the version-dependent worker identity using the reviewed DGD input.
	found := false
	for _, object := range objects {
		if object.GetKind() != upgradeDynamoGraphDeploymentKind {
			continue
		}
		dgd := api.DynamoGraphDeployment{
			Name: object.GetName(), Namespace: object.GetNamespace(),
			Labels: object.GetLabels(), Annotations: object.GetAnnotations()}
		spec, _, err := unstructured.NestedMap(object.Object, "spec")
		require.NoError(t, err)
		require.NoError(t, runtime.DefaultUnstructuredConverter.FromUnstructured(spec, &dgd.Spec))
		variables["worker-hash"], err = dynamo.ComputeDGDWorkersSpecHash(&dgd)
		require.NoError(t, err)
		found = true
	}
	require.True(t, found, "latest golden must include its DGD")
	for _, object := range raw {
		upgradeWorkerStringVariables(object.Object, variables["worker-hash"], variables)
	}

	// Derive provider revisions from reviewed templates after rebinding worker identity.
	objects, err = golden.ReadManifestsWithVariables(path, variables)
	require.NoError(t, err)
	for _, object := range objects {
		if object.GetKind() == upgradePodCliqueSetKind {
			variables["pcs-spec-hash"], err = common.GetSpecHash(&object)
			require.NoError(t, err)
		}
		if object.GetKind() == "DisaggregatedSet" {
			spec := unstructured.Unstructured{Object: map[string]any{"spec": object.Object["spec"]}}
			variables["ds-revision"], err = disaggregatedSetTargetRevision(&spec)
			require.NoError(t, err)
		}
	}
	return variables
}

func TestUpgradeLatestVersionVariables(t *testing.T) {
	t.Log("Version-dependent identities remain exact bindings derived from the reviewed golden")
	path := "testdata/upgrade/latest/sglang-1.5/output.yaml"
	previous := upgradeLatestVariables(t, path, "1.6.0")
	current := upgradeLatestVariables(t, path, "1.7.0")
	require.Equal(t, "1.7.0", current["operator-version"])
	require.NotEqual(t, previous["worker-hash"], current["worker-hash"])
	require.NotEqual(t, previous["pcs-spec-hash"], current["pcs-spec-hash"])

	t.Log("A reviewed runtime change affects both expected hashes without reading candidate output")
	data, err := os.ReadFile(path)
	require.NoError(t, err)
	updated := filepath.Join(t.TempDir(), "output.yaml")
	require.NoError(t, os.WriteFile(updated, []byte(strings.ReplaceAll(string(data), "sglang-runtime:1.5.0", "sglang-runtime:1.5.1")), 0600))
	changed := upgradeLatestVariables(t, updated, "1.7.0")
	require.NotEqual(t, current["worker-hash"], changed["worker-hash"])
	require.NotEqual(t, current["pcs-spec-hash"], changed["pcs-spec-hash"])
}

// Component names and status messages embed the version-dependent worker hash.
func upgradeWorkerStringVariables(value any, hash string, variables golden.Variables) {
	switch value := value.(type) {
	case map[string]any:
		for _, child := range value {
			upgradeWorkerStringVariables(child, hash, variables)
		}
	case []any:
		for _, child := range value {
			upgradeWorkerStringVariables(child, hash, variables)
		}
	case string:
		if name, found := strings.CutPrefix(value, "$var:worker-string:"); found {
			variables["worker-string:"+name] = strings.ReplaceAll(name, "{{worker-hash}}", hash)
		}
	}
}
