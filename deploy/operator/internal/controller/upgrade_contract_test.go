// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package controller

import (
	"fmt"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
)

const upgradePodCliqueSetKind = "PodCliqueSet"

// Output-only metadata can be reviewed in goldens; generation selection and the
// effective Grove pod templates are checked independently against historical state.
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
		if object.GetKind() == upgradePodCliqueSetKind {
			oldWorkloads++
		}
	}
	for _, object := range after {
		if object.GetKind() == upgradePodCliqueSetKind {
			newWorkloads++
		}
	}
	if oldWorkloads != newWorkloads {
		return fmt.Errorf("active workload inventory changed")
	}
	for _, old := range before {
		if old.GetKind() != upgradePodCliqueSetKind && old.GetKind() != "DynamoGraphDeployment" {
			continue
		}
		current, ok := index[old.GetKind()+"/"+old.GetName()]
		if !ok || old.GetUID() != current.GetUID() {
			return fmt.Errorf("active identity changed: %s/%s", old.GetKind(), old.GetName())
		}
		switch old.GetKind() {
		case "DynamoGraphDeployment":
			const origin = "nvidia.com/dynamo-operator-origin-version"
			oldOrigin, oldExists := old.GetAnnotations()[origin]
			newOrigin, newExists := current.GetAnnotations()[origin]
			if oldOrigin != newOrigin || oldExists != newExists {
				return fmt.Errorf("historical origin changed")
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
