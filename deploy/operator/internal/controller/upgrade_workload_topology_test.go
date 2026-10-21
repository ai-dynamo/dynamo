// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package controller

import (
	"fmt"
	"testing"

	api "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
)

// Compare every multinode component's rendered LWS size against its DGD intent, not its golden.
func upgradeLWSSizeError(dgd *api.DynamoGraphDeployment, objects []unstructured.Unstructured) error {
	for _, component := range dgd.Spec.Components {
		if !component.IsMultinode() {
			continue
		}

		// Match the logical component even when workload names carry a worker hash.
		matches := 0
		for _, object := range objects {
			if object.GetKind() != "LeaderWorkerSet" || object.GetLabels()[consts.KubeLabelDynamoComponent] != component.ComponentName {
				continue
			}
			matches++
			size, found, err := unstructured.NestedInt64(object.Object, "spec", "leaderWorkerTemplate", "size")
			if err != nil {
				return fmt.Errorf("LeaderWorkerSet %s for component %s has invalid size: %w", object.GetName(), component.ComponentName, err)
			}
			if !found {
				return fmt.Errorf("LeaderWorkerSet %s for component %s has no leaderWorkerTemplate.size", object.GetName(), component.ComponentName)
			}
			if size != int64(component.Multinode.NodeCount) {
				return fmt.Errorf("LeaderWorkerSet %s for component %s has size %d, want %d", object.GetName(), component.ComponentName, size, component.Multinode.NodeCount)
			}
		}
		if matches != 1 {
			return fmt.Errorf("component %s has %d LeaderWorkerSets, want 1", component.ComponentName, matches)
		}
	}
	return nil
}

func TestUpgradeLWSSize(t *testing.T) {
	for _, test := range []struct {
		name      string
		sizes     []any
		duplicate bool
		want      string
	}{
		{name: "correct", sizes: []any{int64(2), int64(3)}},
		{name: "lowered-to-single-node", sizes: []any{int64(1), int64(3)}, want: "has size 1, want 2"},
		{name: "per-component-size", sizes: []any{int64(2), int64(2)}, want: "has size 2, want 3"},
		{name: "missing-size", sizes: []any{int64(2), nil}, want: "has no leaderWorkerTemplate.size"},
		{name: "invalid-size", sizes: []any{int64(2), "3"}, want: "has invalid size"},
		{name: "missing-component", sizes: []any{int64(2)}, want: "component decode has 0 LeaderWorkerSets"},
		{name: "duplicate-component", sizes: []any{int64(2), int64(3)}, duplicate: true, want: "component prefill has 2 LeaderWorkerSets"},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Require one correctly sized LWS for each multinode component independently of goldens")
			dgd := &api.DynamoGraphDeployment{Spec: api.DynamoGraphDeploymentSpec{
				Components: []api.DynamoComponentDeploymentSharedSpec{
					{ComponentName: "frontend", ComponentType: api.ComponentTypeFrontend},
					{ComponentName: "prefill", ComponentType: api.ComponentTypePrefill, Multinode: &api.MultinodeSpec{NodeCount: 2}},
					{ComponentName: "decode", ComponentType: api.ComponentTypeDecode, Multinode: &api.MultinodeSpec{NodeCount: 3}},
				},
			}}
			var objects []unstructured.Unstructured
			for i, size := range test.sizes {
				name := dgd.Spec.Components[i+1].ComponentName
				object := unstructured.Unstructured{Object: map[string]any{
					"kind": "LeaderWorkerSet", "metadata": map[string]any{
						"name": "workers-" + name, "labels": map[string]any{consts.KubeLabelDynamoComponent: name},
					},
				}}
				if size != nil {
					require.NoError(t, unstructured.SetNestedField(object.Object, size, "spec", "leaderWorkerTemplate", "size"))
				}
				objects = append(objects, object)
			}
			if test.duplicate {
				objects = append(objects, *objects[0].DeepCopy())
			}

			t.Log("A wrong rendered topology must fail even when that output is accepted as a golden")
			err := upgradeLWSSizeError(dgd, objects)
			if test.want == "" {
				require.NoError(t, err)
			} else {
				require.ErrorContains(t, err, test.want)
			}
		})
	}
}
