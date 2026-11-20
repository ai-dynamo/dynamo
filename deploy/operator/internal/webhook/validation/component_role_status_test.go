// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package validation

import (
	"testing"

	"github.com/stretchr/testify/require"
	fieldvalidation "k8s.io/apimachinery/pkg/util/validation/field"
)

func TestComponentRoleStatusEnvironmentSchema(t *testing.T) {
	// The spec-admission harness removes status; validate this status-only CEL
	// contract directly against its generated schema in both native versions.
	validators := requestValidatorsFromCRD(t, "nvidia.com_dynamocomponentdeployments.yaml")
	for _, version := range []string{"v1alpha1", "v1beta1"} {
		for _, test := range []struct {
			name    string
			binding map[string]any
			valid   bool
		}{
			{name: "empty literal", binding: map[string]any{"value": ""}, valid: true},
			{name: "unknown", binding: map[string]any{"unresolved": true}, valid: true},
			{name: "empty binding", binding: map[string]any{}},
			{name: "false unknown", binding: map[string]any{"unresolved": false}},
			{name: "both alternatives", binding: map[string]any{"value": "secret", "unresolved": true}},
		} {
			t.Run(version+"/"+test.name, func(t *testing.T) {
				t.Log("Validate the generated native status schema rather than a handwritten copy of its rule")
				component := map[string]any{"componentKind": "Deployment", "replicas": int64(0), "updatedReplicas": int64(0), "roles": []any{
					map[string]any{"name": "main", "containers": []any{map[string]any{"name": "main", "image": "engine:test", "env": map[string]any{"MODE": test.binding}}}},
				}}
				field := "component"
				if version == "v1alpha1" {
					field = "service"
					component["componentName"] = "worker"
				}
				object := map[string]any{"apiVersion": "nvidia.com/" + version, "kind": "DynamoComponentDeployment", "metadata": map[string]any{"name": "worker"}, "status": map[string]any{field: component}}
				errs := validators[version].celValidator(object, nil)
				if test.valid {
					require.Empty(t, errs)
				} else {
					require.Len(t, errs, 1)
					require.Equal(t, fieldvalidation.ErrorTypeInvalid, errs[0].Type)
					require.Equal(t, "status."+field+".roles[0].containers[0].env[MODE]", errs[0].Field)
					require.Contains(t, errs[0].Detail, "specify either value or unresolved: true")
				}
			})
		}
	}
}
