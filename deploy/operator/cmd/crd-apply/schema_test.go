/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package main

import (
	"bytes"
	"encoding/json"
	"os/exec"
	"reflect"
	"testing"

	"sigs.k8s.io/yaml"
)

func TestLPXSchemaCompaction(t *testing.T) {
	// Exercise native validation and merge metadata without copying an entire generated CRD.
	const podSchema = `
description: A role-specific Pod template.
type: object
required: [spec]
properties:
  metadata:
    type: object
    properties:
      labels:
        description: Pod labels.
        type: object
        additionalProperties: {type: string}
  spec:
    description: Native PodSpec.
    type: object
    required: [containers]
    properties:
      containers:
        type: array
        x-kubernetes-list-type: map
        x-kubernetes-list-map-keys: [name]
        items:
          type: object
          required: [name]
          x-kubernetes-validations:
            - rule: self.name != ''
              message: Name must not be empty.
          properties:
            name: {type: string}
            image: {type: string}
      restartPolicy: {type: string, default: Always, enum: [Always, Never]}
`
	// Keep absence, equal schemas, and role-specific constraints in one behavioral matrix.
	for _, test := range []struct {
		name      string
		withLPX   bool
		different bool
		flow      bool
		aliases   int
	}{
		{name: "no LPX"},
		{name: "identical schemas", withLPX: true, aliases: 1},
		{name: "identical schemas on one line", withLPX: true, flow: true, aliases: 1},
		{name: "different constraints", withLPX: true, different: true},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Render role templates in two API versions alongside unchanged ordinary schemas")
			var alphaRoles, betaRoles, ordinary, undocumented map[string]any
			for _, target := range []*map[string]any{&alphaRoles, &betaRoles, &ordinary, &undocumented} {
				if err := yaml.Unmarshal([]byte(podSchema), target); err != nil {
					t.Fatal(err)
				}
			}
			if test.different {
				betaRoles["required"] = []any{"metadata", "spec"}
			}
			removeSchemaDescriptions(undocumented)
			input := map[string]any{
				"podTemplate":  ordinary,
				"extraPodSpec": map[string]any{"description": "Alpha documentation.", "properties": ordinary},
				"unrelated":    map[string]any{"podTemplate": undocumented},
			}
			if test.withLPX {
				input["versions"] = []any{
					map[string]any{"properties": map[string]any{
						"lpx": map[string]any{"type": "object"},
						"roles": map[string]any{"items": map[string]any{"properties": map[string]any{
							"podTemplate": alphaRoles,
						}}},
					}},
					map[string]any{"properties": map[string]any{
						"lpx": map[string]any{"type": "object"},
						"roles": map[string]any{"items": map[string]any{"properties": map[string]any{
							"podTemplate": betaRoles,
						}}},
					}},
				}
			}
			original, err := yaml.Marshal(input)
			if test.flow {
				original, err = json.Marshal(input)
			}
			if err != nil {
				t.Fatal(err)
			}

			t.Log("Run the production transformation and decode its aliases using the installer YAML library")
			output := compactLPXSchema(t, original)
			var actual map[string]any
			if err := yaml.Unmarshal(output, &actual); err != nil {
				t.Fatalf("installer cannot decode compacted schema: %v\n%s", err, output)
			}
			if test.withLPX {
				removeSchemaDescriptions(alphaRoles)
				removeSchemaDescriptions(betaRoles)
			}
			if !reflect.DeepEqual(actual, input) {
				t.Fatalf("schema semantics changed beyond LPX descriptions:\n%s", output)
			}

			t.Log("Check alias scope and physical order, including an ordinary identical schema")
			if got := bytes.Count(output, []byte("*lpx-pod-template")); got != test.aliases {
				t.Errorf("alias count = %d, want %d", got, test.aliases)
			}
			anchor := bytes.Index(output, []byte("&lpx-pod-template"))
			alias := bytes.Index(output, []byte("*lpx-pod-template"))
			if test.aliases > 0 && anchor > alias {
				t.Error("alias precedes its anchor")
			}
			if !test.withLPX && bytes.Contains(output, []byte("&lpx-pod-template")) {
				t.Error("ordinary schema gained an LPX anchor")
			}

			t.Log("Require a stable second transformation")
			if again := compactLPXSchema(t, output); !bytes.Equal(output, again) {
				t.Errorf("second transformation changed the YAML:\n%s", again)
			}
		})
	}
}

func compactLPXSchema(t *testing.T, input []byte) []byte {
	t.Helper()

	// Use the same pinned yq binary and expression as make manifests.
	command := exec.CommandContext(t.Context(), "../../bin/yq",
		"--from-file", "../../api/scripts/lpx-crd-schema.yq", "--indent", "2")
	command.Stdin = bytes.NewReader(input)
	output, err := command.CombinedOutput()
	if err != nil {
		t.Fatalf("compact LPX schema (run make yq first): %v\n%s", err, output)
	}
	return output
}

func removeSchemaDescriptions(value any) {
	// Remove documentation recursively without interpreting or changing any validation marker.
	switch value := value.(type) {
	case map[string]any:
		delete(value, "description")
		for _, child := range value {
			removeSchemaDescriptions(child)
		}
	case []any:
		for _, child := range value {
			removeSchemaDescriptions(child)
		}
	}
}
