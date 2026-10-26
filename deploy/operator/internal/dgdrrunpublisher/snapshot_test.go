/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package dgdrrunpublisher

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

const validManifest = "apiVersion: nvidia.com/v1beta1\nkind: DynamoGraphDeployment\nspec: {}\n"

const testRunUID = "uid-1"

func cn(id string) string { return CandidateName("run", testRunUID, id) }

type cand struct {
	id      string
	failed  bool
	metrics map[string]any
}

// snapshotJSON renders a snapshot (JSON is valid YAML, which the real parser accepts).
func snapshotJSON(t *testing.T, phase string, cands ...cand) []byte {
	t.Helper()
	var list []map[string]any
	for _, c := range cands {
		entry := map[string]any{"id": c.id, "parameters": map[string]any{"tp": 2}, "metrics": c.metrics}
		if c.failed {
			entry["outcome"] = OutcomeMaterializationFailed
			entry["error"] = "boom"
		} else {
			entry["outcome"] = OutcomeMaterialized
			entry["manifest"] = validManifest
		}
		list = append(list, entry)
	}
	data, err := json.Marshal(map[string]any{
		"schemaVersion": 1,
		"run":           map[string]any{"phase": phase, "terminal": phase != PhaseRunning},
		"progress":      map[string]any{"round": 3, "evaluated": len(cands)},
		"candidates":    list,
	})
	if err != nil {
		t.Fatal(err)
	}
	return data
}

func writeSnapshot(t *testing.T, dir string, data []byte) {
	t.Helper()
	if err := os.WriteFile(filepath.Join(dir, SnapshotFileName), data, 0o600); err != nil {
		t.Fatal(err)
	}
}

func mustParse(t *testing.T, data []byte) *Snapshot {
	t.Helper()
	snap, err := ParseSnapshot(data)
	if err != nil {
		t.Fatal(err)
	}
	return snap
}

func withTimestamp(t *testing.T, data []byte, timestamp string) []byte {
	t.Helper()
	var doc map[string]any
	if err := json.Unmarshal(data, &doc); err != nil {
		t.Fatal(err)
	}
	doc["timestamp"] = timestamp
	out, err := json.Marshal(doc)
	if err != nil {
		t.Fatal(err)
	}
	return out
}

func withRunDiagnostics(t *testing.T, data []byte, message, runError string) []byte {
	t.Helper()
	var doc map[string]any
	if err := json.Unmarshal(data, &doc); err != nil {
		t.Fatal(err)
	}
	run := doc["run"].(map[string]any)
	run["message"] = message
	run["error"] = runError
	out, err := json.Marshal(doc)
	if err != nil {
		t.Fatal(err)
	}
	return out
}

func TestParseSnapshotValidation(t *testing.T) {
	bad := map[string]string{
		"schema":                  `{"schemaVersion":2,"run":{"phase":"Running","terminal":false}}`,
		"phase":                   `{"schemaVersion":1,"run":{"phase":"Weird","terminal":false}}`,
		"terminal flag":           `{"schemaVersion":1,"run":{"phase":"Succeeded","terminal":false}}`,
		"duplicate":               `{"schemaVersion":1,"run":{"phase":"Running","terminal":false},"candidates":[{"id":"a","outcome":"materialized","manifest":"apiVersion: nvidia.com/v1beta1\\nkind: DynamoGraphDeployment\\nspec: {}\\n"},{"id":"a","outcome":"materialized","manifest":"apiVersion: nvidia.com/v1beta1\\nkind: DynamoGraphDeployment\\nspec: {}\\n"}]}`,
		"no manifest":             `{"schemaVersion":1,"run":{"phase":"Running","terminal":false},"candidates":[{"id":"a","outcome":"materialized"}]}`,
		"materialized with error": `{"schemaVersion":1,"run":{"phase":"Running","terminal":false},"candidates":[{"id":"a","outcome":"materialized","manifest":"apiVersion: nvidia.com/v1beta1\\nkind: DynamoGraphDeployment\\nspec: {}\\n","error":"boom"}]}`,
		"no error":                `{"schemaVersion":1,"run":{"phase":"Running","terminal":false},"candidates":[{"id":"a","outcome":"materialization_failed"}]}`,
		"failed with manifest":    `{"schemaVersion":1,"run":{"phase":"Running","terminal":false},"candidates":[{"id":"a","outcome":"materialization_failed","manifest":"apiVersion: nvidia.com/v1beta1\\nkind: DynamoGraphDeployment\\nspec: {}\\n","error":"boom"}]}`,
		"unsafe id":               `{"schemaVersion":1,"run":{"phase":"Running","terminal":false},"candidates":[{"id":"Not_Valid","outcome":"materialized","manifest":"apiVersion: nvidia.com/v1beta1\\nkind: DynamoGraphDeployment\\nspec: {}\\n"}]}`,
		"long id":                 `{"schemaVersion":1,"run":{"phase":"Running","terminal":false},"candidates":[{"id":"%s","outcome":"materialized","manifest":"apiVersion: nvidia.com/v1beta1\\nkind: DynamoGraphDeployment\\nspec: {}\\n"}]}`,
		"wrong kind":              `{"schemaVersion":1,"run":{"phase":"Running","terminal":false},"candidates":[{"id":"a","outcome":"materialized","manifest":"kind: ConfigMap\\nspec: {}\\n"}]}`,
		"empty id":                `{"schemaVersion":1,"run":{"phase":"Running","terminal":false},"candidates":[{"id":"","outcome":"materialized","manifest":"apiVersion: nvidia.com/v1beta1\\nkind: DynamoGraphDeployment\\nspec: {}\\n"}]}`,
	}
	bad["long id"] = strings.Replace(bad["long id"], "%s", strings.Repeat("a", 64), 1)
	for name, doc := range bad {
		if _, err := ParseSnapshot([]byte(doc)); !errors.Is(err, ErrProtocolViolation) {
			t.Errorf("%s: want protocol violation, got %v", name, err)
		}
	}
}

func TestParseManifestHandlesCompanionResources(t *testing.T) {
	dgd := "apiVersion: nvidia.com/v1beta1\nkind: DynamoGraphDeployment\nspec: {}\n"
	cm := "apiVersion: v1\nkind: ConfigMap\nmetadata: {}\n"
	for name, manifest := range map[string]string{
		"dgd first":       dgd + "---\n" + cm,
		"configmap first": "# generated\n---\n" + cm + "---\n" + dgd,
	} {
		got, err := ParseManifest(manifest)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		if got.DGD["kind"] != "DynamoGraphDeployment" || len(got.Companions) != 1 || !strings.Contains(got.Companions[0], "ConfigMap") {
			t.Fatalf("%s: %+v", name, got)
		}
	}
	for name, manifest := range map[string]string{
		"two dgds":               dgd + "---\n" + dgd,
		"no dgd":                 cm,
		"no spec":                "apiVersion: nvidia.com/v1beta1\nkind: DynamoGraphDeployment\n",
		"no apiVersion":          "kind: DynamoGraphDeployment\nspec: {}\n",
		"unsupported apiVersion": "apiVersion: nvidia.com/v9\nkind: DynamoGraphDeployment\nspec: {}\n",
	} {
		if _, err := ParseManifest(manifest); err == nil {
			t.Errorf("%s: want error", name)
		}
	}
}
func writeProgress(t *testing.T, dir, phase string, round, evaluated int) {
	t.Helper()
	doc := map[string]any{}
	if err := json.Unmarshal(snapshotJSON(t, phase, cand{id: "a"}), &doc); err != nil {
		t.Fatal(err)
	}
	doc["progress"] = map[string]any{"round": round, "evaluated": evaluated}
	data, err := json.Marshal(doc)
	if err != nil {
		t.Fatal(err)
	}
	writeSnapshot(t, dir, data)
}
