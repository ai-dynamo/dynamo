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

package dgdcreconcile

import (
	"errors"
	"reflect"
	"testing"
)

func d(ids ...string) []DesiredCandidate {
	out := make([]DesiredCandidate, 0, len(ids))
	for _, id := range ids {
		out = append(out, DesiredCandidate{ID: id, Spec: "spec-" + id})
	}
	return out
}

func c(pairs ...string) []CurrentDGDC {
	var out []CurrentDGDC
	for i := 0; i+1 < len(pairs); i += 2 {
		out = append(out, CurrentDGDC{Name: pairs[i], ID: pairs[i+1]})
	}
	return out
}

func createIDs(a Actions) []string {
	var ids []string
	for _, x := range a.Creates {
		ids = append(ids, x.ID)
	}
	return ids
}

func TestComputeActions(t *testing.T) {
	tests := []struct {
		name        string
		desired     []DesiredCandidate
		current     []CurrentDGDC
		wantCreates []string
		wantDeletes []string
	}{
		{"empty both", nil, nil, nil, nil},
		{"all new", d("a", "b"), nil, []string{"a", "b"}, nil},
		{"all gone", nil, c("n-a", "a", "n-b", "b"), nil, []string{"n-a", "n-b"}},
		{"steady state needs no action", d("a", "b"), c("n-a", "a", "n-b", "b"), nil, nil},
		{"reordering desired needs no action", d("b", "a"), c("n-a", "a", "n-b", "b"), nil, nil},
		{"incomplete DGDC is ensured again", d("a", "b"), []CurrentDGDC{{Name: "n-a", ID: "a"}, {Name: "n-b", ID: "b", Incomplete: true}}, []string{"b"}, nil},
		{"mixed", d("a", "c"), c("n-a", "a", "n-b", "b"), []string{"c"}, []string{"n-b"}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := ComputeActions(tc.desired, tc.current)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if !reflect.DeepEqual(createIDs(got), tc.wantCreates) {
				t.Errorf("creates = %v, want %v", createIDs(got), tc.wantCreates)
			}
			if !reflect.DeepEqual(got.Deletes, tc.wantDeletes) {
				t.Errorf("deletes = %v, want %v", got.Deletes, tc.wantDeletes)
			}
		})
	}
}

func TestComputeActionsInputErrors(t *testing.T) {
	tests := []struct {
		name    string
		desired []DesiredCandidate
		current []CurrentDGDC
	}{
		{"duplicate desired", d("a", "a"), nil},
		{"empty desired id", d(""), nil},
		{"duplicate current", nil, c("n1", "a", "n2", "a")},
		{"empty current id", nil, c("n1", "")},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			_, err := ComputeActions(tc.desired, tc.current)
			var inputErr *DiffInputError
			if !errors.As(err, &inputErr) {
				t.Fatalf("want *DiffInputError, got %v", err)
			}
		})
	}
}

func TestComputeActionsDeterministicOrder(t *testing.T) {
	got, err := ComputeActions(d("z", "m", "a"), c("n-q", "q", "n-b", "b"))
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(createIDs(got), []string{"z", "m", "a"}) {
		t.Errorf("creates order = %v", createIDs(got))
	}
	if !reflect.DeepEqual(got.Deletes, []string{"n-q", "n-b"}) {
		t.Errorf("deletes order = %v", got.Deletes)
	}
}
