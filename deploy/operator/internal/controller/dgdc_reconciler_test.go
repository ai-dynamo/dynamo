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

package controller

import (
	"context"
	"strings"
	"testing"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
)

// -- dgdcName -----------------------------------------------------------

func TestDgdcNameJoinsOwnerAndIdentity(t *testing.T) {
	got := dgdcName("my-dgdr", "abc123def456")
	want := "my-dgdr-abc123def456"
	if got != want {
		t.Errorf("dgdcName() = %q, want %q", got, want)
	}
}

func TestDgdcNameTruncatesLongOwnerNameButKeepsFullIdentity(t *testing.T) {
	// A DGDR name near the 253-char Kubernetes object-name ceiling: the
	// owner part must be truncated, but the identity (what makes Create
	// idempotent) must survive intact so two reconciles of the same
	// candidate still agree on the name.
	longOwner := strings.Repeat("a", 240)
	identity := "0123456789abcdef"

	got := dgdcName(longOwner, identity)

	if len(got) > 253 {
		t.Errorf("dgdcName() produced a %d-char name, exceeds Kubernetes' 253-char limit", len(got))
	}
	if !strings.HasSuffix(got, "-"+identity) {
		t.Errorf("dgdcName() = %q, must end with the full, untruncated identity %q", got, identity)
	}
	if strings.Contains(got, "--") {
		t.Errorf("dgdcName() = %q, truncation left a double hyphen at the seam", got)
	}
}

func TestDgdcNameIsDeterministic(t *testing.T) {
	// Idempotent Create depends on this: the same (owner, identity) pair
	// must always produce the same name, run after run.
	a := dgdcName("sweeper-run-1", "deadbeefcafe0001")
	b := dgdcName("sweeper-run-1", "deadbeefcafe0001")
	if a != b {
		t.Errorf("dgdcName() not deterministic: %q != %q", a, b)
	}
}

// -- mapOutputConfigMapToDGDRRequest -------------------------------------

func TestMapOutputConfigMapToDGDRRequestExtractsDGDRName(t *testing.T) {
	cm := &corev1.ConfigMap{ObjectMeta: metav1.ObjectMeta{
		Name:      ConfigMapOutputPrefix + "my-sweep",
		Namespace: "ns-a",
	}}

	reqs := mapOutputConfigMapToDGDRRequest(context.Background(), cm)

	if len(reqs) != 1 {
		t.Fatalf("got %d requests, want 1", len(reqs))
	}
	if reqs[0].Name != "my-sweep" || reqs[0].Namespace != "ns-a" {
		t.Errorf("got %+v, want {my-sweep ns-a}", reqs[0])
	}
}

func TestMapOutputConfigMapToDGDRRequestIgnoresUnrelatedConfigMaps(t *testing.T) {
	cm := &corev1.ConfigMap{ObjectMeta: metav1.ObjectMeta{
		Name:      "some-other-configmap",
		Namespace: "ns-a",
	}}

	reqs := mapOutputConfigMapToDGDRRequest(context.Background(), cm)

	if reqs != nil {
		t.Errorf("got %+v, want nil for a ConfigMap without the %q prefix", reqs, ConfigMapOutputPrefix)
	}
}

func TestMapOutputConfigMapToDGDRRequestIgnoresBarePrefixWithNoName(t *testing.T) {
	// The prefix with nothing after it isn't a valid DGDR name -- must not
	// enqueue a request for an empty NamespacedName.
	cm := &corev1.ConfigMap{ObjectMeta: metav1.ObjectMeta{
		Name:      ConfigMapOutputPrefix,
		Namespace: "ns-a",
	}}

	reqs := mapOutputConfigMapToDGDRRequest(context.Background(), cm)

	if reqs != nil {
		t.Errorf("got %+v, want nil for the bare prefix with no DGDR name", reqs)
	}
}

// -- toStringAnyMap -------------------------------------------------------

func TestToStringAnyMapRoundTripsPopulatedFields(t *testing.T) {
	spec := nvidiacomv1beta1.DynamoGraphDeploymentSpec{
		BackendFramework: "vllm",
		Labels:           map[string]string{"team": "sweeper"},
	}

	got, err := toStringAnyMap(spec)
	if err != nil {
		t.Fatalf("toStringAnyMap() error = %v", err)
	}

	if got["backendFramework"] != "vllm" {
		t.Errorf("got backendFramework = %v, want %q", got["backendFramework"], "vllm")
	}
	labels, ok := got["labels"].(map[string]any)
	if !ok {
		t.Fatalf("got labels = %T, want map[string]any", got["labels"])
	}
	if labels["team"] != "sweeper" {
		t.Errorf("got labels[team] = %v, want %q", labels["team"], "sweeper")
	}
}

func TestToStringAnyMapOmitsUnsetOptionalFields(t *testing.T) {
	// omitempty fields on the zero-value spec (priorityClassName, components,
	// restart, ...) must not appear -- they'd otherwise make an empty spec's
	// identity differ from another empty spec built a different way.
	got, err := toStringAnyMap(nvidiacomv1beta1.DynamoGraphDeploymentSpec{})
	if err != nil {
		t.Fatalf("toStringAnyMap() error = %v", err)
	}
	if _, present := got["priorityClassName"]; present {
		t.Errorf("got priorityClassName present in %v, want omitted for the zero value", got)
	}
	if _, present := got["components"]; present {
		t.Errorf("got components present in %v, want omitted for the zero value", got)
	}
}

// -- desiredCandidatesFromSnapshot ----------------------------------------

func manifestYAML(backendFramework string) string {
	return "apiVersion: nvidia.com/v1beta1\n" +
		"kind: DynamoGraphDeployment\n" +
		"metadata:\n" +
		"  name: sweeper-dgd-" + backendFramework + "\n" +
		"spec:\n" +
		"  backendFramework: " + backendFramework + "\n"
}

func TestDesiredCandidatesFromSnapshotEmptyCandidatesIsNotAnError(t *testing.T) {
	desired, err := desiredCandidatesFromSnapshot(&sweeperStatusSnapshot{Status: "running"})
	if err != nil {
		t.Fatalf("desiredCandidatesFromSnapshot() error = %v", err)
	}
	if len(desired) != 0 {
		t.Errorf("got %d desired candidates, want 0", len(desired))
	}
}

func TestDesiredCandidatesFromSnapshotSkipsMaterializationFailedEntries(t *testing.T) {
	snapshot := &sweeperStatusSnapshot{
		Candidates: []sweeperCandidateEntry{
			{ID: "c-bad", Outcome: sweeperCandidateOutcomeMaterializationFailed, Error: "renderer blew up"},
			{ID: "c-ok", Outcome: sweeperCandidateOutcomeMaterialized, Manifest: manifestYAML("vllm")},
		},
	}

	desired, err := desiredCandidatesFromSnapshot(snapshot)
	if err != nil {
		t.Fatalf("desiredCandidatesFromSnapshot() error = %v", err)
	}
	if len(desired) != 1 {
		t.Fatalf("got %d desired candidates, want 1 (the materialization_failed entry must be skipped)", len(desired))
	}
	if desired[0].Spec["backendFramework"] != "vllm" {
		t.Errorf("got backendFramework = %v, want %q", desired[0].Spec["backendFramework"], "vllm")
	}
}

func TestDesiredCandidatesFromSnapshotRanksByPositionAmongMaterializedOnly(t *testing.T) {
	// A materialization_failed entry sitting between two materialized ones
	// must not consume a rank slot -- rank counts only candidates that are
	// actually desired.
	snapshot := &sweeperStatusSnapshot{
		Candidates: []sweeperCandidateEntry{
			{ID: "c-first", Outcome: sweeperCandidateOutcomeMaterialized, Manifest: manifestYAML("vllm")},
			{ID: "c-bad", Outcome: sweeperCandidateOutcomeMaterializationFailed, Error: "boom"},
			{ID: "c-second", Outcome: sweeperCandidateOutcomeMaterialized, Manifest: manifestYAML("sglang")},
		},
	}

	desired, err := desiredCandidatesFromSnapshot(snapshot)
	if err != nil {
		t.Fatalf("desiredCandidatesFromSnapshot() error = %v", err)
	}
	if len(desired) != 2 {
		t.Fatalf("got %d desired candidates, want 2", len(desired))
	}
	if desired[0].Rank == nil || *desired[0].Rank != 1 {
		t.Errorf("got first candidate rank = %v, want 1", desired[0].Rank)
	}
	if desired[1].Rank == nil || *desired[1].Rank != 2 {
		t.Errorf("got second candidate rank = %v, want 2", desired[1].Rank)
	}
}

func TestDesiredCandidatesFromSnapshotRejectsMaterializedEntryWithNoManifest(t *testing.T) {
	// Should never happen given kube_status.CandidateStatusEntry's own
	// __post_init__ validation on the Python side, but the controller reads
	// an untrusted relayed blob and must not panic or silently build an
	// empty-Spec DGDC from it.
	snapshot := &sweeperStatusSnapshot{
		Candidates: []sweeperCandidateEntry{
			{ID: "c-broken", Outcome: sweeperCandidateOutcomeMaterialized, Manifest: ""},
		},
	}

	_, err := desiredCandidatesFromSnapshot(snapshot)
	if err == nil {
		t.Fatal("desiredCandidatesFromSnapshot() error = nil, want an error for a materialized entry with no manifest")
	}
	if !strings.Contains(err.Error(), "c-broken") {
		t.Errorf("error %q should name the offending candidate %q", err.Error(), "c-broken")
	}
}

func TestDesiredCandidatesFromSnapshotRejectsUnparsableManifest(t *testing.T) {
	snapshot := &sweeperStatusSnapshot{
		Candidates: []sweeperCandidateEntry{
			{ID: "c-corrupt", Outcome: sweeperCandidateOutcomeMaterialized, Manifest: "{ this is not: valid: yaml: [["},
		},
	}

	_, err := desiredCandidatesFromSnapshot(snapshot)
	if err == nil {
		t.Fatal("desiredCandidatesFromSnapshot() error = nil, want an error for an unparsable manifest")
	}
}
