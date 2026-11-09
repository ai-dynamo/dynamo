/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lws

import (
	"sort"
	"time"

	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
)

const (
	// FailureRecordsAnnotationKey stores the operator-owned LWS failure ledger on the DCD.
	FailureRecordsAnnotationKey = "nvidia.com/lws-failure-records"

	failureRecordsVersion = 1

	// FinishedAt-anchored retention and capacity bounds for the v1 ledger.
	failureRecordRetention        = time.Hour
	failureRecordMaxEntries       = 32
	failureRecordMaxGroups        = 16
	failureRecordMaxPerGroup      = 8
	failureRecordMaxEncodedBytes  = 32 * 1024
	failureRecordAnnotationBudget = 240 * 1024
)

// failureRecordsDocument is the versioned annotation payload.
type failureRecordsDocument struct {
	Version      int                  `json:"version"`
	DCDUID       types.UID            `json:"dcdUID"`
	PrunedBefore *metav1.Time         `json:"prunedBefore,omitempty"`
	Entries      []failureRecordEntry `json:"entries"`
}

// failureRecordEntry is one confirmed failed incarnation within the retention window.
type failureRecordEntry struct {
	DCDGeneration           int64       `json:"dcdGeneration"`
	LWSName                 string      `json:"lwsName"`
	LWSUID                  types.UID   `json:"lwsUID"`
	LWSGeneration           int64       `json:"lwsGeneration"`
	Revision                string      `json:"revision"`
	GroupIndex              int         `json:"groupIndex"`
	LeaderUID               types.UID   `json:"leaderUID"`
	PodName                 string      `json:"podName"`
	PodUID                  types.UID   `json:"podUID"`
	PodResourceVersion      string      `json:"podResourceVersion"`
	Container               string      `json:"container"`
	ExitCode                int32       `json:"exitCode"`
	FinishedAt              metav1.Time `json:"finishedAt"`
	ReadyLastTransitionTime metav1.Time `json:"readyLastTransitionTime"`
	FirstObservedAt         metav1.Time `json:"firstObservedAt"`
}

type recordIdentity struct {
	dcdGeneration int64
	lwsUID        types.UID
	lwsGeneration int64
	revision      string
	groupIndex    int
	leaderUID     types.UID
}

type groupPartition struct {
	dcdGeneration int64
	lwsUID        types.UID
	lwsGeneration int64
	revision      string
	groupIndex    int
}

// MergeOutcome is the pure merge/prune result against one annotation value.
type MergeOutcome struct {
	Encoded           string
	Changed           bool
	CoverageLimited   bool
	Unavailable       bool
	UnavailableReason string
	NextExpiry        *time.Time
	RecordCount       int
}

// MergeFailureRecords validates, prunes, and merges CurrentFailure observations into the ledger.
// dcdUID is the owning object identity. existing is the raw annotation value (may be empty).
// observations must be the immutable Observe batch; Unknown/NoCurrentFailureEvidence add nothing.
//
// Absence is existing == "" && the owned annotation key is not present in annotations.
// A present owned key with an empty value is invalid and is not treated as a fresh ledger.
// Callers that pass a non-empty existing raw with a nil annotations map remain coherent:
// non-empty existing is never treated as absent.
func MergeFailureRecords(
	dcdUID types.UID,
	existing string,
	annotations map[string]string,
	observations []Observation,
	now time.Time,
) MergeOutcome {
	if dcdUID == "" {
		return MergeOutcome{Unavailable: true, UnavailableReason: "missing_dcd_uid"}
	}

	_, ownedKeyPresent := annotations[FailureRecordsAnnotationKey]
	absent := existing == "" && !ownedKeyPresent
	doc, err := decodeFailureRecords(existing, absent)
	if err != nil {
		return MergeOutcome{Unavailable: true, UnavailableReason: err.Error()}
	}
	if reason := validateStoredDocument(doc, dcdUID, now, absent); reason != "" {
		return MergeOutcome{Unavailable: true, UnavailableReason: reason}
	}
	return mergeValidatedFailureRecords(dcdUID, existing, annotations, doc, observations, now)
}

func mergeValidatedFailureRecords(
	dcdUID types.UID,
	existing string,
	annotations map[string]string,
	doc failureRecordsDocument,
	observations []Observation,
	now time.Time,
) MergeOutcome {
	cutoff := now.Add(-failureRecordRetention)
	prunedBefore := time.Time{}
	if doc.PrunedBefore != nil && !doc.PrunedBefore.IsZero() {
		prunedBefore = doc.PrunedBefore.Time
	}

	retained, pruned := pruneExpiredEntries(doc.Entries, cutoff)
	if pruned && cutoff.After(prunedBefore) {
		prunedBefore = cutoff
	}

	// New source FinishedAt must be strictly later than the effective window floor.
	windowFloor := cutoff
	if prunedBefore.After(windowFloor) {
		windowFloor = prunedBefore
	}

	candidates := admitFailureObservations(observations, dcdUID, windowFloor, now)
	sortFailureEntries(candidates)

	merged, admittedNew, coverageLimited := mergeRetainedWithCandidates(retained, candidates)
	if !pruned && !admittedNew {
		return MergeOutcome{
			Encoded:         existing,
			Changed:         false,
			CoverageLimited: coverageLimited,
			NextExpiry:      nextFailureExpiry(retained, now),
			RecordCount:     len(retained),
		}
	}

	return encodeMergedFailureRecords(
		dcdUID,
		existing,
		annotations,
		merged,
		retained,
		prunedBefore,
		pruned,
		admittedNew,
		coverageLimited,
		now,
	)
}

func encodeMergedFailureRecords(
	dcdUID types.UID,
	existing string,
	annotations map[string]string,
	merged []failureRecordEntry,
	retained []failureRecordEntry,
	prunedBefore time.Time,
	pruned bool,
	admittedNew bool,
	coverageLimited bool,
	now time.Time,
) MergeOutcome {
	encoded, err := encodeFailureDocument(dcdUID, prunedBefore, merged)
	if err != nil {
		return MergeOutcome{Unavailable: true, UnavailableReason: "encode_failed"}
	}
	if proposedWriteFits(annotations, encoded) {
		return MergeOutcome{
			Encoded:         encoded,
			Changed:         true,
			CoverageLimited: coverageLimited,
			NextExpiry:      nextFailureExpiry(merged, now),
			RecordCount:     len(merged),
		}
	}

	// Proposed merge is capacity pressure. Prefer a natural prune-only write when possible.
	if pruned && admittedNew {
		pruneEncoded, pruneErr := encodeFailureDocument(dcdUID, prunedBefore, retained)
		if pruneErr == nil && proposedWriteFits(annotations, pruneEncoded) {
			return MergeOutcome{
				Encoded:         pruneEncoded,
				Changed:         true,
				CoverageLimited: true,
				NextExpiry:      nextFailureExpiry(retained, now),
				RecordCount:     len(retained),
			}
		}
	}

	return MergeOutcome{
		Encoded:         existing,
		Changed:         false,
		CoverageLimited: true,
		NextExpiry:      nextFailureExpiry(retained, now),
		RecordCount:     len(retained),
	}
}

func encodeFailureDocument(
	dcdUID types.UID,
	prunedBefore time.Time,
	entries []failureRecordEntry,
) (string, error) {
	sorted := append([]failureRecordEntry(nil), entries...)
	sortFailureEntries(sorted)
	outDoc := failureRecordsDocument{
		Version: failureRecordsVersion,
		DCDUID:  dcdUID,
		Entries: sorted,
	}
	if !prunedBefore.IsZero() {
		t := metav1.NewTime(prunedBefore)
		outDoc.PrunedBefore = &t
	}
	return encodeFailureRecords(outDoc)
}

func proposedWriteFits(annotations map[string]string, encoded string) bool {
	if len(encoded) > failureRecordMaxEncodedBytes {
		return false
	}
	return !annotationBudgetExceeded(annotations, FailureRecordsAnnotationKey, encoded)
}

func admitFailureObservations(
	observations []Observation,
	dcdUID types.UID,
	windowFloor time.Time,
	now time.Time,
) []failureRecordEntry {
	capHint := min(len(observations), failureRecordMaxEntries+1)
	out := make([]failureRecordEntry, 0, capHint)
	seen := map[recordIdentity]struct{}{}
	firstObserved := metav1.NewTime(now)

	for i := range observations {
		obs := observations[i]
		entry, ok := observationToFailureEntry(obs, dcdUID, windowFloor, now, firstObserved)
		if !ok {
			continue
		}
		id := entry.identity()
		if _, dup := seen[id]; dup {
			continue
		}
		seen[id] = struct{}{}
		out = append(out, entry)
		// 33 distinct candidates cannot fit a 32-record union; stop allocating further.
		if len(out) >= failureRecordMaxEntries+1 {
			return out
		}
	}
	return out
}

func observationToFailureEntry(
	obs Observation,
	dcdUID types.UID,
	windowFloor time.Time,
	now time.Time,
	firstObserved metav1.Time,
) (failureRecordEntry, bool) {
	if obs.Verdict != VerdictCurrentFailure {
		return failureRecordEntry{}, false
	}
	if obs.ReadOrigin != ReadOriginManagerCache {
		return failureRecordEntry{}, false
	}
	if obs.Namespace == "" || obs.DCDName == "" {
		return failureRecordEntry{}, false
	}
	if obs.DCDUID != dcdUID || obs.DCDGeneration <= 0 {
		return failureRecordEntry{}, false
	}
	if obs.LWSName == "" || obs.LWSUID == "" || obs.LWSGeneration <= 0 {
		return failureRecordEntry{}, false
	}
	if obs.GroupIndex == nil || *obs.GroupIndex < 0 || obs.Revision == "" || obs.LeaderUID == "" {
		return failureRecordEntry{}, false
	}
	if obs.PodName == "" || obs.PodUID == "" || obs.PodResourceVersion == "" {
		return failureRecordEntry{}, false
	}
	if obs.Container != commonconsts.MainContainerName {
		return failureRecordEntry{}, false
	}
	if obs.ExitCode == nil || *obs.ExitCode == 0 {
		return failureRecordEntry{}, false
	}
	if obs.Ready != corev1.ConditionFalse {
		return failureRecordEntry{}, false
	}
	if obs.FinishedAt == nil || obs.FinishedAt.IsZero() {
		return failureRecordEntry{}, false
	}
	if obs.ReadyLastTransitionTime == nil || obs.ReadyLastTransitionTime.IsZero() {
		return failureRecordEntry{}, false
	}
	if !obs.FinishedAt.Time.After(windowFloor) {
		return failureRecordEntry{}, false
	}
	if obs.FinishedAt.Time.After(now.Add(clockSkewAllowance)) {
		return failureRecordEntry{}, false
	}
	if obs.ReadyLastTransitionTime.Time.After(now.Add(clockSkewAllowance)) {
		return failureRecordEntry{}, false
	}

	return failureRecordEntry{
		DCDGeneration:           obs.DCDGeneration,
		LWSName:                 obs.LWSName,
		LWSUID:                  obs.LWSUID,
		LWSGeneration:           obs.LWSGeneration,
		Revision:                obs.Revision,
		GroupIndex:              *obs.GroupIndex,
		LeaderUID:               obs.LeaderUID,
		PodName:                 obs.PodName,
		PodUID:                  obs.PodUID,
		PodResourceVersion:      obs.PodResourceVersion,
		Container:               obs.Container,
		ExitCode:                *obs.ExitCode,
		FinishedAt:              *obs.FinishedAt,
		ReadyLastTransitionTime: *obs.ReadyLastTransitionTime,
		FirstObservedAt:         firstObserved,
	}, true
}

func mergeRetainedWithCandidates(
	retained, candidates []failureRecordEntry,
) ([]failureRecordEntry, bool, bool) {
	byID := make(map[recordIdentity]failureRecordEntry, len(retained))
	for _, entry := range retained {
		byID[entry.identity()] = entry
	}

	pending := make([]failureRecordEntry, 0, len(candidates))
	for _, candidate := range candidates {
		id := candidate.identity()
		if existing, found := byID[id]; found {
			// Existing record wins byte-for-byte for witness and firstObservedAt.
			_ = existing
			continue
		}
		pending = append(pending, candidate)
	}
	if len(pending) == 0 {
		return retained, false, false
	}

	trial := append([]failureRecordEntry(nil), retained...)
	trial = append(trial, pending...)
	if !withinFailureCapacity(trial) {
		return retained, false, true
	}
	return trial, true, false
}

func withinFailureCapacity(entries []failureRecordEntry) bool {
	if len(entries) > failureRecordMaxEntries {
		return false
	}
	groups := map[groupPartition]int{}
	for _, entry := range entries {
		groups[entry.partition()]++
		if groups[entry.partition()] > failureRecordMaxPerGroup {
			return false
		}
	}
	return len(groups) <= failureRecordMaxGroups
}

func pruneExpiredEntries(entries []failureRecordEntry, cutoff time.Time) ([]failureRecordEntry, bool) {
	retained := make([]failureRecordEntry, 0, len(entries))
	pruned := false
	for _, entry := range entries {
		if !entry.FinishedAt.Time.After(cutoff) {
			pruned = true
			continue
		}
		retained = append(retained, entry)
	}
	return retained, pruned
}

func nextFailureExpiry(entries []failureRecordEntry, now time.Time) *time.Time {
	var earliest time.Time
	found := false
	for _, entry := range entries {
		exp := entry.FinishedAt.Time.Add(failureRecordRetention)
		if !exp.After(now) {
			continue
		}
		if !found || exp.Before(earliest) {
			earliest = exp
			found = true
		}
	}
	if !found {
		return nil
	}
	return &earliest
}

func (e failureRecordEntry) identity() recordIdentity {
	return recordIdentity{
		dcdGeneration: e.DCDGeneration,
		lwsUID:        e.LWSUID,
		lwsGeneration: e.LWSGeneration,
		revision:      e.Revision,
		groupIndex:    e.GroupIndex,
		leaderUID:     e.LeaderUID,
	}
}

func (e failureRecordEntry) partition() groupPartition {
	return groupPartition{
		dcdGeneration: e.DCDGeneration,
		lwsUID:        e.LWSUID,
		lwsGeneration: e.LWSGeneration,
		revision:      e.Revision,
		groupIndex:    e.GroupIndex,
	}
}

func sortFailureEntries(entries []failureRecordEntry) {
	sort.SliceStable(entries, func(i, j int) bool {
		return failureEntryLess(entries[i], entries[j])
	})
}

func failureEntryLess(a, b failureRecordEntry) bool {
	if a.FinishedAt.Time.Before(b.FinishedAt.Time) {
		return true
	}
	if b.FinishedAt.Time.Before(a.FinishedAt.Time) {
		return false
	}
	if a.DCDGeneration != b.DCDGeneration {
		return a.DCDGeneration < b.DCDGeneration
	}
	if a.LWSUID != b.LWSUID {
		return a.LWSUID < b.LWSUID
	}
	if a.LWSGeneration != b.LWSGeneration {
		return a.LWSGeneration < b.LWSGeneration
	}
	if a.Revision != b.Revision {
		return a.Revision < b.Revision
	}
	if a.GroupIndex != b.GroupIndex {
		return a.GroupIndex < b.GroupIndex
	}
	if a.LeaderUID != b.LeaderUID {
		return a.LeaderUID < b.LeaderUID
	}
	return a.PodUID < b.PodUID
}

func annotationBudgetExceeded(annotations map[string]string, key, newValue string) bool {
	total := len(key) + len(newValue)
	for existingKey, existingValue := range annotations {
		if existingKey == key {
			continue
		}
		total += len(existingKey) + len(existingValue)
	}
	return total > failureRecordAnnotationBudget
}
