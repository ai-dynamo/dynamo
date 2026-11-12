/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lws

import (
	"encoding/json"
	"fmt"
	"io"
	"strings"
	"time"

	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"k8s.io/apimachinery/pkg/types"
)

func decodeFailureRecords(raw string, absent bool) (failureRecordsDocument, error) {
	if absent {
		return failureRecordsDocument{
			Version: failureRecordsVersion,
			Entries: nil,
		}, nil
	}
	if raw == "" {
		// Present owned key with empty value is invalid; only a missing key may initialize.
		return failureRecordsDocument{}, fmt.Errorf("empty_annotation")
	}
	if len(raw) > failureRecordMaxEncodedBytes {
		return failureRecordsDocument{}, fmt.Errorf("existing_oversized")
	}

	var doc failureRecordsDocument
	dec := json.NewDecoder(strings.NewReader(raw))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&doc); err != nil {
		return failureRecordsDocument{}, fmt.Errorf("malformed_json")
	}
	// More() is not an EOF check; require a second Decode to hit io.EOF exactly.
	var extra any
	if err := dec.Decode(&extra); err != io.EOF {
		return failureRecordsDocument{}, fmt.Errorf("trailing_garbage")
	}
	return doc, nil
}

func encodeFailureRecords(doc failureRecordsDocument) (string, error) {
	if doc.Entries == nil {
		doc.Entries = []failureRecordEntry{}
	}
	raw, err := json.Marshal(doc)
	if err != nil {
		return "", err
	}
	return string(raw), nil
}

func validateStoredDocument(doc failureRecordsDocument, dcdUID types.UID, now time.Time, absent bool) string {
	// Only a genuinely missing annotation may initialize an empty root.
	if absent {
		return ""
	}
	if doc.Version != failureRecordsVersion {
		return "unknown_version"
	}
	if doc.DCDUID == "" {
		return "missing_stored_dcd_uid"
	}
	if doc.DCDUID != dcdUID {
		return "dcd_uid_mismatch"
	}
	if doc.PrunedBefore != nil && doc.PrunedBefore.Time.After(now.Add(clockSkewAllowance)) {
		return "pruned_before_in_future"
	}

	prunedBefore := time.Time{}
	if doc.PrunedBefore != nil && !doc.PrunedBefore.IsZero() {
		prunedBefore = doc.PrunedBefore.Time
	}

	seen := map[recordIdentity]struct{}{}
	for i := range doc.Entries {
		entry := doc.Entries[i]
		if reason := validateStoredEntry(entry, now, prunedBefore); reason != "" {
			return reason
		}
		id := entry.identity()
		if _, dup := seen[id]; dup {
			return "duplicate_stored_identity"
		}
		seen[id] = struct{}{}
	}
	if !withinFailureCapacity(doc.Entries) {
		return "existing_capacity_invalid"
	}
	return ""
}

func validateStoredEntry(entry failureRecordEntry, now time.Time, prunedBefore time.Time) string {
	if entry.DCDGeneration <= 0 || entry.LWSName == "" || entry.LWSUID == "" || entry.LWSGeneration <= 0 {
		return "invalid_stored_identity"
	}
	if entry.Revision == "" || entry.GroupIndex < 0 || entry.LeaderUID == "" {
		return "invalid_stored_identity"
	}
	if entry.PodName == "" || entry.PodUID == "" || entry.PodResourceVersion == "" {
		return "invalid_stored_witness"
	}
	if entry.Container != commonconsts.MainContainerName {
		return "invalid_stored_container"
	}
	if entry.ExitCode == 0 {
		return "invalid_stored_exit"
	}
	if entry.FinishedAt.IsZero() || entry.ReadyLastTransitionTime.IsZero() || entry.FirstObservedAt.IsZero() {
		return "invalid_stored_time"
	}
	if entry.FinishedAt.Time.After(now.Add(clockSkewAllowance)) {
		return "stored_finished_at_in_future"
	}
	if entry.ReadyLastTransitionTime.Time.After(now.Add(clockSkewAllowance)) {
		return "stored_ready_ltt_in_future"
	}
	if entry.FirstObservedAt.Time.After(now.Add(clockSkewAllowance)) {
		return "stored_first_observed_in_future"
	}
	if !prunedBefore.IsZero() && !entry.FinishedAt.Time.After(prunedBefore) {
		return "finished_at_not_after_pruned_before"
	}
	if entry.FirstObservedAt.Time.Before(entry.FinishedAt.Time.Add(-clockSkewAllowance)) {
		return "first_observed_before_finished_at"
	}
	if entry.FirstObservedAt.Time.Before(entry.ReadyLastTransitionTime.Time.Add(-clockSkewAllowance)) {
		return "first_observed_before_ready_ltt"
	}
	return ""
}
