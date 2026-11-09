/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lws

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/interceptor"
	leaderworkersetv1 "sigs.k8s.io/lws/api/leaderworkerset/v1"
)

func TestMergeFailureRecordsAdmitsAndDedupes(t *testing.T) {
	f := newFixture()
	obs := currentFailureObservation(f, f.leaderPod, 23)

	t.Log("Admit the first CurrentFailure into an empty ledger")
	first := MergeFailureRecords(f.dcd.UID, "", nil, []Observation{obs}, f.now)
	require.False(t, first.Unavailable)
	require.True(t, first.Changed)
	require.Equal(t, 1, first.RecordCount)

	t.Log("Repeat the same incarnation and keep the original firstObservedAt")
	later := f.now.Add(time.Minute)
	second := MergeFailureRecords(f.dcd.UID, first.Encoded, map[string]string{
		FailureRecordsAnnotationKey: first.Encoded,
	}, []Observation{obs}, later)
	require.False(t, second.Unavailable)
	assert.False(t, second.Changed)
	assert.Equal(t, first.Encoded, second.Encoded)

	var doc failureRecordsDocument
	require.NoError(t, json.Unmarshal([]byte(first.Encoded), &doc))
	require.Len(t, doc.Entries, 1)
	assert.Equal(t, f.now.UTC(), doc.Entries[0].FirstObservedAt.UTC())
	assert.Equal(t, f.leaderPod.UID, doc.Entries[0].LeaderUID)
}

func TestMergeFailureRecordsIsolatesGroupsAndWorkers(t *testing.T) {
	f := newFixture()
	leader := currentFailureObservation(f, f.leaderPod, 23)
	follower := currentFailureObservation(f, f.workerPod, 137)

	t.Log("Leader and follower of one incarnation collapse to one durable record")
	same := MergeFailureRecords(f.dcd.UID, "", nil, []Observation{leader, follower}, f.now)
	require.True(t, same.Changed)
	assert.Equal(t, 1, same.RecordCount)

	t.Log("A distinct group keeps a separate record")
	otherGroup := 1
	other := leader
	other.GroupIndex = &otherGroup
	other.LeaderUID = "leader-pod-uid-1"
	other.PodName = "decode-1"
	other.PodUID = "leader-pod-uid-1"
	distinct := MergeFailureRecords(f.dcd.UID, same.Encoded, map[string]string{
		FailureRecordsAnnotationKey: same.Encoded,
	}, []Observation{other}, f.now)
	require.True(t, distinct.Changed)
	assert.Equal(t, 2, distinct.RecordCount)
}

func TestMergeFailureRecordsIdentityIsolationTable(t *testing.T) {
	cases := []struct {
		name string
		mut  func(obs *Observation)
	}{
		{name: "DCDGeneration", mut: func(obs *Observation) { obs.DCDGeneration = obs.DCDGeneration + 1 }},
		{name: "LWSUID", mut: func(obs *Observation) { obs.LWSUID = "lws-uid-isolated" }},
		{name: "LWSGeneration", mut: func(obs *Observation) { obs.LWSGeneration = obs.LWSGeneration + 1 }},
		{name: "Revision", mut: func(obs *Observation) { obs.Revision = "rev-isolated" }},
		{name: "GroupIndex", mut: func(obs *Observation) {
			group := 1
			obs.GroupIndex = &group
		}},
		{name: "LeaderUID", mut: func(obs *Observation) { obs.LeaderUID = "leader-uid-isolated" }},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Log("Seed one CurrentFailure, then change only " + tc.name)
			f := newFixture()
			seed := currentFailureObservation(f, f.leaderPod, 23)
			first := MergeFailureRecords(f.dcd.UID, "", nil, []Observation{seed}, f.now)
			require.False(t, first.Unavailable)
			require.True(t, first.Changed)
			require.Equal(t, 1, first.RecordCount)

			var seeded failureRecordsDocument
			require.NoError(t, json.Unmarshal([]byte(first.Encoded), &seeded))
			require.Len(t, seeded.Entries, 1)
			old := seeded.Entries[0]

			second := seed
			tc.mut(&second)
			merged := MergeFailureRecords(f.dcd.UID, first.Encoded, map[string]string{
				FailureRecordsAnnotationKey: first.Encoded,
			}, []Observation{second}, f.now)
			require.False(t, merged.Unavailable)
			require.True(t, merged.Changed)
			require.Equal(t, 2, merged.RecordCount)

			var doc failureRecordsDocument
			require.NoError(t, json.Unmarshal([]byte(merged.Encoded), &doc))
			require.Len(t, doc.Entries, 2)
			var preserved *failureRecordEntry
			for i := range doc.Entries {
				entry := &doc.Entries[i]
				if entry.identity() == old.identity() {
					preserved = entry
					break
				}
			}
			require.NotNil(t, preserved, "original identity must remain after isolation merge")
			assert.Equal(t, old.PodUID, preserved.PodUID)
			assert.Equal(t, old.PodName, preserved.PodName)
			assert.Equal(t, old.PodResourceVersion, preserved.PodResourceVersion)
			assert.Equal(t, old.FirstObservedAt.UTC(), preserved.FirstObservedAt.UTC())

			t.Log("Replay the new identity later without rewriting the ledger")
			later := MergeFailureRecords(f.dcd.UID, merged.Encoded, map[string]string{
				FailureRecordsAnnotationKey: merged.Encoded,
			}, []Observation{second}, f.now.Add(time.Minute))
			assert.False(t, later.Changed)
			assert.Equal(t, merged.Encoded, later.Encoded)
			assert.Equal(t, 2, later.RecordCount)
		})
	}
}

func TestMergeFailureRecordsIndependentCapacityTable(t *testing.T) {
	t.Run("16 partitions then seventeenth declined", func(t *testing.T) {
		t.Log("Admit one record in each of 16 partitions through public Merge")
		f := newFixture()
		batch := make([]Observation, 0, failureRecordMaxGroups)
		for i := 0; i < failureRecordMaxGroups; i++ {
			obs := currentFailureObservation(f, f.leaderPod, 23)
			group := i
			obs.GroupIndex = &group
			obs.LeaderUID = types.UID(fmt.Sprintf("partition-leader-%d", i))
			obs.PodUID = types.UID(fmt.Sprintf("partition-pod-%d", i))
			obs.PodName = fmt.Sprintf("decode-partition-%d", i)
			batch = append(batch, obs)
		}
		first := MergeFailureRecords(f.dcd.UID, "", nil, batch, f.now)
		require.False(t, first.Unavailable)
		require.True(t, first.Changed)
		require.Equal(t, failureRecordMaxGroups, first.RecordCount)
		require.NotNil(t, first.NextExpiry)
		wantExpiry := *first.NextExpiry

		t.Log("A seventeenth partition under the total-entry limit is declined without eviction")
		extra := currentFailureObservation(f, f.leaderPod, 23)
		group := failureRecordMaxGroups
		extra.GroupIndex = &group
		extra.LeaderUID = "partition-leader-extra"
		extra.PodUID = "partition-pod-extra"
		extra.PodName = "decode-partition-extra"
		out := MergeFailureRecords(f.dcd.UID, first.Encoded, map[string]string{
			FailureRecordsAnnotationKey: first.Encoded,
		}, []Observation{extra}, f.now)
		assert.True(t, out.CoverageLimited)
		assert.False(t, out.Unavailable)
		assert.False(t, out.Changed)
		assert.Equal(t, first.Encoded, out.Encoded)
		assert.Equal(t, failureRecordMaxGroups, out.RecordCount)
		require.NotNil(t, out.NextExpiry)
		assert.Equal(t, wantExpiry, *out.NextExpiry)
	})

	t.Run("8 leaders in one partition then ninth declined", func(t *testing.T) {
		t.Log("Admit eight distinct leaders in one partition through public Merge")
		f := newFixture()
		batch := make([]Observation, 0, failureRecordMaxPerGroup)
		for i := 0; i < failureRecordMaxPerGroup; i++ {
			obs := currentFailureObservation(f, f.leaderPod, 23)
			obs.LeaderUID = types.UID(fmt.Sprintf("per-group-leader-%d", i))
			obs.PodUID = types.UID(fmt.Sprintf("per-group-pod-%d", i))
			obs.PodName = fmt.Sprintf("decode-per-group-%d", i)
			batch = append(batch, obs)
		}
		first := MergeFailureRecords(f.dcd.UID, "", nil, batch, f.now)
		require.False(t, first.Unavailable)
		require.True(t, first.Changed)
		require.Equal(t, failureRecordMaxPerGroup, first.RecordCount)
		require.NotNil(t, first.NextExpiry)
		wantExpiry := *first.NextExpiry

		t.Log("A ninth leader in the same partition under the total-entry limit is declined")
		extra := currentFailureObservation(f, f.leaderPod, 23)
		extra.LeaderUID = "per-group-leader-extra"
		extra.PodUID = "per-group-pod-extra"
		extra.PodName = "decode-per-group-extra"
		out := MergeFailureRecords(f.dcd.UID, first.Encoded, map[string]string{
			FailureRecordsAnnotationKey: first.Encoded,
		}, []Observation{extra}, f.now)
		assert.True(t, out.CoverageLimited)
		assert.False(t, out.Unavailable)
		assert.False(t, out.Changed)
		assert.Equal(t, first.Encoded, out.Encoded)
		assert.Equal(t, failureRecordMaxPerGroup, out.RecordCount)
		require.NotNil(t, out.NextExpiry)
		assert.Equal(t, wantExpiry, *out.NextExpiry)
	})
}

func TestMergeFailureRecordsTTLAndRewindGuard(t *testing.T) {
	f := newFixture()
	obs := currentFailureObservation(f, f.leaderPod, 23)

	t.Log("Store a record then advance past FinishedAt retention")
	first := MergeFailureRecords(f.dcd.UID, "", nil, []Observation{obs}, f.now)
	require.True(t, first.Changed)

	expiredNow := f.finished.Add(failureRecordRetention + time.Minute)
	pruned := MergeFailureRecords(f.dcd.UID, first.Encoded, map[string]string{
		FailureRecordsAnnotationKey: first.Encoded,
	}, nil, expiredNow)
	require.True(t, pruned.Changed)
	assert.Equal(t, 0, pruned.RecordCount)

	var doc failureRecordsDocument
	require.NoError(t, json.Unmarshal([]byte(pruned.Encoded), &doc))
	require.NotNil(t, doc.PrunedBefore)

	t.Log("Reject reinsertion of the same old FinishedAt after prune")
	replay := MergeFailureRecords(f.dcd.UID, pruned.Encoded, map[string]string{
		FailureRecordsAnnotationKey: pruned.Encoded,
	}, []Observation{obs}, expiredNow)
	assert.False(t, replay.Changed)
	assert.Equal(t, 0, replay.RecordCount)
	assert.Equal(t, pruned.Encoded, replay.Encoded)

	t.Log("Large clock rewind past the watermark future allowance is unavailable")
	rewound := MergeFailureRecords(f.dcd.UID, pruned.Encoded, map[string]string{
		FailureRecordsAnnotationKey: pruned.Encoded,
	}, []Observation{obs}, f.now)
	assert.True(t, rewound.Unavailable)
	assert.Equal(t, "pruned_before_in_future", rewound.UnavailableReason)
	assert.False(t, rewound.Changed)

	dcd := f.dcd.DeepCopy()
	dcd.ResourceVersion = "17"
	dcd.Generation = 2
	dcd.Annotations = map[string]string{FailureRecordsAnnotationKey: pruned.Encoded}
	patchCalls := 0
	writer := interceptor.NewClient(f.reader(dcd), interceptor.Funcs{
		Patch: func(ctx context.Context, c client.WithWatch, obj client.Object, patch client.Patch, opts ...client.PatchOption) error {
			patchCalls++
			return c.Patch(ctx, obj, patch, opts...)
		},
	})
	persist, err := PersistFailureRecords(t.Context(), writer, dcd.DeepCopy(), []Observation{obs}, f.now)
	require.NoError(t, err)
	assert.True(t, persist.Unavailable)
	assert.False(t, persist.Written)
	assert.Equal(t, 0, patchCalls)
	stored := &nvidiacomv1beta1.DynamoComponentDeployment{}
	require.NoError(t, writer.Get(t.Context(), client.ObjectKeyFromObject(dcd), stored))
	assert.Equal(t, pruned.Encoded, stored.Annotations[FailureRecordsAnnotationKey])

	t.Log("Within the 5s watermark skew, the watermark still blocks the old FinishedAt without rewrite")
	require.NotNil(t, doc.PrunedBefore)
	skewNow := doc.PrunedBefore.Time
	within := MergeFailureRecords(f.dcd.UID, pruned.Encoded, map[string]string{
		FailureRecordsAnnotationKey: pruned.Encoded,
	}, []Observation{obs}, skewNow)
	assert.False(t, within.Unavailable)
	assert.False(t, within.Changed)
	assert.Equal(t, pruned.Encoded, within.Encoded)
	assert.Equal(t, 0, within.RecordCount)
}

func TestMergeFailureRecordsSourceTTLExactBoundary(t *testing.T) {
	f := newFixture()
	obs := currentFailureObservation(f, f.leaderPod, 23)
	first := MergeFailureRecords(f.dcd.UID, "", nil, []Observation{obs}, f.now)
	require.True(t, first.Changed)

	t.Log("FinishedAt exactly at the retention floor is pruned")
	exactCutoff := f.finished.Add(failureRecordRetention)
	pruned := MergeFailureRecords(f.dcd.UID, first.Encoded, map[string]string{
		FailureRecordsAnnotationKey: first.Encoded,
	}, nil, exactCutoff)
	require.True(t, pruned.Changed)
	assert.Equal(t, 0, pruned.RecordCount)

	t.Log("A new observation finishing exactly at the watermark is refused")
	refused := MergeFailureRecords(f.dcd.UID, pruned.Encoded, map[string]string{
		FailureRecordsAnnotationKey: pruned.Encoded,
	}, []Observation{obs}, exactCutoff)
	assert.False(t, refused.Changed)
	assert.Equal(t, pruned.Encoded, refused.Encoded)
}

func TestMergeFailureRecordsPreservesCorruptAndUnknownSchema(t *testing.T) {
	f := newFixture()
	obs := currentFailureObservation(f, f.leaderPod, 23)

	cases := []struct {
		name   string
		raw    string
		reason string
	}{
		{name: "malformed JSON", raw: "{not-json", reason: "malformed_json"},
		{name: "trailing garbage", raw: `{"version":1,"dcdUID":"dcd-uid","entries":[]} trailing`, reason: "trailing_garbage"},
		{name: "second JSON value", raw: `{"version":1,"dcdUID":"dcd-uid","entries":[]}{"version":1}`, reason: "trailing_garbage"},
		{name: "trailing closing brace", raw: `{"version":1,"dcdUID":"dcd-uid","entries":[]}}`, reason: "trailing_garbage"},
		{name: "trailing closing bracket", raw: `{"version":1,"dcdUID":"dcd-uid","entries":[]}]`, reason: "trailing_garbage"},
		{name: "present missing dcdUID", raw: `{"version":1,"entries":[]}`, reason: "missing_stored_dcd_uid"},
		{name: "present null dcdUID", raw: `{"version":1,"dcdUID":null,"entries":[]}`, reason: "missing_stored_dcd_uid"},
		{name: "unknown version", raw: `{"version":99,"dcdUID":"dcd-uid","entries":[]}`, reason: "unknown_version"},
		{name: "wrong DCD UID", raw: `{"version":1,"dcdUID":"other","entries":[]}`, reason: "dcd_uid_mismatch"},
		{name: "unknown field", raw: `{"version":1,"dcdUID":"dcd-uid","entries":[],"extra":1}`, reason: "malformed_json"},
		{name: "existing oversized", raw: `{"version":1,"dcdUID":"dcd-uid","entries":[],"pad":"` + strings.Repeat("x", failureRecordMaxEncodedBytes) + `"}`, reason: "existing_oversized"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			out := MergeFailureRecords(f.dcd.UID, tc.raw, nil, []Observation{obs}, f.now)
			assert.True(t, out.Unavailable)
			assert.False(t, out.Changed)
			assert.Equal(t, "", out.Encoded)
			assert.Equal(t, tc.reason, out.UnavailableReason)
		})
	}

	t.Run("whitespace after valid document remains accepted", func(t *testing.T) {
		raw := `{"version":1,"dcdUID":"dcd-uid","entries":[]}  ` + "\n\t"
		out := MergeFailureRecords(f.dcd.UID, raw, map[string]string{FailureRecordsAnnotationKey: raw}, nil, f.now)
		assert.False(t, out.Unavailable)
		assert.False(t, out.Changed)
		assert.Equal(t, raw, out.Encoded)
	})
}

func TestMergeFailureRecordsRejectsInconsistentStoredTimes(t *testing.T) {
	f := newFixture()
	obs := currentFailureObservation(f, f.leaderPod, 23)
	base := failureRecordEntry{
		DCDGeneration:           2,
		LWSName:                 "decode",
		LWSUID:                  "lws-uid",
		LWSGeneration:           2,
		Revision:                testRevision,
		GroupIndex:              0,
		LeaderUID:               "leader-0",
		PodName:                 "pod",
		PodUID:                  "pod-0",
		PodResourceVersion:      "1",
		Container:               commonconsts.MainContainerName,
		ExitCode:                23,
		FinishedAt:              f.finished,
		ReadyLastTransitionTime: f.readyLTT,
		FirstObservedAt:         metav1.NewTime(f.now),
	}

	t.Run("finished_at not after prunedBefore", func(t *testing.T) {
		doc := failureRecordsDocument{
			Version:      1,
			DCDUID:       f.dcd.UID,
			PrunedBefore: &f.finished,
			Entries:      []failureRecordEntry{base},
		}
		raw, err := encodeFailureRecords(doc)
		require.NoError(t, err)
		out := MergeFailureRecords(f.dcd.UID, raw, nil, []Observation{obs}, f.now)
		assert.True(t, out.Unavailable)
		assert.Equal(t, "finished_at_not_after_pruned_before", out.UnavailableReason)
	})

	t.Run("firstObserved before FinishedAt beyond skew", func(t *testing.T) {
		entry := base
		entry.FirstObservedAt = metav1.NewTime(f.finished.Add(-time.Minute))
		doc := failureRecordsDocument{Version: 1, DCDUID: f.dcd.UID, Entries: []failureRecordEntry{entry}}
		raw, err := encodeFailureRecords(doc)
		require.NoError(t, err)
		out := MergeFailureRecords(f.dcd.UID, raw, nil, []Observation{obs}, f.now)
		assert.True(t, out.Unavailable)
		assert.Equal(t, "first_observed_before_finished_at", out.UnavailableReason)
	})

	t.Run("future FinishedAt", func(t *testing.T) {
		entry := base
		entry.FinishedAt = metav1.NewTime(f.now.Add(time.Minute))
		entry.FirstObservedAt = metav1.NewTime(f.now.Add(time.Minute))
		doc := failureRecordsDocument{Version: 1, DCDUID: f.dcd.UID, Entries: []failureRecordEntry{entry}}
		raw, err := encodeFailureRecords(doc)
		require.NoError(t, err)
		out := MergeFailureRecords(f.dcd.UID, raw, nil, []Observation{obs}, f.now)
		assert.True(t, out.Unavailable)
		assert.Equal(t, "stored_finished_at_in_future", out.UnavailableReason)
	})
}

func TestMergeFailureRecordsCapacityDoesNotEvict(t *testing.T) {
	f := newFixture()
	existing := make([]failureRecordEntry, 0, failureRecordMaxEntries)
	for i := 0; i < failureRecordMaxEntries; i++ {
		existing = append(existing, failureRecordEntry{
			DCDGeneration:           2,
			LWSName:                 "decode",
			LWSUID:                  "lws-uid",
			LWSGeneration:           2,
			Revision:                testRevision,
			GroupIndex:              i % failureRecordMaxGroups,
			LeaderUID:               types.UID(fmt.Sprintf("leader-%d", i)),
			PodName:                 "pod",
			PodUID:                  types.UID(fmt.Sprintf("pod-%d", i)),
			PodResourceVersion:      "1",
			Container:               commonconsts.MainContainerName,
			ExitCode:                23,
			FinishedAt:              f.finished,
			ReadyLastTransitionTime: f.readyLTT,
			FirstObservedAt:         metav1.NewTime(f.now),
		})
	}
	doc := failureRecordsDocument{Version: 1, DCDUID: f.dcd.UID, Entries: existing}
	raw, err := encodeFailureRecords(doc)
	require.NoError(t, err)

	t.Log("Decline a new identity when the ledger is full and keep unsorted raw bytes")
	extra := currentFailureObservation(f, f.leaderPod, 23)
	extra.LeaderUID = "brand-new-leader"
	extra.PodUID = "brand-new-pod"
	out := MergeFailureRecords(f.dcd.UID, raw, map[string]string{FailureRecordsAnnotationKey: raw}, []Observation{extra}, f.now)
	assert.True(t, out.CoverageLimited)
	assert.False(t, out.Changed)
	assert.Equal(t, raw, out.Encoded)
	assert.Equal(t, failureRecordMaxEntries, out.RecordCount)
	require.NotNil(t, out.NextExpiry)
	assert.Equal(t, f.finished.Add(failureRecordRetention), out.NextExpiry.UTC())
}

func TestMergeFailureRecordsLargeDistinctBatchCoverageLimited(t *testing.T) {
	f := newFixture()
	existing := make([]failureRecordEntry, 0, failureRecordMaxEntries)
	for i := 0; i < failureRecordMaxEntries; i++ {
		existing = append(existing, failureRecordEntry{
			DCDGeneration:           2,
			LWSName:                 "decode",
			LWSUID:                  "lws-uid",
			LWSGeneration:           2,
			Revision:                testRevision,
			GroupIndex:              i % failureRecordMaxGroups,
			LeaderUID:               types.UID(fmt.Sprintf("kept-leader-%d", i)),
			PodName:                 "pod",
			PodUID:                  types.UID(fmt.Sprintf("kept-pod-%d", i)),
			PodResourceVersion:      "1",
			Container:               commonconsts.MainContainerName,
			ExitCode:                23,
			FinishedAt:              f.finished,
			ReadyLastTransitionTime: f.readyLTT,
			FirstObservedAt:         metav1.NewTime(f.now),
		})
	}
	doc := failureRecordsDocument{Version: 1, DCDUID: f.dcd.UID, Entries: existing}
	raw, err := encodeFailureRecords(doc)
	require.NoError(t, err)

	t.Log("A large distinct CurrentFailure batch declines without eviction or rewrite")
	batch := make([]Observation, 0, failureRecordMaxEntries+8)
	for i := 0; i < failureRecordMaxEntries+8; i++ {
		obs := currentFailureObservation(f, f.leaderPod, 23)
		obs.LeaderUID = types.UID(fmt.Sprintf("batch-leader-%d", i))
		obs.PodUID = types.UID(fmt.Sprintf("batch-pod-%d", i))
		obs.PodName = fmt.Sprintf("decode-batch-%d", i)
		batch = append(batch, obs)
	}
	// Many duplicates of one identity still collapse to a single admission attempt.
	dup := currentFailureObservation(f, f.leaderPod, 23)
	dup.LeaderUID = "dup-leader"
	dup.PodUID = "dup-pod"
	dup.PodName = "decode-dup"
	for i := 0; i < 16; i++ {
		batch = append(batch, dup)
	}

	out := MergeFailureRecords(f.dcd.UID, raw, map[string]string{FailureRecordsAnnotationKey: raw}, batch, f.now)
	assert.True(t, out.CoverageLimited)
	assert.False(t, out.Changed)
	assert.Equal(t, raw, out.Encoded)
	assert.Equal(t, failureRecordMaxEntries, out.RecordCount)
	require.NotNil(t, out.NextExpiry)
	assert.Equal(t, f.finished.Add(failureRecordRetention), out.NextExpiry.UTC())

	t.Log("Duplicate-only observations against empty ledger admit one identity")
	one := MergeFailureRecords(f.dcd.UID, "", nil, []Observation{dup, dup, dup}, f.now)
	require.True(t, one.Changed)
	assert.Equal(t, 1, one.RecordCount)
}

func TestMergeFailureRecordsProposedOversizedIsCoverageLimited(t *testing.T) {
	f := newFixture()
	obs := currentFailureObservation(f, f.leaderPod, 23)
	first := MergeFailureRecords(f.dcd.UID, "", nil, []Observation{obs}, f.now)
	require.True(t, first.Changed)

	t.Log("A proposed admission that would exceed the encoded budget declines without disabling expiry")
	huge := currentFailureObservation(f, f.leaderPod, 23)
	huge.LeaderUID = "huge-leader"
	huge.PodUID = "huge-pod"
	huge.PodName = strings.Repeat("p", failureRecordMaxEncodedBytes)
	out := MergeFailureRecords(f.dcd.UID, first.Encoded, map[string]string{
		FailureRecordsAnnotationKey: first.Encoded,
	}, []Observation{huge}, f.now)
	assert.True(t, out.CoverageLimited)
	assert.False(t, out.Changed)
	assert.Equal(t, first.Encoded, out.Encoded)
	require.NotNil(t, out.NextExpiry)
	assert.Equal(t, f.finished.Add(failureRecordRetention), out.NextExpiry.UTC())
}

func TestMergeFailureRecordsAnnotationBudgetUsesByteLengths(t *testing.T) {
	f := newFixture()
	obs := currentFailureObservation(f, f.leaderPod, 23)
	first := MergeFailureRecords(f.dcd.UID, "", nil, []Observation{obs}, f.now)
	require.True(t, first.Changed)

	foreignKey := "example.com/foreign"
	// Multibyte runes: rune count stays under budget while byte length crosses it.
	multibyte := strings.Repeat("世", (failureRecordAnnotationBudget/3)+1024)
	anns := map[string]string{
		FailureRecordsAnnotationKey: first.Encoded,
		foreignKey:                  multibyte,
	}
	require.Greater(t, len(foreignKey)+len(multibyte), failureRecordAnnotationBudget)
	require.Less(t, len(foreignKey)+len([]rune(multibyte)), failureRecordAnnotationBudget)

	extra := currentFailureObservation(f, f.leaderPod, 42)
	extra.LeaderUID = "budget-leader"
	extra.PodUID = "budget-pod"
	out := MergeFailureRecords(f.dcd.UID, first.Encoded, anns, []Observation{extra}, f.now)
	assert.True(t, out.CoverageLimited)
	assert.False(t, out.Changed)
	assert.Equal(t, first.Encoded, out.Encoded)
}

func TestMergeFailureRecordsMainContainerOnly(t *testing.T) {
	f := newFixture()
	sidecar := currentFailureObservation(f, f.leaderPod, 23)
	sidecar.Container = "sidecar"
	out := MergeFailureRecords(f.dcd.UID, "", nil, []Observation{sidecar}, f.now)
	assert.False(t, out.Changed)
	assert.Equal(t, "", out.Encoded)
	assert.Equal(t, 0, out.RecordCount)
}

func TestMergeFailureRecordsZeroPatchAbsentAndNoncanonical(t *testing.T) {
	f := newFixture()

	t.Log("Absent annotation with no CurrentFailure evidence is a true zero-Patch no-op")
	unknown := currentFailureObservation(f, f.leaderPod, 23)
	unknown.Verdict = VerdictUnknown
	healthy := currentFailureObservation(f, f.leaderPod, 23)
	healthy.Verdict = VerdictNoCurrentFailureEvidence
	healthy.ExitCode = nil
	absent := MergeFailureRecords(f.dcd.UID, "", nil, []Observation{unknown, healthy}, f.now)
	assert.False(t, absent.Changed)
	assert.Equal(t, "", absent.Encoded)
	assert.Nil(t, absent.NextExpiry)

	t.Log("Present owned key with empty value is unavailable and does not initialize")
	presentEmpty := MergeFailureRecords(f.dcd.UID, "", map[string]string{
		FailureRecordsAnnotationKey: "",
	}, []Observation{unknown}, f.now)
	assert.True(t, presentEmpty.Unavailable)
	assert.Equal(t, "empty_annotation", presentEmpty.UnavailableReason)
	assert.False(t, presentEmpty.Changed)

	t.Log("Existing valid payload with no new admission preserves exact raw bytes")
	obs := currentFailureObservation(f, f.leaderPod, 23)
	first := MergeFailureRecords(f.dcd.UID, "", nil, []Observation{obs}, f.now)
	require.True(t, first.Changed)
	preserved := MergeFailureRecords(f.dcd.UID, first.Encoded, map[string]string{
		FailureRecordsAnnotationKey: first.Encoded,
	}, []Observation{unknown}, f.now)
	assert.False(t, preserved.Changed)
	assert.Equal(t, first.Encoded, preserved.Encoded)

	t.Log("Persist present-empty annotation is zero-Patch and leaves bytes unchanged")
	emptyDCD := f.dcd.DeepCopy()
	emptyDCD.ResourceVersion = "8"
	emptyDCD.Generation = 2
	emptyDCD.Annotations = map[string]string{FailureRecordsAnnotationKey: ""}
	emptyPatches := 0
	emptyWriter := interceptor.NewClient(f.reader(emptyDCD), interceptor.Funcs{
		Patch: func(ctx context.Context, c client.WithWatch, obj client.Object, patch client.Patch, opts ...client.PatchOption) error {
			emptyPatches++
			return c.Patch(ctx, obj, patch, opts...)
		},
	})
	emptyResult, err := PersistFailureRecords(t.Context(), emptyWriter, emptyDCD.DeepCopy(), []Observation{obs}, f.now)
	require.NoError(t, err)
	assert.True(t, emptyResult.Unavailable)
	assert.False(t, emptyResult.Written)
	assert.Equal(t, 0, emptyPatches)
	emptyStored := &nvidiacomv1beta1.DynamoComponentDeployment{}
	require.NoError(t, emptyWriter.Get(t.Context(), client.ObjectKeyFromObject(emptyDCD), emptyStored))
	assert.Equal(t, "", emptyStored.Annotations[FailureRecordsAnnotationKey])

	t.Log("Persist against absent empty evidence must not Patch")
	dcd := f.dcd.DeepCopy()
	dcd.ResourceVersion = "9"
	dcd.Generation = 2
	patchCalls := 0
	writer := interceptor.NewClient(f.reader(dcd), interceptor.Funcs{
		Patch: func(ctx context.Context, c client.WithWatch, obj client.Object, patch client.Patch, opts ...client.PatchOption) error {
			patchCalls++
			return c.Patch(ctx, obj, patch, opts...)
		},
	})
	result, err := PersistFailureRecords(t.Context(), writer, dcd, []Observation{unknown}, f.now)
	require.NoError(t, err)
	assert.True(t, result.Noop)
	assert.False(t, result.Written)
	assert.Equal(t, 0, patchCalls)

	t.Log("Genuinely absent annotation with valid CurrentFailure persists")
	fresh := f.dcd.DeepCopy()
	fresh.ResourceVersion = "10"
	fresh.Generation = 2
	freshWriter := interceptor.NewClient(f.reader(fresh), interceptor.Funcs{})
	written, err := PersistFailureRecords(t.Context(), freshWriter, fresh, []Observation{obs}, f.now)
	require.NoError(t, err)
	assert.True(t, written.Written)
	stored := &nvidiacomv1beta1.DynamoComponentDeployment{}
	require.NoError(t, freshWriter.Get(t.Context(), client.ObjectKeyFromObject(fresh), stored))
	assert.Contains(t, stored.Annotations, FailureRecordsAnnotationKey)
	assert.NotEmpty(t, stored.Annotations[FailureRecordsAnnotationKey])
}

func TestPersistFailureRecordsUIDResourceVersionGuard(t *testing.T) {
	f := newFixture()
	obs := currentFailureObservation(f, f.leaderPod, 23)
	dcd := f.dcd.DeepCopy()
	dcd.ResourceVersion = "7"
	dcd.Generation = 2

	t.Log("Capture the raw merge patch body for UID and resourceVersion")
	var sawPatch []byte
	base := f.reader(dcd)
	writer := interceptor.NewClient(base, interceptor.Funcs{
		Patch: func(ctx context.Context, c client.WithWatch, obj client.Object, patch client.Patch, opts ...client.PatchOption) error {
			data, err := patch.Data(obj)
			require.NoError(t, err)
			sawPatch = append([]byte(nil), data...)
			return c.Patch(ctx, obj, patch, opts...)
		},
	})

	result, err := PersistFailureRecords(t.Context(), writer, dcd, []Observation{obs}, f.now)
	require.NoError(t, err)
	require.True(t, result.Written)
	require.NotEmpty(t, sawPatch)

	var body map[string]any
	require.NoError(t, json.Unmarshal(sawPatch, &body))
	metadata := body["metadata"].(map[string]any)
	assert.Equal(t, string(dcd.UID), metadata["uid"])
	assert.Equal(t, "7", metadata["resourceVersion"])
	annotations := metadata["annotations"].(map[string]any)
	_, ok := annotations[FailureRecordsAnnotationKey]
	assert.True(t, ok)
	assert.Len(t, annotations, 1)
}

func TestPersistFailureRecordsSkipsNilBatchAndConflicts(t *testing.T) {
	f := newFixture()
	dcd := f.dcd.DeepCopy()
	dcd.ResourceVersion = "3"
	dcd.Generation = 2

	t.Log("Nil observation batch means observation failed and must not write")
	result, err := PersistFailureRecords(t.Context(), f.reader(dcd), dcd, nil, f.now)
	require.NoError(t, err)
	assert.True(t, result.Noop)
	assert.False(t, result.Written)

	t.Log("Conflict returns a bounded requeue without claiming stored success")
	obs := currentFailureObservation(f, f.leaderPod, 23)
	writer := interceptor.NewClient(f.reader(dcd), interceptor.Funcs{
		Patch: func(context.Context, client.WithWatch, client.Object, client.Patch, ...client.PatchOption) error {
			return apierrors.NewConflict(
				nvidiacomv1beta1.GroupVersion.WithResource("dynamocomponentdeployments").GroupResource(),
				dcd.Name,
				nil,
			)
		},
	})
	conflict, err := PersistFailureRecords(t.Context(), writer, dcd, []Observation{obs}, f.now)
	require.NoError(t, err)
	assert.True(t, conflict.Conflict)
	assert.False(t, conflict.Written)
	assert.Equal(t, FailureRecordConflictRequeue, conflict.RequeueAfter)

	t.Log("Invalid is unavailable without a hot conflict retry")
	invalidWriter := interceptor.NewClient(f.reader(dcd), interceptor.Funcs{
		Patch: func(context.Context, client.WithWatch, client.Object, client.Patch, ...client.PatchOption) error {
			return apierrors.NewInvalid(
				schema.GroupKind{Group: nvidiacomv1beta1.GroupVersion.Group, Kind: "DynamoComponentDeployment"},
				dcd.Name,
				nil,
			)
		},
	})
	invalid, err := PersistFailureRecords(t.Context(), invalidWriter, dcd, []Observation{obs}, f.now)
	require.NoError(t, err)
	assert.True(t, invalid.Unavailable)
	assert.False(t, invalid.Conflict)
	assert.False(t, invalid.Written)
	assert.Equal(t, "patch_rejected", invalid.UnavailableReason)
	assert.Equal(t, time.Duration(0), invalid.RequeueAfter)

	t.Log("Forbidden is unavailable without a hot conflict retry")
	forbiddenWriter := interceptor.NewClient(f.reader(dcd), interceptor.Funcs{
		Patch: func(context.Context, client.WithWatch, client.Object, client.Patch, ...client.PatchOption) error {
			return apierrors.NewForbidden(
				nvidiacomv1beta1.GroupVersion.WithResource("dynamocomponentdeployments").GroupResource(),
				dcd.Name,
				fmt.Errorf("denied"),
			)
		},
	})
	forbidden, err := PersistFailureRecords(t.Context(), forbiddenWriter, dcd, []Observation{obs}, f.now)
	require.NoError(t, err)
	assert.True(t, forbidden.Unavailable)
	assert.False(t, forbidden.Conflict)
	assert.Equal(t, time.Duration(0), forbidden.RequeueAfter)

	t.Log("RequestEntityTooLarge is unavailable without a hot conflict retry")
	tooLargeWriter := interceptor.NewClient(f.reader(dcd), interceptor.Funcs{
		Patch: func(context.Context, client.WithWatch, client.Object, client.Patch, ...client.PatchOption) error {
			return apierrors.NewRequestEntityTooLargeError("payload too large")
		},
	})
	tooLarge, err := PersistFailureRecords(t.Context(), tooLargeWriter, dcd, []Observation{obs}, f.now)
	require.NoError(t, err)
	assert.True(t, tooLarge.Unavailable)
	assert.False(t, tooLarge.Conflict)
	assert.False(t, tooLarge.Written)
	assert.Equal(t, "patch_rejected", tooLarge.UnavailableReason)
	assert.Equal(t, time.Duration(0), tooLarge.RequeueAfter)
}

func TestPersistFailureRecordsExplicitClockExpiry(t *testing.T) {
	f := newFixture()
	obs := currentFailureObservation(f, f.leaderPod, 23)
	dcd := f.dcd.DeepCopy()
	dcd.ResourceVersion = "11"
	dcd.Generation = 2

	result, err := PersistFailureRecords(t.Context(), f.reader(dcd), dcd, []Observation{obs}, f.now)
	require.NoError(t, err)
	require.True(t, result.Written)
	want := f.finished.Add(failureRecordRetention).Sub(f.now)
	assert.Equal(t, want, result.RequeueAfter)
}

func TestPersistFailureRecordsRejectsIncompleteCurrentRoot(t *testing.T) {
	f := newFixture()
	dcd := f.dcd.DeepCopy()
	dcd.ResourceVersion = "4"
	dcd.Generation = 2
	obs := currentFailureObservation(f, f.leaderPod, 23)
	obs.Namespace = ""
	out, err := PersistFailureRecords(t.Context(), f.reader(dcd), dcd, []Observation{obs}, f.now)
	require.NoError(t, err)
	assert.True(t, out.Unavailable)
	assert.Equal(t, "incomplete_current_failure_root", out.UnavailableReason)
	assert.False(t, out.Written)
}

func TestPersistFailureRecordsSimultaneousWritersPreserveUnion(t *testing.T) {
	f := newFixture()
	base := f.dcd.DeepCopy()
	base.ResourceVersion = "21"
	base.Generation = 2

	obsA := currentFailureObservation(f, f.leaderPod, 23)
	obsB := currentFailureObservation(f, f.leaderPod, 23)
	obsB.LeaderUID = "leader-b"
	obsB.PodUID = "pod-b"
	obsB.PodName = "decode-b"

	var patchCalls atomic.Int32
	var raceDone atomic.Bool
	shared := interceptor.NewClient(f.reader(base), interceptor.Funcs{
		Patch: func(ctx context.Context, c client.WithWatch, obj client.Object, patch client.Patch, opts ...client.PatchOption) error {
			if !raceDone.Load() {
				if patchCalls.Add(1) == 1 {
					return c.Patch(ctx, obj, patch, opts...)
				}
				return apierrors.NewConflict(
					nvidiacomv1beta1.GroupVersion.WithResource("dynamocomponentdeployments").GroupResource(),
					base.Name,
					nil,
				)
			}
			return c.Patch(ctx, obj, patch, opts...)
		},
	})

	start := make(chan struct{})
	var wg sync.WaitGroup
	results := make([]PersistResult, 2)
	errs := make([]error, 2)
	run := func(idx int, dcd *nvidiacomv1beta1.DynamoComponentDeployment, batch []Observation) {
		defer wg.Done()
		<-start
		results[idx], errs[idx] = PersistFailureRecords(t.Context(), shared, dcd, batch, f.now)
	}

	wg.Add(2)
	go run(0, base.DeepCopy(), []Observation{obsA})
	go run(1, base.DeepCopy(), []Observation{obsB})
	close(start)
	wg.Wait()

	require.NoError(t, errs[0])
	require.NoError(t, errs[1])
	var winner, loser PersistResult
	var loserBatch []Observation
	switch {
	case results[0].Written && results[1].Conflict:
		winner, loser, loserBatch = results[0], results[1], []Observation{obsB}
	case results[1].Written && results[0].Conflict:
		winner, loser, loserBatch = results[1], results[0], []Observation{obsA}
	default:
		t.Fatalf("expected one Written and one Conflict, got %#v %#v", results[0], results[1])
	}
	assert.True(t, winner.Written)
	assert.True(t, loser.Conflict)
	assert.False(t, loser.Written)
	assert.Equal(t, FailureRecordConflictRequeue, loser.RequeueAfter)

	t.Log("Fresh reread/remerge keeps the union of both writers' distinct records")
	raceDone.Store(true)
	live := &nvidiacomv1beta1.DynamoComponentDeployment{}
	require.NoError(t, shared.Get(t.Context(), client.ObjectKeyFromObject(base), live))
	require.Contains(t, live.Annotations, FailureRecordsAnnotationKey)
	second, err := PersistFailureRecords(t.Context(), shared, live.DeepCopy(), loserBatch, f.now.Add(time.Second))
	require.NoError(t, err)
	require.True(t, second.Written)

	final := &nvidiacomv1beta1.DynamoComponentDeployment{}
	require.NoError(t, shared.Get(t.Context(), client.ObjectKeyFromObject(base), final))
	var doc failureRecordsDocument
	require.NoError(t, json.Unmarshal([]byte(final.Annotations[FailureRecordsAnnotationKey]), &doc))
	assert.Len(t, doc.Entries, 2)
}

func TestUnknownAndNoCurrentFailureDoNotPersist(t *testing.T) {
	f := newFixture()
	unknown := currentFailureObservation(f, f.leaderPod, 23)
	unknown.Verdict = VerdictUnknown
	healthy := currentFailureObservation(f, f.leaderPod, 23)
	healthy.Verdict = VerdictNoCurrentFailureEvidence
	healthy.ExitCode = nil

	out := MergeFailureRecords(f.dcd.UID, "", nil, []Observation{unknown, healthy}, f.now)
	assert.False(t, out.Changed)
	assert.Equal(t, "", out.Encoded)
	assert.Equal(t, 0, out.RecordCount)
}

func currentFailureObservation(f *fixture, pod *corev1.Pod, exit int32) Observation {
	group := 0
	if pod.Labels[leaderworkersetv1.GroupIndexLabelKey] == "1" {
		group = 1
	}
	leaderUID := f.leaderPod.UID
	if pod.Labels[leaderworkersetv1.WorkerIndexLabelKey] == "0" {
		leaderUID = pod.UID
	}
	return Observation{
		Verdict:                 VerdictCurrentFailure,
		Reason:                  ReasonCurrentMainNonzeroExit,
		Namespace:               f.dcd.Namespace,
		DCDName:                 f.dcd.Name,
		DCDUID:                  f.dcd.UID,
		DCDGeneration:           f.dcd.Generation,
		LWSName:                 f.lws.Name,
		LWSUID:                  f.lws.UID,
		LWSGeneration:           f.lws.Generation,
		GroupIndex:              &group,
		Revision:                testRevision,
		LeaderUID:               leaderUID,
		PodName:                 pod.Name,
		PodUID:                  pod.UID,
		PodResourceVersion:      pod.ResourceVersion,
		Container:               commonconsts.MainContainerName,
		ExitCode:                &exit,
		FinishedAt:              f.finished.DeepCopy(),
		Ready:                   corev1.ConditionFalse,
		ReadyLastTransitionTime: f.readyLTT.DeepCopy(),
		ReadOrigin:              ReadOriginManagerCache,
	}
}
