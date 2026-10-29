// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package dynamodb

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	ddb "github.com/aws/aws-sdk-go-v2/service/dynamodb"
	ddbtypes "github.com/aws/aws-sdk-go-v2/service/dynamodb/types"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/store"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/store/storetest"
)

func newTestStore(t *testing.T, db *fakeDB, opts ...func(*Options)) *Store {
	t.Helper()
	o := Options{Client: db, TableName: "global-discovery-mcs", QueryConcurrency: 2}
	for _, fn := range opts {
		fn(&o)
	}
	st, err := New(o)
	require.NoError(t, err)
	return st
}

// TestConformanceFake runs the store suite against the in-memory fake.
// Every subtest gets its own Store on one shared fake, as replicas share
// one table; small query pages exercise pagination.
func TestConformanceFake(t *testing.T) {
	db := newFakeDB()
	storetest.Run(t, func(t *testing.T) store.Store { return newTestStore(t, db) })
}

func fastBackoff(t *testing.T) {
	base, maxB := baseBackoff, maxBackoff
	baseBackoff, maxBackoff = time.Millisecond, 2*time.Millisecond
	t.Cleanup(func() { baseBackoff, maxBackoff = base, maxB })
}

func ck(cluster string) store.ClusterKey {
	return store.ClusterKey{ClusterSet: "cs", Cluster: cluster}
}

func ek(cluster, name string) store.ExportKey {
	return store.ExportKey{ClusterSet: "cs", Cluster: cluster, Namespace: "ns", Name: name}
}

func TestNew(t *testing.T) {
	_, err := New(Options{TableName: "global-discovery-mcs"})
	assert.ErrorContains(t, err, "Client is required")
	_, err = New(Options{Client: newFakeDB()})
	assert.ErrorContains(t, err, "TableName is required")

	st, err := New(Options{Client: newFakeDB(), TableName: "global-discovery-mcs", QueryConcurrency: -3})
	require.NoError(t, err)
	assert.Equal(t, 1, st.concurrency)
	assert.Equal(t, DefaultExpiryAfter, st.expiryAfter)
	assert.NotNil(t, st.clock)

	st, err = New(Options{Client: newFakeDB(), TableName: "global-discovery-mcs"})
	require.NoError(t, err)
	assert.Equal(t, DefaultQueryConcurrency, st.concurrency)
}

func TestClosedStore(t *testing.T) {
	st := newTestStore(t, newFakeDB())
	require.NoError(t, st.Ping(t.Context()))
	require.NoError(t, st.Close())
	require.NoError(t, st.Close(), "Close is idempotent")
	for name, err := range allCalls(t, st) {
		assert.ErrorIs(t, err, store.ErrUnavailable, name)
	}
}

// allCalls calls every Store method once and returns the errors by name.
func allCalls(t *testing.T, st *Store) map[string]error {
	ctx := t.Context()
	errs := map[string]error{}
	_, errs["PutCluster"] = st.PutCluster(ctx, store.ClusterRecord{Key: ck("c1")}, t0)
	_, errs["RenewClusterLease"] = st.RenewClusterLease(ctx, ck("c1"), 1, t0)
	_, errs["GetCluster"] = st.GetCluster(ctx, ck("c1"))
	errs["DeleteCluster"] = st.DeleteCluster(ctx, ck("c1"))
	errs["PutExport"] = st.PutExport(ctx, store.ExportRecord{Key: ek("c1", "s"), Generation: 1})
	errs["DeleteExport"] = st.DeleteExport(ctx, ek("c1", "s"))
	errs["PutEndpointSet"] = st.PutEndpointSet(ctx, store.EndpointSetRecord{Key: ek("c1", "s"), Generation: 1})
	errs["DeleteEndpointSet"] = st.DeleteEndpointSet(ctx, ek("c1", "s"))
	_, errs["ListClusterSets"] = st.ListClusterSets(ctx)
	_, errs["LoadClusterSet"] = st.LoadClusterSet(ctx, "cs")
	errs["Ping"] = st.Ping(ctx)
	return errs
}

// TestBackendUnavailable checks that every method reports a throttled or
// unreachable backend as store.ErrUnavailable.
func TestBackendUnavailable(t *testing.T) {
	for _, cause := range []error{
		&ddbtypes.ProvisionedThroughputExceededException{Message: aws.String("slow down")},
		&ddbtypes.ResourceNotFoundException{Message: aws.String("no table")},
		errors.New("dial tcp: connection refused"),
	} {
		db := newFakeDB()
		db.before = func(string) error { return cause }
		st := newTestStore(t, db)
		for name, err := range allCalls(t, st) {
			assert.ErrorIs(t, err, store.ErrUnavailable, "%s with %v", name, cause)
		}
	}
}

// TestBackendUnavailableAfterFirstCall fails each operation kind only after
// the first call of that kind, which reaches the error paths behind the
// index and marker writes and the initial reads.
func TestBackendUnavailableAfterFirstCall(t *testing.T) {
	for _, failing := range []string{"GetItem", "PutItem", "UpdateItem", "Query", "BatchWriteItem", "DeleteItem"} {
		t.Run(failing, func(t *testing.T) {
			db := newFakeDB()
			st := newTestStore(t, db)
			ctx := t.Context()
			reg, err := st.PutCluster(ctx, store.ClusterRecord{Key: ck("c1")}, t0)
			require.NoError(t, err)
			require.NoError(t, st.PutEndpointSet(ctx, store.EndpointSetRecord{Key: ek("c1", "s"), Generation: 1}))

			db.mu.Lock()
			db.before = func(op string) error {
				if op == failing {
					return &ddbtypes.InternalServerError{Message: aws.String("oops")}
				}
				return nil
			}
			db.mu.Unlock()
			errs := allCalls(t, st)
			_, errs["RenewClusterLeaseKnown"] = st.RenewClusterLease(ctx, ck("c1"), reg.Lease.Epoch, t0)
			failed := 0
			for name, err := range errs {
				if err != nil && !errors.Is(err, store.ErrNotFound) && !errors.Is(err, store.ErrLeaseEpochMismatch) && !errors.Is(err, store.ErrStaleGeneration{}) {
					assert.ErrorIs(t, err, store.ErrUnavailable, name)
					failed++
				}
			}
			assert.Positive(t, failed)
		})
	}
}

func TestCanceledDuringLoad(t *testing.T) {
	db := newFakeDB()
	st := newTestStore(t, db)
	for i := range 5 {
		require.NoError(t, st.PutEndpointSet(t.Context(), store.EndpointSetRecord{Key: ek(fmt.Sprintf("c%d", i), "s"), Generation: 1}))
	}
	ctx, cancel := context.WithCancel(t.Context())
	queries := 0
	db.before = func(op string) error {
		if op == "Query" {
			queries++
			if queries == 2 {
				cancel()
			}
		}
		return nil
	}
	_, err := st.LoadClusterSet(ctx, "cs")
	assert.ErrorIs(t, err, context.Canceled)
}

func TestDeleteClusterRetriesUnprocessed(t *testing.T) {
	fastBackoff(t)
	db := newFakeDB()
	st := newTestStore(t, db)
	ctx := t.Context()
	for i := range 30 {
		require.NoError(t, st.PutExport(ctx, store.ExportRecord{Key: ek("c1", fmt.Sprintf("svc-%02d", i)), Generation: 1}))
		require.NoError(t, st.PutEndpointSet(ctx, store.EndpointSetRecord{Key: ek("c1", fmt.Sprintf("svc-%02d", i)), Generation: 1}))
	}
	db.unprocessed = 3
	require.NoError(t, st.DeleteCluster(ctx, ck("c1")))
	assert.GreaterOrEqual(t, db.calls["BatchWriteItem"], 4, "62 deletes in 3 chunks plus retries")
	snap, err := st.LoadClusterSet(ctx, "cs")
	require.NoError(t, err)
	assert.Empty(t, snap.Exports)
	assert.Empty(t, snap.EndpointSets)

	require.NoError(t, st.PutExport(ctx, store.ExportRecord{Key: ek("c1", "svc"), Generation: 1}))
	db.unprocessed = 1000
	err = st.DeleteCluster(ctx, ck("c1"))
	assert.ErrorIs(t, err, store.ErrUnavailable)
	assert.ErrorContains(t, err, "still unprocessed")

	ctx2, cancel := context.WithCancel(ctx)
	db.before = func(op string) error {
		if op == "BatchWriteItem" {
			cancel()
		}
		return nil
	}
	assert.ErrorIs(t, st.DeleteCluster(ctx2, ck("c1")), context.Canceled)
}

func TestContention(t *testing.T) {
	db := newFakeDB()
	st := newTestStore(t, db)
	ctx := t.Context()
	_, err := st.PutCluster(ctx, store.ClusterRecord{Key: ck("c1")}, t0)
	require.NoError(t, err)

	// Another writer changes every item before each write.
	gen := uint64(100)
	db.before = func(op string) error {
		switch op {
		case "PutItem", "UpdateItem":
			gen++
			for _, it := range db.items {
				if _, ok := it[attrGen]; ok {
					it[attrGen] = numU(gen)
					it[attrBody] = bin([]byte(fmt.Sprintf("\x0a\x01%c", 'a'+gen%26)))
				}
			}
		}
		return nil
	}
	_, err = st.PutCluster(ctx, store.ClusterRecord{Key: ck("c1")}, t0)
	assert.ErrorIs(t, err, store.ErrUnavailable)
	assert.ErrorContains(t, err, "too many concurrent writes")
}

func TestEndpointMarkCache(t *testing.T) {
	db := newFakeDB()
	now := t0
	st := newTestStore(t, db, func(o *Options) { o.Clock = func() time.Time { return now } })
	ctx := t.Context()
	markWrites := 0
	db.before = func(op string) error {
		if op == "PutItem" {
			markWrites++
		}
		return nil
	}
	put := func(gen uint64) {
		t.Helper()
		require.NoError(t, st.PutEndpointSet(ctx, store.EndpointSetRecord{Key: ek("c1", "s"), Generation: gen}))
	}
	put(1) // index, marker, set
	put(2) // set
	assert.Equal(t, 4, markWrites)
	now = now.Add(markCacheTTL + time.Second)
	put(3) // marker, set
	assert.Equal(t, 6, markWrites)
	require.NoError(t, st.DeleteCluster(ctx, ck("c1")))
	put(4) // marker, set
	assert.Equal(t, 8, markWrites)
	snap, err := st.LoadClusterSet(ctx, "cs")
	require.NoError(t, err)
	require.Len(t, snap.EndpointSets, 1, "an orphan endpoint set is found through its marker")
}

func TestLoadClusterSetErrors(t *testing.T) {
	db := newFakeDB()
	st := newTestStore(t, db)
	ctx := t.Context()
	require.NoError(t, st.PutEndpointSet(ctx, store.EndpointSetRecord{Key: ek("c1", "s"), Generation: 1}))
	setID := itemID(endpointSetKey(ek("c1", "s")).attributes())
	db.items[setID][attrBody] = bin([]byte{0xff})
	_, err := st.LoadClusterSet(ctx, "cs")
	assert.ErrorContains(t, err, "unmarshal body")

	db.items[setID][attrGen] = str("not a number")
	_, err = st.LoadClusterSet(ctx, "cs")
	assert.ErrorContains(t, err, "unmarshal item")
	delete(db.items, setID)

	for _, write := range []func(){
		func() { _, _ = st.PutCluster(ctx, store.ClusterRecord{Key: ck("c2")}, t0) },
		func() { _ = st.PutExport(ctx, store.ExportRecord{Key: ek("c2", "s"), Generation: 1}) },
	} {
		write()
		for id, it := range db.items {
			if strings.HasPrefix(id, "CS#cs\x00") && it[attrBody] != nil {
				it[attrBody] = bin([]byte{0xff})
			}
		}
		_, err = st.LoadClusterSet(ctx, "cs")
		assert.ErrorContains(t, err, "unmarshal body")
		for id := range db.items {
			if strings.HasPrefix(id, "CS#cs\x00") {
				delete(db.items, id)
			}
		}
	}
}

func TestCorruptItemsFailReadModifyWrite(t *testing.T) {
	db := newFakeDB()
	st := newTestStore(t, db)
	ctx := t.Context()
	reg, err := st.PutCluster(ctx, store.ClusterRecord{Key: ck("c1")}, t0)
	require.NoError(t, err)
	for _, it := range db.items {
		if it[attrBody] != nil {
			it[attrBody] = bin([]byte{0xff})
		}
	}
	_, err = st.PutCluster(ctx, store.ClusterRecord{Key: ck("c1")}, t0)
	assert.Error(t, err)
	_, err = st.GetCluster(ctx, ck("c1"))
	assert.Error(t, err)
	_, err = st.RenewClusterLease(ctx, ck("c1"), reg.Lease.Epoch, t0)
	assert.Error(t, err)
}

func TestInvalidRecords(t *testing.T) {
	st := newTestStore(t, newFakeDB())
	ctx := t.Context()
	err := st.PutExport(ctx, store.ExportRecord{Key: ek("c1", "s"), Spec: store.ExportSpec{Type: "LoadBalancer"}})
	assert.ErrorContains(t, err, "unknown service type")
	err = st.PutEndpointSet(ctx, store.EndpointSetRecord{Key: ek("c1", "s"), Endpoints: []store.Endpoint{{AddressType: "MAC"}}})
	assert.ErrorContains(t, err, "unknown address type")

	big := strings.Repeat("x", maxItemBytes)
	err = st.PutExport(ctx, store.ExportRecord{Key: ek("c1", "s"), Spec: store.ExportSpec{Annotations: map[string]string{"big": big}}})
	assert.ErrorIs(t, err, store.ErrTooLarge)
	_, err = st.PutCluster(ctx, store.ClusterRecord{Key: ck("c1"), Labels: map[string]string{"big": big}}, t0)
	assert.ErrorIs(t, err, store.ErrTooLarge)
}

func TestExpiry(t *testing.T) {
	db := newFakeDB()
	now := t0
	st := newTestStore(t, db, func(o *Options) {
		o.ExpiryAfter = time.Hour
		o.Clock = func() time.Time { return now }
	})
	ctx := t.Context()
	reg, err := st.PutCluster(ctx, store.ClusterRecord{Key: ck("c1")}, t0)
	require.NoError(t, err)
	clusterID := itemID(clusterKey(ck("c1")).attributes())
	assert.Equal(t, numI(t0.Add(time.Hour).Unix()), db.items[clusterID][attrExpiresAt])

	// A write without a server time expires from the store's clock.
	now = t0.Add(10 * time.Minute)
	_, err = st.RenewClusterLease(ctx, ck("c1"), reg.Lease.Epoch, time.Time{})
	require.NoError(t, err)
	assert.Equal(t, numI(now.Add(time.Hour).Unix()), db.items[clusterID][attrExpiresAt])
	assert.NotContains(t, db.items[clusterID], attrLeaseRenewedAt, "a zero renewal time is stored as absent")

	off := newTestStore(t, db, func(o *Options) { o.ExpiryAfter = -1 })
	_, err = off.PutCluster(ctx, store.ClusterRecord{Key: ck("c2")}, t0)
	require.NoError(t, err)
	assert.NotContains(t, db.items[itemID(clusterKey(ck("c2")).attributes())], attrExpiresAt)
}

func TestListClusterSetsIgnoresOtherKeys(t *testing.T) {
	db := newFakeDB()
	st := newTestStore(t, db)
	ctx := t.Context()
	_, err := st.PutCluster(ctx, store.ClusterRecord{Key: ck("c1")}, t0)
	require.NoError(t, err)
	extra := key{pk: clusterSetIndexPK, sk: "CS#a#b"}.attributes()
	db.items[itemID(extra)] = extra
	names, err := st.ListClusterSets(ctx)
	require.NoError(t, err)
	assert.Equal(t, []string{"cs"}, names)
}

// TestFakeRejectsUnusedPlaceholders keeps the fake honest about the
// DynamoDB rule the store must follow.
func TestFakeRejectsUnusedPlaceholders(t *testing.T) {
	db := newFakeDB()
	k := clusterKey(ck("c1")).attributes()
	_, err := db.PutItem(t.Context(), &ddb.PutItemInput{
		TableName:                aws.String("global-discovery-mcs"),
		Item:                     k,
		ConditionExpression:      aws.String("attribute_not_exists(#pk)"),
		ExpressionAttributeNames: map[string]string{"#pk": "pk", "#gen": "gen"},
	})
	assert.ErrorContains(t, err, "unused in expressions")
	_, err = db.UpdateItem(t.Context(), &ddb.UpdateItemInput{
		TableName:                 aws.String("global-discovery-mcs"),
		Key:                       k,
		UpdateExpression:          aws.String("SET #gen = :gen"),
		ExpressionAttributeNames:  map[string]string{"#gen": "gen"},
		ExpressionAttributeValues: map[string]ddbtypes.AttributeValue{":gen": numU(1), ":x": numU(2)},
	})
	assert.ErrorContains(t, err, "unused in expressions")
}

func TestConcurrentStoresOnOneTable(t *testing.T) {
	db := newFakeDB()
	a, b := newTestStore(t, db), newTestStore(t, db)
	ctx := t.Context()
	var wg sync.WaitGroup
	for i := range 20 {
		st := a
		if i%2 == 1 {
			st = b
		}
		wg.Go(func() {
			_, err := st.PutCluster(ctx, store.ClusterRecord{Key: ck("c1")}, t0)
			assert.NoError(t, err)
		})
	}
	wg.Wait()
	got, err := a.GetCluster(ctx, ck("c1"))
	require.NoError(t, err)
	assert.Equal(t, uint64(20), got.Generation)
	assert.Equal(t, uint64(t0.UnixMilli())+19, got.Lease.Epoch)
}
