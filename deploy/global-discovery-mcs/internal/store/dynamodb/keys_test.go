// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package dynamodb

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/store"
)

func TestKeyLayout(t *testing.T) {
	ek := store.ExportKey{ClusterSet: "prod", Cluster: "c1", Namespace: "llm", Name: "router"}
	tests := []struct {
		name string
		got  key
		want key
	}{
		{"clusterset index", clusterSetIndexKey("prod"), key{"CSX", "CS#prod"}},
		{"cluster", clusterKey(store.ClusterKey{ClusterSet: "prod", Cluster: "c1"}), key{"CS#prod", "CL#c1"}},
		{"export", exportKey(ek), key{"CS#prod", "EX#c1#llm#router"}},
		{"endpoint partition marker", endpointMarkKey("prod", "c1"), key{"CS#prod", "EPX#c1"}},
		{"endpoint set", endpointSetKey(ek), key{"EP#prod#c1", "llm#router"}},
	}
	for _, tt := range tests {
		assert.Equal(t, tt.want, tt.got, tt.name)
	}
	assert.Equal(t, "EX#c1#", exportSKPrefix("c1"))
}

func TestKeyEscaping(t *testing.T) {
	odd := "a#b%23c%"
	assert.Equal(t, "a%23b%2523c%25", esc(odd))
	assert.Equal(t, odd, unesc(esc(odd)))
	assert.NotContains(t, esc(odd), "#")

	ek := store.ExportKey{ClusterSet: odd, Cluster: odd, Namespace: odd, Name: odd}
	k := exportKey(ek)
	cs, ok := parseClusterSetPK(k.pk)
	assert.True(t, ok)
	assert.Equal(t, odd, cs)
	got, ok := parseExportSK(cs, k.sk)
	assert.True(t, ok)
	assert.Equal(t, ek, got)

	ek2 := endpointSetKey(ek)
	gcs, gcluster, ok := parseEndpointPK(ek2.pk)
	assert.True(t, ok)
	got, ok = parseEndpointSetSK(gcs, gcluster, ek2.sk)
	assert.True(t, ok)
	assert.Equal(t, ek, got)

	ck := store.ClusterKey{ClusterSet: odd, Cluster: odd}
	gotCluster, ok := parseClusterSK(odd, clusterKey(ck).sk)
	assert.True(t, ok)
	assert.Equal(t, ck, gotCluster)

	name, ok := parseClusterSetIndexSK(clusterSetIndexKey(odd).sk)
	assert.True(t, ok)
	assert.Equal(t, odd, name)
	cluster, ok := parseEndpointMarkSK(endpointMarkKey(odd, odd).sk)
	assert.True(t, ok)
	assert.Equal(t, odd, cluster)
}

// TestKeyPrefixesDoNotCollide checks that the prefix of one cluster never
// matches the records of another whose name extends it.
func TestKeyPrefixesDoNotCollide(t *testing.T) {
	c10 := exportKey(store.ExportKey{ClusterSet: "cs", Cluster: "c10", Namespace: "ns", Name: "svc"})
	assert.False(t, strings.HasPrefix(c10.sk, exportSKPrefix("c1")))
	assert.NotEqual(t, endpointPK("cs", "c1#x"), endpointPK("cs#c1", "x"))
}

func TestParseRejectsOtherShapes(t *testing.T) {
	_, ok := parseClusterSK("cs", "EX#c1#ns#name")
	assert.False(t, ok)
	_, ok = parseExportSK("cs", "EX#c1#ns")
	assert.False(t, ok)
	_, ok = parseEndpointSetSK("cs", "c1", "ns")
	assert.False(t, ok)
	_, _, ok = parseEndpointPK("CS#cs")
	assert.False(t, ok)
	_, ok = parseClusterSetPK("CSX")
	assert.False(t, ok)
	_, ok = parseEndpointMarkSK("EP#c1")
	assert.False(t, ok)
	_, ok = parseClusterSetIndexSK("PING")
	assert.False(t, ok)
}
