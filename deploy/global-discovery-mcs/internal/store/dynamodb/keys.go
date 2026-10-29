// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package dynamodb

import (
	"strings"

	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/store"
)

// Item kinds are told apart by key prefixes:
//
//	Item                     pk                          sk
//	clusterset index         CSX                         CS#<clusterset>
//	cluster                  CS#<clusterset>             CL#<cluster>
//	service export           CS#<clusterset>             EX#<cluster>#<namespace>#<name>
//	endpoint partition mark  CS#<clusterset>             EPX#<cluster>
//	endpoint set             EP#<clusterset>#<cluster>   <namespace>#<name>
//
// Every key component is escaped so that it never contains the separator:
// '%' becomes "%25" and '#' becomes "%23". DNS labels, which is what the API
// accepts, are unchanged. Prefix queries therefore always end with a
// separator, so "EX#c1#" never matches the exports of cluster "c10".
const (
	sep = "#"

	clusterSetIndexPK = "CSX"
	clusterSetPrefix  = "CS#"
	clusterPrefix     = "CL#"
	exportPrefix      = "EX#"
	endpointMarkPfx   = "EPX#"
	endpointPKPrefix  = "EP#"

	// pingSK is read by Ping. No item has this key.
	pingSK = "PING"
)

var (
	escaper   = strings.NewReplacer("%", "%25", "#", "%23")
	unescaper = strings.NewReplacer("%25", "%", "%23", "#")
)

func esc(s string) string   { return escaper.Replace(s) }
func unesc(s string) string { return unescaper.Replace(s) }

// key is the primary key of one item.
type key struct {
	pk, sk string
}

func clusterSetIndexKey(clusterSet string) key {
	return key{pk: clusterSetIndexPK, sk: clusterSetPrefix + esc(clusterSet)}
}

func clusterSetPK(clusterSet string) string {
	return clusterSetPrefix + esc(clusterSet)
}

func clusterKey(k store.ClusterKey) key {
	return key{pk: clusterSetPK(k.ClusterSet), sk: clusterPrefix + esc(k.Cluster)}
}

func exportKey(k store.ExportKey) key {
	return key{pk: clusterSetPK(k.ClusterSet), sk: exportSKPrefix(k.Cluster) + esc(k.Namespace) + sep + esc(k.Name)}
}

// exportSKPrefix is the sort key prefix of every export of a cluster.
func exportSKPrefix(cluster string) string {
	return exportPrefix + esc(cluster) + sep
}

func endpointMarkKey(clusterSet, cluster string) key {
	return key{pk: clusterSetPK(clusterSet), sk: endpointMarkPfx + esc(cluster)}
}

// endpointPK is the partition holding every endpoint set of a cluster.
func endpointPK(clusterSet, cluster string) string {
	return endpointPKPrefix + esc(clusterSet) + sep + esc(cluster)
}

func endpointSetKey(k store.ExportKey) key {
	return key{pk: endpointPK(k.ClusterSet, k.Cluster), sk: esc(k.Namespace) + sep + esc(k.Name)}
}

// splitSK removes prefix from sk and splits the rest into n unescaped
// components. It reports false when sk does not have that shape.
func splitSK(sk, prefix string, n int) ([]string, bool) {
	rest, ok := strings.CutPrefix(sk, prefix)
	if !ok {
		return nil, false
	}
	parts := strings.Split(rest, sep)
	if len(parts) != n {
		return nil, false
	}
	for i, p := range parts {
		parts[i] = unesc(p)
	}
	return parts, true
}

func parseClusterSetIndexSK(sk string) (string, bool) {
	parts, ok := splitSK(sk, clusterSetPrefix, 1)
	if !ok {
		return "", false
	}
	return parts[0], true
}

func parseClusterSK(clusterSet, sk string) (store.ClusterKey, bool) {
	parts, ok := splitSK(sk, clusterPrefix, 1)
	if !ok {
		return store.ClusterKey{}, false
	}
	return store.ClusterKey{ClusterSet: clusterSet, Cluster: parts[0]}, true
}

func parseExportSK(clusterSet, sk string) (store.ExportKey, bool) {
	parts, ok := splitSK(sk, exportPrefix, 3)
	if !ok {
		return store.ExportKey{}, false
	}
	return store.ExportKey{ClusterSet: clusterSet, Cluster: parts[0], Namespace: parts[1], Name: parts[2]}, true
}

func parseEndpointMarkSK(sk string) (string, bool) {
	parts, ok := splitSK(sk, endpointMarkPfx, 1)
	if !ok {
		return "", false
	}
	return parts[0], true
}

func parseEndpointSetSK(clusterSet, cluster, sk string) (store.ExportKey, bool) {
	parts, ok := splitSK(sk, "", 2)
	if !ok {
		return store.ExportKey{}, false
	}
	return store.ExportKey{ClusterSet: clusterSet, Cluster: cluster, Namespace: parts[0], Name: parts[1]}, true
}

func parseClusterSetPK(pk string) (string, bool) {
	return parseClusterSetIndexSK(pk)
}

func parseEndpointPK(pk string) (clusterSet, cluster string, ok bool) {
	parts, ok := splitSK(pk, endpointPKPrefix, 2)
	if !ok {
		return "", "", false
	}
	return parts[0], parts[1], true
}
