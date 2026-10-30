// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package merge

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/api"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/store"
)

const (
	localCluster  = "local"
	remoteCluster = "remote"
)

var (
	nowTime   = time.Date(2026, 10, 26, 12, 0, 0, 0, time.UTC)
	olderTime = nowTime.AddDate(0, 0, -1)
	newerTime = nowTime.AddDate(0, 0, 1)
)

func ipv4() []store.IPFamily { return []store.IPFamily{store.IPFamilyIPv4} }

func clientIP(timeout int32) *store.SessionAffinityConfig {
	return &store.SessionAffinityConfig{ClientIP: &store.ClientIPConfig{TimeoutSeconds: timeout}}
}

// The conflict cases, one per field that can conflict. Each runs with the
// remote export older and with the local export older: the oldest export wins
// and the reason is the mcs-api condition reason of the conflicting field.
var conflictCases = []struct {
	name        string
	local       store.ExportSpec
	remote      store.ExportSpec
	remoteWins  func(t *testing.T, s store.ExportSpec)
	localWins   func(t *testing.T, s store.ExportSpec)
	reason      string
	msgIncludes string
}{
	{
		name:   "conflict-type",
		local:  store.ExportSpec{Type: store.ServiceTypeClusterSetIP, SessionAffinity: store.SessionAffinityNone, IPFamilies: ipv4()},
		remote: store.ExportSpec{Type: store.ServiceTypeHeadless, SessionAffinity: store.SessionAffinityNone, IPFamilies: ipv4()},
		remoteWins: func(t *testing.T, s store.ExportSpec) {
			assert.Equal(t, store.ServiceTypeHeadless, s.Type)
		},
		localWins: func(t *testing.T, s store.ExportSpec) {
			assert.Equal(t, store.ServiceTypeClusterSetIP, s.Type)
		},
		reason:      api.ConditionReasonTypeConflict,
		msgIncludes: "1/2 clusters disagree",
	},
	{
		name:   "conflict-port-name",
		local:  store.ExportSpec{Type: store.ServiceTypeClusterSetIP, IPFamilies: ipv4(), Ports: []store.ServicePort{{Port: 4242}}},
		remote: store.ExportSpec{Type: store.ServiceTypeClusterSetIP, IPFamilies: ipv4(), Ports: []store.ServicePort{{Name: "remote", Port: 4242}}},
		remoteWins: func(t *testing.T, s store.ExportSpec) {
			require.Len(t, s.Ports, 1)
			assert.Equal(t, "remote", s.Ports[0].Name)
		},
		localWins: func(t *testing.T, s store.ExportSpec) {
			require.Len(t, s.Ports, 1)
			assert.Empty(t, s.Ports[0].Name)
		},
		reason:      api.ConditionReasonPortConflict,
		msgIncludes: "Conflicting port name",
	},
	{
		name:   "conflict-port-appprotocol",
		local:  store.ExportSpec{Type: store.ServiceTypeClusterSetIP, IPFamilies: ipv4(), Ports: []store.ServicePort{{Port: 4242}}},
		remote: store.ExportSpec{Type: store.ServiceTypeClusterSetIP, IPFamilies: ipv4(), Ports: []store.ServicePort{{Port: 4242, AppProtocol: "something-else"}}},
		remoteWins: func(t *testing.T, s store.ExportSpec) {
			require.Len(t, s.Ports, 1)
			assert.Equal(t, "something-else", s.Ports[0].AppProtocol)
		},
		localWins: func(t *testing.T, s store.ExportSpec) {
			require.Len(t, s.Ports, 1)
			assert.Empty(t, s.Ports[0].AppProtocol)
		},
		reason:      api.ConditionReasonPortConflict,
		msgIncludes: "Conflicting appProtocol",
	},
	{
		name:   "conflict-duplicated-port-name",
		local:  store.ExportSpec{Type: store.ServiceTypeClusterSetIP, IPFamilies: ipv4(), Ports: []store.ServicePort{{Name: "myport", Port: 4242}}},
		remote: store.ExportSpec{Type: store.ServiceTypeClusterSetIP, IPFamilies: ipv4(), Ports: []store.ServicePort{{Name: "myport", Port: 4243}}},
		remoteWins: func(t *testing.T, s store.ExportSpec) {
			require.Len(t, s.Ports, 1)
			assert.Equal(t, int32(4243), s.Ports[0].Port)
		},
		localWins: func(t *testing.T, s store.ExportSpec) {
			require.Len(t, s.Ports, 1)
			assert.Equal(t, int32(4242), s.Ports[0].Port)
		},
		reason:      api.ConditionReasonPortConflict,
		msgIncludes: "Duplicated port name",
	},
	{
		name:  "conflict-port-union-1",
		local: store.ExportSpec{Type: store.ServiceTypeClusterSetIP, Ports: []store.ServicePort{{Name: "myport1", Port: 4242}}},
		remote: store.ExportSpec{Type: store.ServiceTypeClusterSetIP, IPFamilies: ipv4(), Ports: []store.ServicePort{
			{Name: "myport1", Port: 4242}, {Name: "myport2", Port: 4243},
		}},
		remoteWins: func(t *testing.T, s store.ExportSpec) {
			assert.Equal(t, []store.ServicePort{{Name: "myport1", Port: 4242}, {Name: "myport2", Port: 4243}}, s.Ports)
		},
		localWins: func(t *testing.T, s store.ExportSpec) {
			assert.Equal(t, []store.ServicePort{{Name: "myport1", Port: 4242}, {Name: "myport2", Port: 4243}}, s.Ports)
		},
		reason:      api.ConditionReasonPortConflict,
		msgIncludes: "do not match ports",
	},
	{
		name: "conflict-port-union-2",
		local: store.ExportSpec{Type: store.ServiceTypeClusterSetIP, Ports: []store.ServicePort{
			{Name: "myport1", Port: 4242}, {Name: "myport2", Port: 4243},
		}},
		remote: store.ExportSpec{Type: store.ServiceTypeClusterSetIP, IPFamilies: ipv4(), Ports: []store.ServicePort{{Name: "myport1", Port: 4242}}},
		remoteWins: func(t *testing.T, s store.ExportSpec) {
			assert.Equal(t, []store.ServicePort{{Name: "myport1", Port: 4242}, {Name: "myport2", Port: 4243}}, s.Ports)
		},
		localWins: func(t *testing.T, s store.ExportSpec) {
			assert.Equal(t, []store.ServicePort{{Name: "myport1", Port: 4242}, {Name: "myport2", Port: 4243}}, s.Ports)
		},
		reason:      api.ConditionReasonPortConflict,
		msgIncludes: "do not match ports",
	},
	{
		name:   "conflict-session-affinity",
		local:  store.ExportSpec{Type: store.ServiceTypeClusterSetIP, SessionAffinity: store.SessionAffinityNone, IPFamilies: ipv4()},
		remote: store.ExportSpec{Type: store.ServiceTypeClusterSetIP, SessionAffinity: store.SessionAffinityClientIP, IPFamilies: ipv4()},
		remoteWins: func(t *testing.T, s store.ExportSpec) {
			assert.Equal(t, store.SessionAffinityClientIP, s.SessionAffinity)
		},
		localWins: func(t *testing.T, s store.ExportSpec) {
			assert.Equal(t, store.SessionAffinityNone, s.SessionAffinity)
		},
		reason: api.ConditionReasonSessionAffinityConflict,
	},
	{
		name: "conflict-session-affinity-config",
		local: store.ExportSpec{
			Type: store.ServiceTypeClusterSetIP, SessionAffinity: store.SessionAffinityClientIP,
			SessionAffinityConfig: clientIP(4242), IPFamilies: ipv4(),
		},
		remote: store.ExportSpec{
			Type: store.ServiceTypeClusterSetIP, SessionAffinity: store.SessionAffinityClientIP,
			SessionAffinityConfig: clientIP(42), IPFamilies: ipv4(),
		},
		remoteWins: func(t *testing.T, s store.ExportSpec) {
			assert.Equal(t, int32(42), s.SessionAffinityConfig.ClientIP.TimeoutSeconds)
		},
		localWins: func(t *testing.T, s store.ExportSpec) {
			assert.Equal(t, int32(4242), s.SessionAffinityConfig.ClientIP.TimeoutSeconds)
		},
		reason: api.ConditionReasonSessionAffinityConfigConflict,
	},
	{
		name: "conflict-annotations",
		local: store.ExportSpec{Type: store.ServiceTypeClusterSetIP, IPFamilies: ipv4(), Annotations: map[string]string{
			"example.com/sync-endpoint-slices": "true",
		}},
		remote: store.ExportSpec{Type: store.ServiceTypeClusterSetIP, IPFamilies: ipv4(), Annotations: map[string]string{
			"example.com/sync-endpoint-slices": "true",
			"example.com/l7-balancer":          "true",
		}},
		remoteWins: func(t *testing.T, s store.ExportSpec) {
			assert.Equal(t, map[string]string{
				"example.com/sync-endpoint-slices": "true",
				"example.com/l7-balancer":          "true",
			}, s.Annotations)
		},
		localWins: func(t *testing.T, s store.ExportSpec) {
			assert.Equal(t, map[string]string{"example.com/sync-endpoint-slices": "true"}, s.Annotations)
		},
		reason: api.ConditionReasonAnnotationsConflict,
	},
	{
		name:   "conflict-labels",
		local:  store.ExportSpec{Type: store.ServiceTypeClusterSetIP, IPFamilies: ipv4(), Labels: map[string]string{"my-label": "test"}},
		remote: store.ExportSpec{Type: store.ServiceTypeClusterSetIP, IPFamilies: ipv4(), Labels: map[string]string{"my-label": "test", "my-label2": "test"}},
		remoteWins: func(t *testing.T, s store.ExportSpec) {
			assert.Equal(t, map[string]string{"my-label": "test", "my-label2": "test"}, s.Labels)
		},
		localWins: func(t *testing.T, s store.ExportSpec) {
			assert.Equal(t, map[string]string{"my-label": "test"}, s.Labels)
		},
		reason: api.ConditionReasonLabelsConflict,
	},
	{
		name:   "conflict-internal-traffic-policy",
		local:  store.ExportSpec{Type: store.ServiceTypeClusterSetIP, IPFamilies: ipv4(), InternalTrafficPolicy: store.InternalTrafficPolicyLocal},
		remote: store.ExportSpec{Type: store.ServiceTypeClusterSetIP, IPFamilies: ipv4()},
		remoteWins: func(t *testing.T, s store.ExportSpec) {
			assert.Empty(t, s.InternalTrafficPolicy)
		},
		localWins: func(t *testing.T, s store.ExportSpec) {
			assert.Equal(t, store.InternalTrafficPolicyLocal, s.InternalTrafficPolicy)
		},
		reason: api.ConditionReasonInternalTrafficPolicyConflict,
	},
	{
		name:   "conflict-traffic-distribution",
		local:  store.ExportSpec{Type: store.ServiceTypeClusterSetIP, IPFamilies: ipv4(), TrafficDistribution: "PreferClose"},
		remote: store.ExportSpec{Type: store.ServiceTypeClusterSetIP, IPFamilies: ipv4()},
		remoteWins: func(t *testing.T, s store.ExportSpec) {
			assert.Empty(t, s.TrafficDistribution)
		},
		localWins: func(t *testing.T, s store.ExportSpec) {
			assert.Equal(t, "PreferClose", s.TrafficDistribution)
		},
		reason: api.ConditionReasonTrafficDistributionConflict,
	},
}

func export(cluster string, created time.Time, spec store.ExportSpec) Export {
	spec.ExportCreationTimestamp = created
	return Export{Cluster: cluster, LeaseActive: true, Spec: spec}
}

func condition(t *testing.T, cr ClusterResult, typ string) Condition {
	t.Helper()
	for _, c := range cr.Conditions {
		if c.Type == typ {
			return c
		}
	}
	t.Fatalf("cluster %s has no %s condition", cr.Cluster, typ)
	return Condition{}
}

func cluster(t *testing.T, res Result, name string) ClusterResult {
	t.Helper()
	for _, c := range res.Clusters {
		if c.Cluster == name {
			return c
		}
	}
	t.Fatalf("no cluster %s in result", name)
	return ClusterResult{}
}

// assertConflict checks the import conflict and the per-cluster conditions
// as KEP-1645 asks: Valid, Ready and Conflict true with the expected reason,
// on every participating cluster.
func assertConflict(t *testing.T, res Result, winner, loser, reason, msgIncludes string) {
	t.Helper()
	require.NotNil(t, res.Conflict)
	assert.Equal(t, reason, res.Conflict.Reason)
	assert.Equal(t, winner, res.Conflict.WinningCluster)
	assert.Equal(t, winner, res.Winner)
	if msgIncludes != "" {
		assert.Contains(t, res.Conflict.Message, msgIncludes)
	}
	for _, name := range []string{localCluster, remoteCluster} {
		cr := cluster(t, res, name)
		assert.True(t, cr.Participating)
		assert.Equal(t, ConditionTrue, condition(t, cr, api.ConditionValid).Status)
		assert.Equal(t, ConditionTrue, condition(t, cr, api.ConditionReady).Status)
		c := condition(t, cr, api.ConditionConflict)
		assert.Equal(t, ConditionTrue, c.Status)
		assert.Equal(t, reason, c.Reason)
	}
	assert.Nil(t, cluster(t, res, winner).Conflict, "the winner never conflicts with itself")
	lc := cluster(t, res, loser).Conflict
	require.NotNil(t, lc, "the losing cluster disagrees with the winner")
	assert.Equal(t, winner, lc.WinningCluster)
}

func TestConflictRemoteOlder(t *testing.T) {
	for _, tc := range conflictCases {
		t.Run(tc.name, func(t *testing.T) {
			res := Merge([]Export{export(localCluster, nowTime, tc.local), export(remoteCluster, olderTime, tc.remote)})
			tc.remoteWins(t, res.Spec)
			assertConflict(t, res, remoteCluster, localCluster, tc.reason, tc.msgIncludes)
		})
	}
}

func TestConflictLocalPrecedence(t *testing.T) {
	for _, tc := range conflictCases {
		t.Run(tc.name, func(t *testing.T) {
			res := Merge([]Export{export(localCluster, nowTime, tc.local), export(remoteCluster, newerTime, tc.remote)})
			tc.localWins(t, res.Spec)
			assertConflict(t, res, localCluster, remoteCluster, tc.reason, tc.msgIncludes)
		})
	}
}

// Identical exports merge without a conflict and union their ports.
func TestBasic(t *testing.T) {
	spec := store.ExportSpec{
		Type:            store.ServiceTypeClusterSetIP,
		SessionAffinity: store.SessionAffinityNone,
		IPFamilies:      ipv4(),
		Annotations:     map[string]string{"example.com/sync-endpoint-slices": "true"},
		Labels:          map[string]string{"my-label": "test"},
		Ports:           []store.ServicePort{{Name: "named", Port: 80}, {Port: 4242}},
	}
	local := spec.Clone()
	local.Ports = []store.ServicePort{{Port: 4242}, {Name: "named", Port: 80}}
	res := Merge([]Export{export(localCluster, nowTime, local), export(remoteCluster, olderTime, spec)})

	assert.Nil(t, res.Conflict)
	assert.Equal(t, remoteCluster, res.Winner)
	assert.ElementsMatch(t, []store.ServicePort{{Port: 4242}, {Name: "named", Port: 80}}, res.Spec.Ports)
	assert.Equal(t, spec.Labels, res.Spec.Labels)
	assert.Equal(t, spec.Annotations, res.Spec.Annotations)
	require.Len(t, res.Clusters, 2)
	assert.Equal(t, localCluster, res.Clusters[0].Cluster)
	for _, cr := range res.Clusters {
		assert.Nil(t, cr.Conflict)
		c := condition(t, cr, api.ConditionConflict)
		assert.Equal(t, ConditionFalse, c.Status)
		assert.Equal(t, api.ConditionReasonNoConflicts, c.Reason)
		assert.Equal(t, api.ConditionReasonReady, condition(t, cr, api.ConditionReady).Reason)
	}
}

// Once the newer export agrees, the conflict is gone.
func TestConflictRemoval(t *testing.T) {
	local := store.ExportSpec{Type: store.ServiceTypeClusterSetIP, SessionAffinity: store.SessionAffinityNone, IPFamilies: ipv4()}
	remote := local.Clone()
	remote.Type = store.ServiceTypeHeadless
	res := Merge([]Export{export(localCluster, olderTime, local), export(remoteCluster, nowTime, remote)})
	require.NotNil(t, res.Conflict)
	assert.Equal(t, store.ServiceTypeClusterSetIP, res.Spec.Type)

	res = Merge([]Export{export(localCluster, olderTime, local), export(remoteCluster, nowTime, local)})
	assert.Nil(t, res.Conflict)
	assert.Equal(t, store.ServiceTypeClusterSetIP, res.Spec.Type)
}

func TestConflictCountsDisagreeingClusters(t *testing.T) {
	base := store.ExportSpec{Type: store.ServiceTypeClusterSetIP, IPFamilies: ipv4()}
	headless := base.Clone()
	headless.Type = store.ServiceTypeHeadless
	res := Merge([]Export{
		export("a", olderTime, base),
		export("b", nowTime, headless),
		export("c", nowTime.Add(time.Second), base),
		export("d", newerTime, headless),
	})
	require.NotNil(t, res.Conflict)
	assert.Equal(t, "a", res.Winner)
	assert.Contains(t, res.Conflict.Message, "2/4 clusters disagree")
	assert.Contains(t, res.Conflict.Message, `Using "ClusterSetIP"`)
	assert.Nil(t, cluster(t, res, "c").Conflict)
	assert.NotNil(t, cluster(t, res, "b").Conflict)
	assert.NotNil(t, cluster(t, res, "d").Conflict)
	assert.Equal(t, api.ConditionReasonTypeConflict, cluster(t, res, "d").Conflict.Reason)
	assert.Contains(t, cluster(t, res, "d").Conflict.Message, `Cluster "d" has "Headless"`)
}

// An expired cluster does not take part: its old export cannot pin the
// winner, and it gets Ready=False and Conflict=Unknown.
func TestExpiredClusterExcluded(t *testing.T) {
	base := store.ExportSpec{Type: store.ServiceTypeClusterSetIP, IPFamilies: ipv4(), Ports: []store.ServicePort{{Name: "http", Port: 80}}}
	old := base.Clone()
	old.Type = store.ServiceTypeHeadless
	old.Ports = []store.ServicePort{{Name: "grpc", Port: 9090}}
	expired := export("a", olderTime, old)
	expired.LeaseActive = false

	res := Merge([]Export{expired, export("b", nowTime, base), export("c", newerTime, base)})
	assert.Equal(t, "b", res.Winner)
	assert.Nil(t, res.Conflict)
	assert.Equal(t, store.ServiceTypeClusterSetIP, res.Spec.Type)
	assert.Equal(t, base.Ports, res.Spec.Ports)

	a := cluster(t, res, "a")
	assert.False(t, a.Participating)
	assert.Nil(t, a.Conflict)
	ready := condition(t, a, api.ConditionReady)
	assert.Equal(t, ConditionFalse, ready.Status)
	assert.Equal(t, api.ConditionReasonClusterLeaseExpired, ready.Reason)
	conflict := condition(t, a, api.ConditionConflict)
	assert.Equal(t, ConditionUnknown, conflict.Status)
	assert.Equal(t, api.ConditionReasonClusterLeaseExpired, conflict.Reason)
	assert.Equal(t, ConditionTrue, condition(t, a, api.ConditionValid).Status)
	assert.True(t, cluster(t, res, "b").Participating)
}

// With every cluster expired the shape is kept from all exports, nothing
// participates and no conflict is reported.
func TestNoParticipants(t *testing.T) {
	a := export("a", olderTime, store.ExportSpec{Type: store.ServiceTypeHeadless, Ports: []store.ServicePort{{Name: "a", Port: 80}}})
	b := export("b", nowTime, store.ExportSpec{Type: store.ServiceTypeClusterSetIP, Ports: []store.ServicePort{{Name: "b", Port: 81}}})
	a.LeaseActive, b.LeaseActive = false, false

	res := Merge([]Export{b, a})
	assert.Empty(t, res.Winner)
	assert.Nil(t, res.Conflict)
	assert.Equal(t, store.ServiceTypeHeadless, res.Spec.Type)
	assert.Equal(t, []store.ServicePort{{Name: "a", Port: 80}, {Name: "b", Port: 81}}, res.Spec.Ports)
	for _, cr := range res.Clusters {
		assert.False(t, cr.Participating)
		assert.Nil(t, cr.Conflict)
	}
}

// Equal export creation timestamps are ordered by cluster name.
func TestTimestampTieBrokenByClusterName(t *testing.T) {
	headless := store.ExportSpec{Type: store.ServiceTypeHeadless}
	setIP := store.ExportSpec{Type: store.ServiceTypeClusterSetIP}
	res := Merge([]Export{export("zeta", nowTime, setIP), export("alpha", nowTime, headless)})
	assert.Equal(t, "alpha", res.Winner)
	assert.Equal(t, store.ServiceTypeHeadless, res.Spec.Type)
	require.NotNil(t, res.Conflict)
	assert.Equal(t, "alpha", res.Conflict.WinningCluster)

	ordered := Participants([]Export{export("b", nowTime, setIP), export("a", nowTime, setIP), export("c", olderTime, setIP)})
	assert.Equal(t, []string{"c", "a", "b"}, []string{ordered[0].Cluster, ordered[1].Cluster, ordered[2].Cluster})
}

func TestEmpty(t *testing.T) {
	assert.Equal(t, Result{}, Merge(nil))
	reason, msg := CheckConflictExport(nil)
	assert.Equal(t, api.ConditionReasonNoConflicts, reason)
	assert.Empty(t, msg)
	families, reason, _ := IntersectIPFamilies(nil)
	assert.Nil(t, families)
	assert.Equal(t, api.ConditionReasonNoConflicts, reason)
}

func TestDefaultsCompareEqual(t *testing.T) {
	explicit := store.ExportSpec{
		Type:                  store.ServiceTypeClusterSetIP,
		SessionAffinity:       store.SessionAffinityNone,
		InternalTrafficPolicy: store.InternalTrafficPolicyCluster,
		Ports:                 []store.ServicePort{{Port: 80, Protocol: store.ProtocolTCP}},
	}
	implicit := store.ExportSpec{Ports: []store.ServicePort{{Port: 80}}}
	res := Merge([]Export{export("a", olderTime, explicit), export("b", nowTime, implicit)})
	assert.Nil(t, res.Conflict)
	assert.Nil(t, cluster(t, res, "b").Conflict)
}

func TestSessionAffinityConfigPresence(t *testing.T) {
	with := store.ExportSpec{SessionAffinity: store.SessionAffinityClientIP, SessionAffinityConfig: clientIP(0)}
	without := store.ExportSpec{SessionAffinity: store.SessionAffinityClientIP, SessionAffinityConfig: &store.SessionAffinityConfig{}}
	reason, msg := CheckConflictExport([]Export{export("a", olderTime, without), export("b", nowTime, with)})
	assert.Equal(t, api.ConditionReasonSessionAffinityConfigConflict, reason)
	assert.Contains(t, msg, `Using ""`)
}

func TestPerClusterIPFamilyConflict(t *testing.T) {
	res := Merge([]Export{
		export("a", olderTime, store.ExportSpec{IPFamilies: ipv4()}),
		export("b", nowTime, store.ExportSpec{IPFamilies: []store.IPFamily{store.IPFamilyIPv6}}),
	})
	require.NotNil(t, res.Conflict)
	assert.Equal(t, api.ConditionReasonIPFamilyConflict, res.Conflict.Reason)
	require.NotNil(t, cluster(t, res, "b").Conflict)
	assert.Equal(t, api.ConditionReasonIPFamilyConflict, cluster(t, res, "b").Conflict.Reason)
	assert.Equal(t, ipv4(), res.Spec.IPFamilies)
}

func TestPortsWithProtocols(t *testing.T) {
	ports, reason, _ := MergePorts([]Export{
		export("a", olderTime, store.ExportSpec{Ports: []store.ServicePort{{Name: "dns", Port: 53, Protocol: store.ProtocolUDP}}}),
		export("b", nowTime, store.ExportSpec{Ports: []store.ServicePort{{Name: "dns-tcp", Port: 53, Protocol: store.ProtocolTCP}}}),
	})
	assert.Len(t, ports, 2)
	assert.Equal(t, api.ConditionReasonPortConflict, reason, "each export misses the other's port")
}

// Unnamed ports count as the same name.
func TestUnnamedPortsDuplicate(t *testing.T) {
	ports, reason, msg := MergePorts([]Export{
		export("a", olderTime, store.ExportSpec{Ports: []store.ServicePort{{Port: 80}}}),
		export("b", nowTime, store.ExportSpec{Ports: []store.ServicePort{{Port: 81}}}),
	})
	assert.Equal(t, []store.ServicePort{{Port: 80}}, ports)
	assert.Equal(t, api.ConditionReasonPortConflict, reason)
	assert.Contains(t, msg, `Duplicated port name ""`)
}

// A union that mixes an unnamed port with named ones is not a valid Service:
// the oldest export's ports are kept whole and the mixing cluster conflicts.
func TestMixedUnnamedAndNamedPortsKeepTheOldest(t *testing.T) {
	a := export("a", olderTime, store.ExportSpec{Ports: []store.ServicePort{{Port: 80}}})
	b := export("b", nowTime, store.ExportSpec{Ports: []store.ServicePort{{Name: "http", Port: 8080}, {Name: "metrics", Port: 9090}}})
	ports, reason, msg := MergePorts([]Export{a, b})
	assert.Equal(t, []store.ServicePort{{Port: 80}}, ports)
	assert.Equal(t, api.ConditionReasonPortConflict, reason)
	assert.Contains(t, msg, `from cluster "b" cannot be merged`)
	assert.Contains(t, msg, "needs every port named")

	// The other way round the named ports of the oldest export win.
	ports, reason, msg = MergePorts([]Export{export("a", olderTime, b.Spec), export("b", nowTime, a.Spec)})
	assert.Equal(t, b.Spec.Ports, ports)
	assert.Equal(t, api.ConditionReasonPortConflict, reason)
	assert.Contains(t, msg, `from cluster "b" cannot be merged`)

	res := Merge([]Export{a, b})
	assert.Equal(t, []store.ServicePort{{Port: 80}}, res.Spec.Ports)
	require.NotNil(t, res.Conflict)
	assert.Equal(t, api.ConditionReasonPortConflict, res.Conflict.Reason)
	assert.Nil(t, cluster(t, res, "a").Conflict)
	require.NotNil(t, cluster(t, res, "b").Conflict)
	assert.Equal(t, api.ConditionReasonPortConflict, cluster(t, res, "b").Conflict.Reason)
	assert.True(t, SimilarPort(store.ServicePort{Port: 80}, store.ServicePort{Port: 80, Protocol: store.ProtocolTCP}))
}

func newIPFamilyExport(families ...store.IPFamily) Export {
	return export("c1", nowTime, store.ExportSpec{IPFamilies: families})
}

// IP families are intersected in priority order.
func TestIntersectIPFamilies(t *testing.T) {
	v4, v6 := store.IPFamilyIPv4, store.IPFamilyIPv6
	tests := []struct {
		name     string
		exports  []Export
		families []store.IPFamily
		reason   string
	}{
		{"all dual-stack", []Export{newIPFamilyExport(v4, v6), newIPFamilyExport(v6, v4)}, []store.IPFamily{v4, v6}, api.ConditionReasonNoConflicts},
		{"dual-stack then narrows to IPv4", []Export{newIPFamilyExport(v4, v6), newIPFamilyExport(v4)}, []store.IPFamily{v4}, api.ConditionReasonNoConflicts},
		{"keep oldest single stack IPv6", []Export{newIPFamilyExport(v6), newIPFamilyExport(v4, v6)}, []store.IPFamily{v6}, api.ConditionReasonNoConflicts},
		{"simple conflict", []Export{newIPFamilyExport(v4), newIPFamilyExport(v6)}, []store.IPFamily{v4}, api.ConditionReasonIPFamilyConflict},
		{"no families", []Export{newIPFamilyExport(), newIPFamilyExport()}, nil, api.ConditionReasonNoConflicts},
		{"unknown families do not constrain", []Export{newIPFamilyExport(), newIPFamilyExport(v6), newIPFamilyExport()}, []store.IPFamily{v6}, api.ConditionReasonNoConflicts},
		{
			"dual-stack then narrow to IPv4 and conflict with IPv6",
			[]Export{newIPFamilyExport(v4, v6), newIPFamilyExport(v4), newIPFamilyExport(v6)},
			[]store.IPFamily{v4}, api.ConditionReasonIPFamilyConflict,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			families, reason, _ := IntersectIPFamilies(tc.exports)
			assert.Equal(t, tc.families, families)
			assert.Equal(t, tc.reason, reason)
		})
	}
}
