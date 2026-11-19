// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package leaderworkerset

import (
	"crypto/sha256"
	"encoding/hex"
	"maps"
	"strings"

	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	leaderworkersetv1 "sigs.k8s.io/lws/api/leaderworkerset/v1"
)

// Render constructs the desired LeaderWorkerSet from fully resolved Pod templates.
// dcd and both templates must be non-nil. Inputs are not modified.
func Render(dcd *v1beta1.DynamoComponentDeployment, selectedName string, metadata metav1.ObjectMeta, leaderPodTemplateSpec, workerPodTemplateSpec *corev1.PodTemplateSpec) *leaderworkersetv1.LeaderWorkerSet {
	leaderPodTemplateSpec = RenderPodTemplate(leaderPodTemplateSpec)
	workerPodTemplateSpec = RenderPodTemplate(workerPodTemplateSpec)

	kubeNs := dcd.Namespace
	labels := maps.Clone(metadata.Labels)

	if labels == nil {
		labels = make(map[string]string)
	}

	leaderWorkerSet := &leaderworkersetv1.LeaderWorkerSet{
		Name:      selectedName,
		Namespace: kubeNs,
		Labels:    labels,
	}

	desiredReplicas := int32(1)
	if dcd.Spec.Replicas != nil {
		desiredReplicas = *dcd.Spec.Replicas
	}
	groupSize := dcd.GetNumberOfNodes()

	leaderWorkerSet.Spec = leaderworkersetv1.LeaderWorkerSetSpec{
		Replicas:      &desiredReplicas,
		StartupPolicy: leaderworkersetv1.LeaderCreatedStartupPolicy,
		LeaderWorkerTemplate: leaderworkersetv1.LeaderWorkerTemplate{
			LeaderTemplate: leaderPodTemplateSpec,
			WorkerTemplate: *workerPodTemplateSpec,
			Size:           &groupSize,
		},
	}

	return leaderWorkerSet
}

// RenderPodTemplate removes the Deployment-only selector from an LWS role template.
// template must be non-nil. The input is not modified.
func RenderPodTemplate(template *corev1.PodTemplateSpec) *corev1.PodTemplateSpec {
	template = template.DeepCopy()
	delete(template.Labels, consts.KubeLabelDynamoSelector)
	return template
}

// Name reserves room for StatefulSet revision labels at every supported group index.
// Reconciliation separately preserves existing legacy workloads.
func Name(workloadName string) string {
	legacyName := workloadName + "-0"
	if len(legacyName) <= maxHashedLeaderWorkerSetNameLength {
		return legacyName
	}
	hash := sha256.Sum256([]byte(legacyName))
	suffix := "-" + hex.EncodeToString(hash[:])[:8]
	prefix := strings.TrimRight(legacyName[:maxHashedLeaderWorkerSetNameLength-len(suffix)], "-")
	return prefix + suffix
}

// Pod label controller-revision-hash=<statefulset>-<10-char hash> must fit 63 chars, and LWS
// names its StatefulSets <lws> and <lws>-<groupIndex>, so the LWS name leaves room for both.
const (
	maxLeaderWorkerSetLabelValueLength  = 63
	leaderWorkerSetRevisionHashLength   = 10
	leaderWorkerSetRevisionSuffixLength = 2*len("-") + leaderWorkerSetRevisionHashLength
	maxHashedLeaderWorkerSetGroupDigits = len("999999") // LWS allows up to 1,000,000 replicas.
	maxHashedLeaderWorkerSetNameLength  = maxLeaderWorkerSetLabelValueLength - leaderWorkerSetRevisionSuffixLength - maxHashedLeaderWorkerSetGroupDigits
)
