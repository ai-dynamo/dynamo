/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package poolrelay

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"time"

	v1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

const relaySourcesStatusPort = "8081"

type relaySourcesAcknowledgement struct {
	Mode               string `json:"mode"`
	PodUID             string `json:"podUID"`
	ConnectionRevision string `json:"connectionRevision"`
	Sources            struct {
		DesiredRevision string  `json:"desiredRevision"`
		AppliedRevision string  `json:"appliedRevision"`
		LastError       *string `json:"lastError"`
	} `json:"sources"`
}

// readRelaySourcesAcknowledgement bounds both response time and size. Redirects are
// rejected so a Pod cannot redirect the operator to an unrelated endpoint.
func readRelaySourcesAcknowledgement(ctx context.Context, address, podUID, connectionRevision string) (relaySourcesAcknowledgement, error) {
	var ack relaySourcesAcknowledgement
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, "http://"+address+"/engine/state", nil)
	if err != nil {
		return ack, err
	}
	transport := &http.Transport{Proxy: nil, DisableKeepAlives: true}
	defer transport.CloseIdleConnections()
	client := &http.Client{Timeout: 3 * time.Second, Transport: transport, CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }}
	response, err := client.Do(request)
	if err != nil {
		return ack, fmt.Errorf("sources acknowledgement endpoint unavailable")
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		return ack, fmt.Errorf("sources acknowledgement endpoint is not ready")
	}
	data, err := io.ReadAll(io.LimitReader(response.Body, 16385))
	if err != nil || len(data) > 16384 {
		return ack, fmt.Errorf("sources acknowledgement response exceeds limit or is unreadable")
	}
	if err := json.Unmarshal(data, &ack); err != nil {
		return ack, fmt.Errorf("invalid sources acknowledgement")
	}
	if ack.Mode != "from-file" || ack.PodUID != podUID || ack.ConnectionRevision != connectionRevision {
		return ack, fmt.Errorf("sources acknowledgement belongs to another Pod or connection")
	}
	if len(ack.Sources.AppliedRevision) > 64 || len(ack.Sources.DesiredRevision) > 64 {
		return ack, fmt.Errorf("invalid sources revision length")
	}
	return ack, nil
}

// observeRelaySources requires acknowledgements from every requested current replica.
// relay and deployment must be non-nil, and deployment.Spec.Replicas must be set.
// Missing Pods, a pending rollout, or unavailable HTTP endpoints remain pending.
func (r *relayReconciler) observeRelaySources(ctx context.Context, relay *v1alpha1.DynamoPoolRelay, deployment *appsv1.Deployment, sources relaySources) (metav1.Condition, error) {
	condition := metav1.Condition{Type: "SourcesApplied", Status: metav1.ConditionFalse, Reason: "WaitingForAcknowledgement", ObservedGeneration: relay.Generation}
	relay.Status.AppliedSourcesRevision = ""
	relay.Status.SourcesAppliedReplicas = 0
	desiredReplicas := *deployment.Spec.Replicas
	condition.Message = fmt.Sprintf("Waiting for all %d current Relay replicas to apply their source set", desiredReplicas)

	// Count all owned non-terminating Pods so old or surge replicas cannot hide a partial rollout.
	var pods corev1.PodList
	if err := r.List(ctx, &pods, client.InNamespace(relay.Namespace), client.MatchingLabels{"nvidia.com/relay-uid": string(relay.UID)}); err != nil {
		return condition, err
	}
	activeReplicas := int32(0)
	for _, pod := range pods.Items {
		if !pod.DeletionTimestamp.IsZero() {
			continue
		}
		owner := metav1.GetControllerOf(&pod)
		if owner == nil || owner.Kind != "ReplicaSet" {
			continue
		}
		var replicaSet appsv1.ReplicaSet
		if err := r.Get(ctx, client.ObjectKey{Namespace: relay.Namespace, Name: owner.Name}, &replicaSet); err != nil {
			continue
		}
		if owner.UID != replicaSet.UID || !metav1.IsControlledBy(&replicaSet, deployment) {
			continue
		}
		activeReplicas++

		// Ignore Kubernetes' ReplicaSet hash label when identifying the desired Pod template.
		current := replicaSet.Spec.Template.DeepCopy()
		delete(current.Labels, appsv1.DefaultDeploymentUniqueLabelKey)
		if !equality.Semantic.DeepEqual(*current, deployment.Spec.Template) || pod.Status.Phase != corev1.PodRunning || net.ParseIP(pod.Status.PodIP) == nil {
			continue
		}
		ack, err := readRelaySourcesAcknowledgement(ctx, net.JoinHostPort(pod.Status.PodIP, relaySourcesStatusPort), string(pod.UID), sources.ConnectionRevision)
		if err != nil {
			continue
		}
		if ack.Sources.LastError != nil {
			condition.Reason = "ReloadFailed"
			condition.Message = "A current Relay replica rejected its source set; not all replicas have applied the desired revision"
		} else if ack.Sources.AppliedRevision == sources.Revision && ack.Sources.DesiredRevision == sources.Revision {
			relay.Status.SourcesAppliedReplicas++
		}
	}

	// A single successful replica never acknowledges the complete active-active deployment.
	if activeReplicas == desiredReplicas && relay.Status.SourcesAppliedReplicas == desiredReplicas {
		relay.Status.AppliedSourcesRevision = sources.Revision
		condition.Status = metav1.ConditionTrue
		condition.Reason = "Applied"
		condition.Message = fmt.Sprintf("All %d current Relay replicas have applied their source set; KV recovery readiness is separate", desiredReplicas)
	}
	return condition, nil
}
