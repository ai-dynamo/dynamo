/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

package observability

import (
	"time"

	"github.com/prometheus/client_golang/prometheus"
	k8serrors "k8s.io/apimachinery/pkg/api/errors"
	ctrlmetrics "sigs.k8s.io/controller-runtime/pkg/metrics"
)

const (
	metricsNamespace = "dynamo_operator"
)

var (
	// Reconciliation metrics
	reconcileDuration = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Namespace: metricsNamespace,
			Name:      "reconcile_duration_seconds",
			Help:      "Duration of reconciliation loops in seconds",
			Buckets:   []float64{.005, .01, .025, .05, .1, .25, .5, 1, 2.5, 5, 10, 30, 60},
		},
		[]string{"resource_type", "namespace", "result"},
	)

	reconcileTotal = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Namespace: metricsNamespace,
			Name:      "reconcile_total",
			Help:      "Total number of reconciliations by resource type",
		},
		[]string{"resource_type", "namespace", "result"},
	)

	reconcileErrors = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Namespace: metricsNamespace,
			Name:      "reconcile_errors_total",
			Help:      "Total number of reconciliation errors by resource type and error type",
		},
		[]string{"resource_type", "namespace", "error_type"},
	)

	// Resource metrics (populated by resource counter)
	resourcesTotal = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Namespace: metricsNamespace,
			Name:      "resources_total",
			Help:      "Total number of resources by type and status",
		},
		[]string{"resource_type", "namespace", "status"},
	)

	// Webhook metrics
	webhookDuration = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Namespace: metricsNamespace,
			Name:      "webhook_duration_seconds",
			Help:      "Duration of webhook validation in seconds",
			Buckets:   []float64{.001, .005, .01, .025, .05, .1, .25, .5, 1, 2.5, 5},
		},
		[]string{"resource_type", "operation"},
	)

	webhookRequestsTotal = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Namespace: metricsNamespace,
			Name:      "webhook_requests_total",
			Help:      "Total number of webhook admission requests",
		},
		[]string{"resource_type", "operation", "result"},
	)

	webhookDenialsTotal = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Namespace: metricsNamespace,
			Name:      "webhook_denials_total",
			Help:      "Total number of webhook admission denials",
		},
		[]string{"resource_type", "operation", "reason"},
	)
)

// InitMetrics registers all custom metrics with the controller-runtime metrics registry
func InitMetrics() {
	ctrlmetrics.Registry.MustRegister(
		reconcileDuration,
		reconcileTotal,
		reconcileErrors,
		resourcesTotal,
		webhookDuration,
		webhookRequestsTotal,
		webhookDenialsTotal,
	)
}

// RecordReconciliation records metrics for a reconciliation loop
func RecordReconciliation(resourceType, namespace string, err error, requeue bool, duration time.Duration) {
	result := "success"
	if err != nil {
		result = "error"
		errorType := categorizeError(err)
		reconcileErrors.WithLabelValues(resourceType, namespace, errorType).Inc()
	} else if requeue {
		result = "requeue"
	}

	reconcileDuration.WithLabelValues(resourceType, namespace, result).Observe(duration.Seconds())
	reconcileTotal.WithLabelValues(resourceType, namespace, result).Inc()
}

// RecordWebhookAdmission records metrics for a webhook admission request
func RecordWebhookAdmission(resourceType, operation string, allowed bool, duration time.Duration) {
	webhookDuration.WithLabelValues(resourceType, operation).Observe(duration.Seconds())

	result := "allowed"
	if !allowed {
		result = "denied"
	}
	webhookRequestsTotal.WithLabelValues(resourceType, operation, result).Inc()
}

// RecordWebhookDenial records a webhook denial with a categorized reason
func RecordWebhookDenial(resourceType, operation string, err error) {
	reason := categorizeError(err)
	webhookDenialsTotal.WithLabelValues(resourceType, operation, reason).Inc()
}

// UpdateResourceCount updates the gauge for a specific resource type and status
func UpdateResourceCount(resourceType, namespace, status string, count float64) {
	resourcesTotal.WithLabelValues(resourceType, namespace, status).Set(count)
}

// categorizeError categorizes Kubernetes errors for better metrics granularity
func categorizeError(err error) string {
	if err == nil {
		return "none"
	}

	switch {
	case k8serrors.IsNotFound(err):
		return "not_found"
	case k8serrors.IsAlreadyExists(err):
		return "already_exists"
	case k8serrors.IsConflict(err):
		return "conflict"
	case k8serrors.IsInvalid(err):
		return "validation"
	case k8serrors.IsBadRequest(err):
		return "bad_request"
	case k8serrors.IsUnauthorized(err):
		return "unauthorized"
	case k8serrors.IsForbidden(err):
		return "forbidden"
	case k8serrors.IsTimeout(err):
		return "timeout"
	case k8serrors.IsServerTimeout(err):
		return "server_timeout"
	case k8serrors.IsServiceUnavailable(err):
		return "unavailable"
	case k8serrors.IsTooManyRequests(err):
		return "rate_limited"
	default:
		return "internal"
	}
}
