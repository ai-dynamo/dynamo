/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package controller

import (
	"context"
	"fmt"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/enginemetadata/sglang"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/selectionservice"
)

const selectionAdapterExternalSGLang = "external-sglang"

type sglangMetadataFetcher interface {
	Fetch(context.Context) (sglang.MetadataSnapshot, error)
}

type sglangMetadataClientFactory func(endpoint string) (sglangMetadataFetcher, error)

type externalSGLangSelectionAdapter struct {
	MetadataClientFactory sglangMetadataClientFactory
}

func (a externalSGLangSelectionAdapter) BuildWorker(ctx context.Context, reg selectionWorkerRegistration) (selectionservice.WorkerRequest, error) {
	metadataClientFactory := a.MetadataClientFactory
	if metadataClientFactory == nil {
		metadataClientFactory = func(endpoint string) (sglangMetadataFetcher, error) {
			return sglang.NewMetadataClient(endpoint, selectionSGLangRequestTimeout)
		}
	}
	metadataClient, err := metadataClientFactory(reg.WorkerEndpoint)
	if err != nil {
		return selectionservice.WorkerRequest{}, fmt.Errorf("create SGLang metadata client for %q: %w", reg.WorkerEndpoint, err)
	}
	metadata, err := metadataClient.Fetch(ctx)
	if err != nil {
		return selectionservice.WorkerRequest{}, fmt.Errorf("fetch SGLang metadata from %q: %w", reg.WorkerEndpoint, err)
	}
	worker, err := sglang.BuildWorkerRequest(sglang.WorkerRegistration{
		WorkerID:        reg.WorkerID,
		WorkerEndpoint:  reg.WorkerEndpoint,
		TenantID:        reg.TenantID,
		StableRoutingID: reg.StableRoutingID,
		RequireKVEvents: reg.RequireKVEvents,
		TopologyDomains: reg.TopologyDomains,
	}, metadata)
	if err != nil {
		return selectionservice.WorkerRequest{}, fmt.Errorf("build SGLang worker request for %q: %w", reg.WorkerEndpoint, err)
	}
	return worker, nil
}
