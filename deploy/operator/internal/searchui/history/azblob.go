/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package history

import (
	"context"
	"errors"
	"fmt"
	"io"

	"github.com/Azure/azure-sdk-for-go/sdk/azidentity"
	"github.com/Azure/azure-sdk-for-go/sdk/storage/azblob"
	"github.com/Azure/azure-sdk-for-go/sdk/storage/azblob/bloberror"
)

// NewAzureBlob stores history in one container of an Azure Storage account. accountURL is
// the blob service endpoint, for example https://myaccount.blob.core.windows.net.
// Credentials come from the environment through DefaultAzureCredential, which on AKS with
// Workload Identity means the federated token mounted into the pod.
func NewAzureBlob(ctx context.Context, accountURL, container string) (Store, error) {
	if accountURL == "" || container == "" {
		return nil, errors.New("azure blob history needs an account URL and a container")
	}
	cred, err := azidentity.NewDefaultAzureCredential(nil)
	if err != nil {
		return nil, fmt.Errorf("azure credential: %w", err)
	}
	client, err := azblob.NewClient(accountURL, cred, nil)
	if err != nil {
		return nil, fmt.Errorf("azure blob client: %w", err)
	}
	store := &kvStore{kv: &azureKV{client: client, container: container}}

	// Fail at startup, not on first write, when the identity cannot reach the container.
	if _, err := store.kv.List(ctx, "runs/"); err != nil {
		return nil, fmt.Errorf("azure blob container %q is not readable: %w", container, err)
	}
	return store, nil
}

type azureKV struct {
	client    *azblob.Client
	container string
}

func (a *azureKV) Put(ctx context.Context, key string, data []byte) error {
	_, err := a.client.UploadBuffer(ctx, a.container, key, data, nil)
	return err
}

func (a *azureKV) Get(ctx context.Context, key string) ([]byte, error) {
	resp, err := a.client.DownloadStream(ctx, a.container, key, nil)
	if err != nil {
		if bloberror.HasCode(err, bloberror.BlobNotFound) {
			return nil, ErrNotFound
		}
		return nil, err
	}
	defer func() { _ = resp.Body.Close() }()
	return io.ReadAll(resp.Body)
}

func (a *azureKV) List(ctx context.Context, prefix string) ([]string, error) {
	var keys []string
	pager := a.client.NewListBlobsFlatPager(a.container, &azblob.ListBlobsFlatOptions{Prefix: &prefix})
	for pager.More() {
		page, err := pager.NextPage(ctx)
		if err != nil {
			return nil, err
		}
		for _, item := range page.Segment.BlobItems {
			if item.Name != nil {
				keys = append(keys, *item.Name)
			}
		}
	}
	return keys, nil
}
