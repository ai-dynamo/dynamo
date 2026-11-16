/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"testing"
)

func TestNewModelExpressClientAcceptsModelExpressURLs(t *testing.T) {
	t.Log("Define accepted ModelExpress endpoint forms")
	tests := []string{
		"model-express:8080",
		"http://model-express:8080",
		"http://model-express:8080/",
		"https://model-express.example.com:443",
	}

	for _, modelExpressURL := range tests {
		t.Run(modelExpressURL, func(t *testing.T) {
			t.Logf("Create a ModelExpress client for %q", modelExpressURL)
			client, err := NewModelExpressClient(modelExpressURL)

			t.Log("Verify the endpoint produces a usable client")
			if err != nil {
				t.Fatalf("NewModelExpressClient() error = %v", err)
			}
			if client == nil {
				t.Fatal("NewModelExpressClient() = nil, want client")
			}
		})
	}
}

func TestNewModelExpressClientRejectsInvalidModelExpressURL(t *testing.T) {
	t.Log("Define malformed or unsupported ModelExpress endpoint forms")
	tests := []string{
		"://bad-url",
		"http:model-express:8080",
		"http://model-express:8080/mx",
		"http://model-express:8080?version=1",
		"http://model-express:8080#mx",
		"https://model-express.example.com:443/mx",
	}

	for _, modelExpressURL := range tests {
		t.Run(modelExpressURL, func(t *testing.T) {
			t.Logf("Attempt to create a ModelExpress client for %q", modelExpressURL)
			client, err := NewModelExpressClient(modelExpressURL)

			t.Log("Verify the invalid endpoint is rejected without a client")
			if err == nil {
				t.Fatal("NewModelExpressClient() error = nil, want error")
			}
			if client != nil {
				t.Fatalf("NewModelExpressClient() client = %#v, want nil", client)
			}
		})
	}
}
