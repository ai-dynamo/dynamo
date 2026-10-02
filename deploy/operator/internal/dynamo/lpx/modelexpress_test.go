/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestNewModelExpressClientValidatesURL(t *testing.T) {
	for _, test := range []struct {
		url     string
		wantErr bool
	}{
		{url: "model-express:8080"},
		{url: "http://model-express:8080"},
		{url: "http://model-express:8080/"},
		{url: "https://model-express.example.com:443"},
		{url: "://bad-url", wantErr: true},
		{url: "http:model-express:8080", wantErr: true},
		{url: "http://model-express:8080/mx", wantErr: true},
		{url: "http://model-express:8080?version=1", wantErr: true},
		{url: "http://model-express:8080#mx", wantErr: true},
		{url: "https://model-express.example.com:443/mx", wantErr: true},
	} {
		t.Run(test.url, func(t *testing.T) {
			t.Log("Create a client only for supported ModelExpress endpoint forms")
			client, err := NewModelExpressClient(test.url)
			if test.wantErr {
				require.Error(t, err)
				require.Nil(t, client)
				return
			}
			require.NoError(t, err)
			require.NotNil(t, client)
		})
	}
}
