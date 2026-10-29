// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package api

import (
	"context"

	mcsv1 "github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/api/gen/go/dynamo/mcs/v1"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/convert"
)

// GetServerInfo implements mcsv1.ServerInfoServiceServer.
func (s *Server) GetServerInfo(context.Context, *mcsv1.GetServerInfoRequest) (*mcsv1.GetServerInfoResponse, error) {
	return &mcsv1.GetServerInfoResponse{
		Region:     s.opts.Region,
		Version:    s.opts.Version,
		ViewEpoch:  s.view.Epoch(),
		Endpoints:  s.opts.Endpoints,
		ServerTime: convert.Timestamp(s.now()),
	}, nil
}
