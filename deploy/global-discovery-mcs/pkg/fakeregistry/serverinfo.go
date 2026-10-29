// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package fakeregistry

import (
	"context"

	mcsv1 "github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/api/gen/go/dynamo/mcs/v1"
)

type serverInfoService struct {
	mcsv1.UnimplementedServerInfoServiceServer
	s *Server
}

func (si *serverInfoService) GetServerInfo(context.Context, *mcsv1.GetServerInfoRequest) (*mcsv1.GetServerInfoResponse, error) {
	return si.s.serverInfo(), nil
}
