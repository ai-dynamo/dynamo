# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Build stage
FROM golang:1.24 as builder

# Docker buildx automatically provides these
ARG TARGETOS=linux
ARG TARGETARCH=amd64

RUN echo "Building for ${TARGETOS}/${TARGETARCH}"

RUN apt-get update && apt-get install -y git && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
# Copy go mod and sum files
COPY go.mod go.sum ./
# Download dependencies
RUN go mod download

# Copy the source code
COPY . .

# Build
RUN CGO_ENABLED=0 GOOS=${TARGETOS} GOARCH=${TARGETARCH} go build -o manager ./cmd/main.go

# Runtime stage
FROM nvcr.io/nvidia/distroless/go:v3.1.12
WORKDIR /
COPY --from=builder /workspace/manager .
USER 65532:65532

ENTRYPOINT ["./manager"]
