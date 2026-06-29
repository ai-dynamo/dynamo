/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package sglang

import (
	"encoding/json"
	"fmt"
	"net"
	"net/url"
	"strconv"
	"strings"
)

func kvEventsEndpoints(serverInfo ServerInfo, workerEndpoint string, blockSize uint32, dpSize uint32, require bool) (map[uint32]string, error) {
	if serverInfo.KVEvents != nil {
		return kvEventsDescriptorEndpoints(*serverInfo.KVEvents, workerEndpoint, blockSize, dpSize)
	}
	if len(serverInfo.KVEventsConfig) > 0 && string(serverInfo.KVEventsConfig) != "null" {
		return kvEventsConfigEndpoints(serverInfo.KVEventsConfig, workerEndpoint, dpSize)
	}
	if require {
		return nil, fmt.Errorf("SGLang KV events metadata is required")
	}
	return nil, nil
}

func kvEventsDescriptorEndpoints(descriptor KVEventsDescriptor, workerEndpoint string, blockSize uint32, dpSize uint32) (map[uint32]string, error) {
	if descriptor.Publisher != "zmq" {
		return nil, fmt.Errorf("SGLang kv_events publisher must be 'zmq'")
	}
	descriptorBlockSize, err := requiredPositiveUint32("block_size", descriptor.BlockSize)
	if err != nil {
		return nil, err
	}
	if descriptorBlockSize != blockSize {
		return nil, fmt.Errorf("SGLang kv_events block_size does not match server page_size")
	}
	descriptorDPSize, err := requiredPositiveUint32("dp_size", descriptor.DPSize)
	if err != nil {
		return nil, err
	}
	if descriptorDPSize != dpSize {
		return nil, fmt.Errorf("SGLang kv_events dp_size does not match server dp_size")
	}
	portBase, err := requiredPositiveUint32("endpoint_port_base", descriptor.EndpointPortBase)
	if err != nil {
		return nil, err
	}
	lastPort := uint64(portBase) + uint64(dpSize) - 1
	if portBase > 65535 || lastPort > 65535 {
		return nil, fmt.Errorf("SGLang kv_events endpoint port is out of range")
	}
	if descriptor.EndpointHost == "" {
		return nil, fmt.Errorf("SGLang kv_events endpoint_host is required")
	}
	host, err := subscriberHost(descriptor.EndpointHost, workerEndpoint)
	if err != nil {
		return nil, err
	}

	endpoints := make(map[uint32]string, dpSize)
	for dpRank := uint32(0); dpRank < dpSize; dpRank++ {
		endpoints[dpRank] = "tcp://" + net.JoinHostPort(host, strconv.Itoa(int(portBase+dpRank)))
	}
	return endpoints, nil
}

func kvEventsConfigEndpoints(descriptor json.RawMessage, workerEndpoint string, dpSize uint32) (map[uint32]string, error) {
	var payload KVEventsConfig
	var encoded string
	if err := json.Unmarshal(descriptor, &encoded); err == nil {
		if strings.TrimSpace(encoded) == "" {
			return nil, fmt.Errorf("SGLang kv_events_config metadata is empty")
		}
		if err := json.Unmarshal([]byte(encoded), &payload); err != nil {
			return nil, fmt.Errorf("decode SGLang kv_events_config metadata JSON: %w", err)
		}
	} else if err := json.Unmarshal(descriptor, &payload); err != nil {
		return nil, fmt.Errorf("SGLang kv_events_config metadata must be an object or JSON string")
	}

	if payload.Publisher != "zmq" {
		return nil, fmt.Errorf("SGLang kv_events_config publisher must be 'zmq'")
	}
	if payload.Endpoint == "" {
		return nil, fmt.Errorf("SGLang kv_events_config endpoint is required")
	}
	if dpSize != 1 {
		return nil, fmt.Errorf("SGLang kv_events_config endpoint currently supports data_parallel_size=1")
	}
	parsed, err := url.Parse(payload.Endpoint)
	if err != nil {
		return nil, fmt.Errorf("parse SGLang kv_events_config endpoint %q: %w", payload.Endpoint, err)
	}
	if parsed.Scheme != "tcp" {
		return nil, fmt.Errorf("SGLang kv_events_config endpoint must use tcp")
	}
	port := parsed.Port()
	if port == "" {
		return nil, fmt.Errorf("SGLang kv_events_config endpoint must include a port")
	}
	host, err := subscriberHost(parsed.Hostname(), workerEndpoint)
	if err != nil {
		return nil, err
	}
	return map[uint32]string{
		0: "tcp://" + net.JoinHostPort(host, port),
	}, nil
}

func subscriberHost(endpointHost string, workerEndpoint string) (string, error) {
	if isWildcardHost(endpointHost) {
		parsed, err := url.Parse(workerEndpoint)
		if err != nil {
			return "", fmt.Errorf("parse worker endpoint %q: %w", workerEndpoint, err)
		}
		host := parsed.Hostname()
		if host == "" {
			return "", fmt.Errorf("worker endpoint must include a host")
		}
		return host, nil
	}
	return strings.TrimSuffix(strings.TrimPrefix(endpointHost, "["), "]"), nil
}

func isWildcardHost(host string) bool {
	switch host {
	case "*", "0.0.0.0", "::", "[::]":
		return true
	default:
		return false
	}
}
