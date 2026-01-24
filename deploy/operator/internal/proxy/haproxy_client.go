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

// Package proxy provides client implementations for communicating with HAProxy
// during rolling updates. The HAProxyClient uses the HAProxy Runtime API to
// dynamically update backend weights without requiring configuration reloads.
package proxy

import (
	"bufio"
	"context"
	"fmt"
	"net"
	"regexp"
	"strconv"
	"strings"
	"time"
)

const (
	// BackendName is the name of the HAProxy backend containing the frontend servers
	BackendName = "frontends"
	// OldServerName is the HAProxy server name for the old frontend
	OldServerName = "old_frontend"
	// NewServerName is the HAProxy server name for the new frontend
	NewServerName = "new_frontend"

	// DefaultTimeout is the default timeout for HAProxy runtime API operations
	DefaultTimeout = 5 * time.Second
	// DefaultDialTimeout is the default timeout for establishing connections
	DefaultDialTimeout = 2 * time.Second
)

// HAProxyClient communicates with HAProxy via the Runtime API
// It can connect either via Unix socket (for local access) or TCP (for remote access)
type HAProxyClient struct {
	// address is either a Unix socket path or TCP address (host:port)
	address string
	// network is either "unix" or "tcp"
	network string
	// timeout for operations
	timeout time.Duration
	// dialTimeout for connection establishment
	dialTimeout time.Duration
}

// ClientOption configures the HAProxyClient
type ClientOption func(*HAProxyClient)

// WithTimeout sets the operation timeout
func WithTimeout(d time.Duration) ClientOption {
	return func(c *HAProxyClient) {
		c.timeout = d
	}
}

// WithDialTimeout sets the connection establishment timeout
func WithDialTimeout(d time.Duration) ClientOption {
	return func(c *HAProxyClient) {
		c.dialTimeout = d
	}
}

// NewHAProxyClientTCP creates a client that connects via TCP to the HAProxy runtime API
func NewHAProxyClientTCP(host string, port int, opts ...ClientOption) *HAProxyClient {
	c := &HAProxyClient{
		address:     fmt.Sprintf("%s:%d", host, port),
		network:     "tcp",
		timeout:     DefaultTimeout,
		dialTimeout: DefaultDialTimeout,
	}
	for _, opt := range opts {
		opt(c)
	}
	return c
}

// NewHAProxyClientUnix creates a client that connects via Unix socket
func NewHAProxyClientUnix(socketPath string, opts ...ClientOption) *HAProxyClient {
	c := &HAProxyClient{
		address:     socketPath,
		network:     "unix",
		timeout:     DefaultTimeout,
		dialTimeout: DefaultDialTimeout,
	}
	for _, opt := range opts {
		opt(c)
	}
	return c
}

// UpdateWeights updates the backend server weights via the HAProxy Runtime API
// oldWeight and newWeight should be values from 0-100
func (c *HAProxyClient) UpdateWeights(ctx context.Context, oldWeight, newWeight int32) error {
	// Validate weights
	if oldWeight < 0 || oldWeight > 100 {
		return fmt.Errorf("invalid oldWeight %d: must be between 0 and 100", oldWeight)
	}
	if newWeight < 0 || newWeight > 100 {
		return fmt.Errorf("invalid newWeight %d: must be between 0 and 100", newWeight)
	}

	// Update old server weight
	if err := c.setServerWeight(ctx, BackendName, OldServerName, oldWeight); err != nil {
		return fmt.Errorf("failed to set old_frontend weight: %w", err)
	}

	// Update new server weight
	if err := c.setServerWeight(ctx, BackendName, NewServerName, newWeight); err != nil {
		return fmt.Errorf("failed to set new_frontend weight: %w", err)
	}

	return nil
}

// GetWeights retrieves the current weights for old and new backends
func (c *HAProxyClient) GetWeights(ctx context.Context) (oldWeight, newWeight int32, err error) {
	oldWeight, err = c.getServerWeight(ctx, BackendName, OldServerName)
	if err != nil {
		return 0, 0, fmt.Errorf("failed to get old_frontend weight: %w", err)
	}

	newWeight, err = c.getServerWeight(ctx, BackendName, NewServerName)
	if err != nil {
		return 0, 0, fmt.Errorf("failed to get new_frontend weight: %w", err)
	}

	return oldWeight, newWeight, nil
}

// setServerWeight sets the weight for a specific server in a backend
func (c *HAProxyClient) setServerWeight(ctx context.Context, backend, server string, weight int32) error {
	// HAProxy Runtime API command: set server <backend>/<server> weight <weight>
	cmd := fmt.Sprintf("set server %s/%s weight %d\n", backend, server, weight)

	resp, err := c.executeCommand(ctx, cmd)
	if err != nil {
		return err
	}

	// Check for error responses
	// Successful commands typically return empty response or just a newline
	resp = strings.TrimSpace(resp)
	if resp != "" && !strings.HasPrefix(resp, "Weight changed") {
		// Check if it's an error
		if strings.Contains(strings.ToLower(resp), "error") ||
			strings.Contains(strings.ToLower(resp), "no such") ||
			strings.Contains(strings.ToLower(resp), "unknown") {
			return fmt.Errorf("HAProxy error: %s", resp)
		}
	}

	return nil
}

// SetServerAddr updates the address and port for a backend server dynamically.
// This allows updating backend server addresses without reloading HAProxy configuration.
func (c *HAProxyClient) SetServerAddr(ctx context.Context, backend, server, addr string, port int32) error {
	// HAProxy Runtime API command: set server <backend>/<server> addr <addr> port <port>
	cmd := fmt.Sprintf("set server %s/%s addr %s port %d\n", backend, server, addr, port)

	resp, err := c.executeCommand(ctx, cmd)
	if err != nil {
		return err
	}

	// Check for error responses
	resp = strings.TrimSpace(resp)
	// Successful response typically contains "IP changed" or is empty
	if resp != "" && !strings.Contains(resp, "changed") && !strings.Contains(resp, "addr") {
		// Check if it's an error
		if strings.Contains(strings.ToLower(resp), "error") ||
			strings.Contains(strings.ToLower(resp), "no such") ||
			strings.Contains(strings.ToLower(resp), "unknown") {
			return fmt.Errorf("HAProxy error: %s", resp)
		}
	}

	return nil
}

// getServerWeight retrieves the current weight for a specific server
func (c *HAProxyClient) getServerWeight(ctx context.Context, backend, server string) (int32, error) {
	// HAProxy Runtime API command: show servers state <backend>
	cmd := fmt.Sprintf("show servers state %s\n", backend)

	resp, err := c.executeCommand(ctx, cmd)
	if err != nil {
		return 0, err
	}

	// Parse the response to find the server's weight
	// Format: be_id be_name srv_id srv_name srv_addr srv_op_state srv_admin_state srv_uweight srv_iweight ...
	weight, err := parseServerWeight(resp, server)
	if err != nil {
		return 0, err
	}

	return weight, nil
}

// parseServerWeight extracts the weight for a given server from "show servers state" output
func parseServerWeight(output, serverName string) (int32, error) {
	scanner := bufio.NewScanner(strings.NewReader(output))
	lineNum := 0

	for scanner.Scan() {
		line := scanner.Text()
		lineNum++

		// Skip header line (first line) and empty lines
		if lineNum == 1 || strings.TrimSpace(line) == "" {
			continue
		}

		// Skip comment lines
		if strings.HasPrefix(strings.TrimSpace(line), "#") {
			continue
		}

		fields := strings.Fields(line)
		// Format: be_id be_name srv_id srv_name srv_addr srv_op_state srv_admin_state srv_uweight srv_iweight ...
		// Index:  0     1       2      3        4        5            6               7           8
		if len(fields) < 8 {
			continue
		}

		if fields[3] == serverName {
			// srv_uweight is at index 7 (user-configured weight)
			weight, err := strconv.ParseInt(fields[7], 10, 32)
			if err != nil {
				return 0, fmt.Errorf("failed to parse weight for server %s: %w", serverName, err)
			}
			return int32(weight), nil
		}
	}

	if err := scanner.Err(); err != nil {
		return 0, fmt.Errorf("error scanning output: %w", err)
	}

	return 0, fmt.Errorf("server %s not found in backend", serverName)
}

// executeCommand sends a command to HAProxy and returns the response
func (c *HAProxyClient) executeCommand(ctx context.Context, cmd string) (string, error) {
	// Create a context with timeout if not already set
	if _, ok := ctx.Deadline(); !ok {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, c.timeout)
		defer cancel()
	}

	// Dial with context
	dialer := &net.Dialer{
		Timeout: c.dialTimeout,
	}

	conn, err := dialer.DialContext(ctx, c.network, c.address)
	if err != nil {
		return "", fmt.Errorf("failed to connect to HAProxy at %s: %w", c.address, err)
	}
	defer conn.Close()

	// Set deadline based on context
	if deadline, ok := ctx.Deadline(); ok {
		if err := conn.SetDeadline(deadline); err != nil {
			return "", fmt.Errorf("failed to set deadline: %w", err)
		}
	}

	// Send command
	if _, err := conn.Write([]byte(cmd)); err != nil {
		return "", fmt.Errorf("failed to send command: %w", err)
	}

	// Read response
	var response strings.Builder
	buf := make([]byte, 4096)

	for {
		n, err := conn.Read(buf)
		if n > 0 {
			response.Write(buf[:n])
		}
		if err != nil {
			// EOF is expected when the server closes the connection
			break
		}
		// Check if we've received a complete response
		// HAProxy closes the connection after sending the response
		if n < len(buf) {
			break
		}
	}

	return response.String(), nil
}

// Ping checks if the HAProxy runtime API is accessible
func (c *HAProxyClient) Ping(ctx context.Context) error {
	// Use "show info" as a simple health check command
	resp, err := c.executeCommand(ctx, "show info\n")
	if err != nil {
		return err
	}

	// Verify we got a valid response
	if !strings.Contains(resp, "Name:") {
		return fmt.Errorf("unexpected response from HAProxy: %s", truncateString(resp, 100))
	}

	return nil
}

// GetBackendStats retrieves statistics for the frontends backend
func (c *HAProxyClient) GetBackendStats(ctx context.Context) (*BackendStats, error) {
	resp, err := c.executeCommand(ctx, fmt.Sprintf("show servers state %s\n", BackendName))
	if err != nil {
		return nil, err
	}

	stats, err := parseBackendStats(resp)
	if err != nil {
		return nil, err
	}

	return stats, nil
}

// BackendStats contains statistics for the backend servers
type BackendStats struct {
	OldServer *ServerStats
	NewServer *ServerStats
}

// ServerStats contains statistics for a single server
type ServerStats struct {
	Name          string
	Address       string
	OperState     string
	AdminState    string
	Weight        int32
	InitialWeight int32
}

// parseBackendStats parses the "show servers state" output into BackendStats
func parseBackendStats(output string) (*BackendStats, error) {
	stats := &BackendStats{}
	scanner := bufio.NewScanner(strings.NewReader(output))
	lineNum := 0

	// Regex to match operational state names
	opStateRegex := regexp.MustCompile(`^[0-2]$`) // 0=stopped, 1=starting, 2=running

	for scanner.Scan() {
		line := scanner.Text()
		lineNum++

		// Skip header and empty lines
		if lineNum == 1 || strings.TrimSpace(line) == "" || strings.HasPrefix(strings.TrimSpace(line), "#") {
			continue
		}

		fields := strings.Fields(line)
		if len(fields) < 9 {
			continue
		}

		serverName := fields[3]
		server := &ServerStats{
			Name:    serverName,
			Address: fields[4],
		}

		// Parse operational state
		if opStateRegex.MatchString(fields[5]) {
			switch fields[5] {
			case "0":
				server.OperState = "stopped"
			case "1":
				server.OperState = "starting"
			case "2":
				server.OperState = "running"
			}
		} else {
			server.OperState = fields[5]
		}

		server.AdminState = fields[6]

		// Parse weights
		if w, err := strconv.ParseInt(fields[7], 10, 32); err == nil {
			server.Weight = int32(w)
		}
		if w, err := strconv.ParseInt(fields[8], 10, 32); err == nil {
			server.InitialWeight = int32(w)
		}

		switch serverName {
		case OldServerName:
			stats.OldServer = server
		case NewServerName:
			stats.NewServer = server
		}
	}

	return stats, scanner.Err()
}

// truncateString truncates a string to maxLen characters
func truncateString(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}

// EnableServer enables a server that was previously disabled
func (c *HAProxyClient) EnableServer(ctx context.Context, backend, server string) error {
	cmd := fmt.Sprintf("set server %s/%s state ready\n", backend, server)
	_, err := c.executeCommand(ctx, cmd)
	return err
}

// DisableServer disables a server (puts it in maintenance mode)
func (c *HAProxyClient) DisableServer(ctx context.Context, backend, server string) error {
	cmd := fmt.Sprintf("set server %s/%s state maint\n", backend, server)
	_, err := c.executeCommand(ctx, cmd)
	return err
}

// DrainServer puts a server into drain mode (no new connections, existing ones finish)
func (c *HAProxyClient) DrainServer(ctx context.Context, backend, server string) error {
	cmd := fmt.Sprintf("set server %s/%s state drain\n", backend, server)
	_, err := c.executeCommand(ctx, cmd)
	return err
}
