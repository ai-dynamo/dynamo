// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
package pagebroker

import (
	"context"
	"fmt"
	"net"
	"os"
	"path/filepath"
	"time"

	"github.com/google/uuid"
)

const defaultSocket = "/run/pagebroker/pagebroker.sock"

type Client struct{ socket string }

func NewClient(socket string) *Client {
	if socket == "" {
		socket = defaultSocket
	}
	return &Client{socket: socket}
}

type Transaction struct {
	client           *Client
	id               string
	staging, scratch string
	provider         *os.File
}

func (t *Transaction) StagingPath() string { return t.staging }

func Stage(ctx context.Context, socket, checkpoint string) (*Transaction, error) {
	return NewClient(socket).Stage(ctx, checkpoint)
}

func (c *Client) Stage(ctx context.Context, checkpoint string) (*Transaction, error) {
	id := "tx-" + uuid.NewString()
	r, provider, err := c.submit(ctx, id, checkpoint)
	if err != nil {
		return nil, err
	}
	if !r.ok {
		return nil, fmt.Errorf("submit rejected: %s", r.err)
	}
	if _, err := os.Stat(checkpoint); err != nil {
		_ = provider.Close()
		cleanup, cancel := context.WithTimeout(context.WithoutCancel(ctx), 5*time.Second)
		defer cancel()
		_, _ = c.call(cleanup, 4, id, "")
		return nil, fmt.Errorf("checkpoint is unavailable after submit: %w", err)
	}
	return &Transaction{client: c, id: id, staging: checkpoint, scratch: r.scratch, provider: provider}, nil
}

func PrepareCheckpoint(ctx context.Context, socket, checkpoint string) (*Transaction, error) {
	return NewClient(socket).PrepareCheckpoint(ctx, checkpoint)
}

func (c *Client) PrepareCheckpoint(ctx context.Context, checkpoint string) (*Transaction, error) {
	id := "tx-" + uuid.NewString()
	r, err := c.call(ctx, 2, id, checkpoint)
	if err != nil {
		return nil, err
	}
	if !r.ok {
		return nil, fmt.Errorf("checkpoint prepare rejected: %s", r.err)
	}
	return &Transaction{client: c, id: id, staging: r.staging, scratch: r.scratch}, nil
}

func (t *Transaction) Files() ([]*os.File, error) {
	if t.provider == nil {
		return nil, fmt.Errorf("PageBroker provider connection is unavailable")
	}
	image, err := os.Open(t.staging)
	if err != nil {
		return nil, fmt.Errorf("open staged checkpoint: %w", err)
	}
	scratch := filepath.Clean(t.scratch)
	if err := os.MkdirAll(scratch, 0755); err != nil {
		image.Close()
		return nil, err
	}
	work, err := os.Open(scratch)
	if err != nil {
		image.Close()
		return nil, fmt.Errorf("open PageBroker scratch: %w", err)
	}
	provider := t.provider
	t.provider = nil
	return []*os.File{provider, image, work}, nil
}
func (t *Transaction) Commit(ctx context.Context) error {
	t.closeProvider()
	r, err := t.client.call(ctx, 3, t.id, "")
	if err != nil {
		return err
	}
	if !r.ok {
		return fmt.Errorf("commit rejected: %s", r.err)
	}
	return nil
}
func (t *Transaction) Abort(ctx context.Context) error {
	t.closeProvider()
	r, err := t.client.call(ctx, 4, t.id, "")
	if err != nil {
		return err
	}
	if !r.ok {
		return fmt.Errorf("abort rejected: %s", r.err)
	}
	return nil
}

func (t *Transaction) closeProvider() {
	if t.provider != nil {
		_ = t.provider.Close()
		t.provider = nil
	}
}

type response struct {
	ok                    bool
	staging, scratch, err string
}

func varint(v uint64) []byte {
	var b []byte
	for v >= 128 {
		b = append(b, byte(v)|128)
		v >>= 7
	}
	return append(b, byte(v))
}
func field(n int, v []byte) []byte {
	return append(append(varint(uint64(n*8+2)), varint(uint64(len(v)))...), v...)
}
func request(op int, id, path string) []byte {
	b := append(varint(8), varint(uint64(op))...)
	if id != "" {
		b = append(b, field(2, []byte(id))...)
	}
	if path != "" {
		b = append(b, field(3, []byte(path))...)
	}
	return b
}

func watchContext(ctx context.Context, connection net.Conn) func() bool {
	if deadline, ok := ctx.Deadline(); ok {
		_ = connection.SetDeadline(deadline)
	}
	return context.AfterFunc(ctx, func() {
		_ = connection.SetDeadline(time.Now())
	})
}

func contextError(ctx context.Context, err error) error {
	if ctx.Err() != nil {
		return ctx.Err()
	}
	return err
}

func (c *Client) submit(ctx context.Context, id, checkpoint string) (response, *os.File, error) {
	connection, err := (&net.Dialer{}).DialContext(ctx, "unixpacket", c.socket)
	if err != nil {
		return response{}, nil, err
	}
	defer connection.Close()
	stop := watchContext(ctx, connection)
	defer stop()
	unixConnection, ok := connection.(*net.UnixConn)
	if !ok {
		return response{}, nil, fmt.Errorf("PageBroker connection is %T, want UnixConn", connection)
	}
	if _, err := connection.Write(request(1, id, checkpoint)); err != nil {
		return response{}, nil, contextError(ctx, err)
	}
	buf := make([]byte, 65536)
	n, err := connection.Read(buf)
	if err != nil {
		return response{}, nil, contextError(ctx, err)
	}
	r, err := parse(buf[:n])
	if err != nil || !r.ok {
		return r, nil, err
	}
	provider, err := unixConnection.File()
	if err != nil {
		return response{}, nil, fmt.Errorf("duplicate PageBroker provider connection: %w", err)
	}
	return r, provider, nil
}

func call(ctx context.Context, socket string, op int, id, path string) (response, error) {
	return NewClient(socket).call(ctx, op, id, path)
}

func (c *Client) call(ctx context.Context, op int, id, path string) (response, error) {
	connection, err := (&net.Dialer{}).DialContext(ctx, "unixpacket", c.socket)
	if err != nil {
		return response{}, err
	}
	defer connection.Close()
	stop := watchContext(ctx, connection)
	defer stop()
	if _, err := connection.Write(request(op, id, path)); err != nil {
		return response{}, contextError(ctx, err)
	}
	buf := make([]byte, 65536)
	n, err := connection.Read(buf)
	if err != nil {
		return response{}, contextError(ctx, err)
	}
	return parse(buf[:n])
}

func skipField(b []byte, tag uint64) ([]byte, error) {
	switch tag & 7 {
	case 0:
		_, n := read(b)
		if n == 0 {
			return nil, fmt.Errorf("invalid unknown varint")
		}
		return b[n:], nil
	case 1:
		if len(b) < 8 {
			return nil, fmt.Errorf("invalid unknown fixed64")
		}
		return b[8:], nil
	case 2:
		length, n := read(b)
		if n == 0 || length > uint64(len(b)-n) {
			return nil, fmt.Errorf("invalid unknown string")
		}
		return b[n+int(length):], nil
	case 3:
		field := tag >> 3
		for len(b) > 0 {
			nested, n := read(b)
			if n == 0 {
				break
			}
			b = b[n:]
			if nested&7 == 4 {
				if nested>>3 != field {
					return nil, fmt.Errorf("mismatched unknown group")
				}
				return b, nil
			}
			var err error
			b, err = skipField(b, nested)
			if err != nil {
				return nil, err
			}
		}
		return nil, fmt.Errorf("unterminated unknown group")
	case 5:
		if len(b) < 4 {
			return nil, fmt.Errorf("invalid unknown fixed32")
		}
		return b[4:], nil
	default:
		return nil, fmt.Errorf("unsupported response wire type")
	}
}

func parse(b []byte) (response, error) {
	var r response
	for len(b) > 0 {
		tag, n := read(b)
		if n == 0 {
			return r, fmt.Errorf("invalid PageBroker response")
		}
		b = b[n:]
		f, w := int(tag>>3), tag&7
		if w == 0 {
			v, k := read(b)
			if k == 0 {
				return r, fmt.Errorf("invalid response varint")
			}
			b = b[k:]
			if f == 1 {
				r.ok = v != 0
			}
		} else if w == 2 {
			l, k := read(b)
			if k == 0 || l > uint64(len(b)-k) {
				return r, fmt.Errorf("invalid response string")
			}
			v := string(b[k : k+int(l)])
			b = b[k+int(l):]
			switch f {
			case 3:
				r.staging = v
			case 4:
				r.scratch = v
			case 5:
				r.err = v
			}
		} else if w == 1 {
			if len(b) < 8 {
				return r, fmt.Errorf("invalid response fixed64")
			}
			b = b[8:]
		} else if w == 5 {
			if len(b) < 4 {
				return r, fmt.Errorf("invalid response fixed32")
			}
			b = b[4:]
		} else if w == 3 {
			var err error
			b, err = skipField(b, tag)
			if err != nil {
				return r, err
			}
		} else {
			return r, fmt.Errorf("unsupported response wire type")
		}
	}
	return r, nil
}
func read(b []byte) (uint64, int) {
	var v uint64
	for i, c := range b {
		if i == 9 {
			if c > 1 {
				return 0, 0
			}
			return v | uint64(c)<<63, i + 1
		}
		v |= uint64(c&127) << uint(7*i)
		if c < 128 {
			return v, i + 1
		}
	}
	return 0, 0
}
