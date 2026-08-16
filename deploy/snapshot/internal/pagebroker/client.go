// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package pagebroker

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"net"
	"time"

	"github.com/google/uuid"
	"google.golang.org/protobuf/proto"
)

const (
	// PageBroker control requests and responses are limited to 64 KiB.
	maxMessageSize   = 64 << 10
	commitRetryDelay = 100 * time.Millisecond
)

var errMessageTooLarge = fmt.Errorf("message exceeds %d bytes", maxMessageSize)

// Client uses the deployment-wide filesystem/POSIX PageBroker plan.
type Client struct {
	ControlSocketPath string
}

func (c Client) PrepareCheckpoint(ctx context.Context, transactionID, destination string) (string, error) {
	response, err := c.request(ctx, transactionID, &Request_PrepareStagedCheckpoint{
		PrepareStagedCheckpoint: &PrepareStagedCheckpointRequest{Destination: filesystem(destination), IoEngine: posixCopy()},
	})
	if err != nil {
		return "", err
	}
	if response.GetStagedCheckpointDirectory() == nil {
		return "", fmt.Errorf("unexpected PageBroker checkpoint response")
	}
	return response.GetStagedCheckpointDirectory().GetImageDirectory(), nil
}

func (c Client) Commit(ctx context.Context, transactionID string) error {
	for {
		response, err := c.request(ctx, transactionID, &Request_Commit{Commit: &CommitRequest{}})
		if err == nil {
			if response.GetCommitComplete() != nil {
				return nil
			}
			return fmt.Errorf("unexpected PageBroker commit response")
		}
		var transport transportError
		if !errors.As(err, &transport) {
			return err
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(commitRetryDelay):
		}
	}
}

func (c Client) Abort(ctx context.Context, transactionID string) error {
	response, err := c.request(ctx, transactionID, &Request_Abort{Abort: &AbortRequest{}})
	if err != nil {
		return err
	}
	if response.GetAbortComplete() == nil {
		return fmt.Errorf("unexpected PageBroker abort response")
	}
	return nil
}

func (c Client) request(ctx context.Context, transactionID string, command isRequest_Command) (*Response, error) {
	connection, err := (&net.Dialer{}).DialContext(ctx, "unix", c.ControlSocketPath)
	if err != nil {
		return nil, transportError{cause: fmt.Errorf("dial PageBroker: %w", err)}
	}
	defer connection.Close()
	stopCancel := context.AfterFunc(ctx, func() { _ = connection.Close() })
	defer stopCancel()

	requestID := uuid.NewString()
	request := &Request{RequestId: &requestID, TransactionId: &transactionID, Command: command}
	message, err := proto.Marshal(request)
	if err != nil {
		return nil, fmt.Errorf("marshal PageBroker request: %w", err)
	}
	if err := writeMessage(connection, message); err != nil {
		return nil, transportError{cause: fmt.Errorf("write PageBroker request: %w", err)}
	}
	message, err = readMessage(connection)
	if err != nil {
		if errors.Is(err, errMessageTooLarge) {
			return nil, err
		}
		return nil, transportError{cause: fmt.Errorf("read PageBroker response: %w", err)}
	}
	response := new(Response)
	if err := proto.Unmarshal(message, response); err != nil {
		return nil, fmt.Errorf("unmarshal PageBroker response: %w", err)
	}
	if response.GetRequestId() != requestID || response.GetTransactionId() != transactionID {
		return nil, fmt.Errorf("PageBroker response identifiers do not match request")
	}
	if failure := response.GetFailure(); failure != nil {
		return nil, failureError{code: failureCode(failure.GetCode()), message: failure.GetMessage()}
	}
	return response, nil
}

type failureError struct {
	code    Failure_Code
	message string
}

func failureCode(code Failure_Code) Failure_Code {
	switch code {
	case Failure_UNSPECIFIED, Failure_INVALID_REQUEST, Failure_TRANSACTION_NOT_FOUND, Failure_TRANSACTION_CONFLICT,
		Failure_INSUFFICIENT_STORAGE, Failure_STORAGE_ERROR, Failure_INTERNAL_ERROR:
		return code
	default:
		return Failure_UNSPECIFIED
	}
}

type transportError struct {
	cause error
}

func (e transportError) Error() string { return e.cause.Error() }

func (e transportError) Unwrap() error { return e.cause }

func (e failureError) Error() string {
	return fmt.Sprintf("PageBroker %s: %s", e.code, e.message)
}

func filesystem(directory string) *StorageBackend {
	return &StorageBackend{Kind: &StorageBackend_Filesystem{Filesystem: &FilesystemStorage{Directory: &directory}}}
}

func posixCopy() *IOEngine {
	return &IOEngine{Kind: &IOEngine_PosixCopy{PosixCopy: &PosixCopyIOEngine{}}}
}

func writeMessage(writer io.Writer, message []byte) error {
	if len(message) > maxMessageSize {
		return fmt.Errorf("message exceeds %d bytes", maxMessageSize)
	}
	if err := binary.Write(writer, binary.BigEndian, uint32(len(message))); err != nil {
		return err
	}
	for len(message) > 0 {
		written, err := writer.Write(message)
		if err != nil {
			return err
		}
		if written == 0 {
			return io.ErrShortWrite
		}
		message = message[written:]
	}
	return nil
}

func readMessage(reader io.Reader) ([]byte, error) {
	var size uint32
	if err := binary.Read(reader, binary.BigEndian, &size); err != nil {
		return nil, err
	}
	if size > maxMessageSize {
		return nil, errMessageTooLarge
	}
	message := make([]byte, size)
	_, err := io.ReadFull(reader, message)
	return message, err
}
