package pagebroker

import (
	"context"
	"encoding/binary"
	"errors"
	"net"
	"path/filepath"
	"testing"
	"time"

	"google.golang.org/protobuf/proto"
)

func TestRequestStopsWhenContextIsCanceled(t *testing.T) {
	listener, err := net.Listen("unix", filepath.Join(t.TempDir(), "pagebroker.sock"))
	if err != nil {
		t.Fatal(err)
	}
	defer listener.Close()

	accepted := make(chan net.Conn, 1)
	go func() {
		connection, err := listener.Accept()
		if err == nil {
			accepted <- connection
		}
	}()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	result := make(chan error, 1)
	go func() {
		result <- (Client{ControlSocketPath: listener.Addr().String()}).Abort(ctx, "transaction")
	}()

	connection := <-accepted
	defer connection.Close()
	if _, err := readMessage(connection); err != nil {
		t.Fatal(err)
	}

	cancel()
	select {
	case err := <-result:
		if err == nil {
			t.Fatal("request succeeded after its context was canceled")
		}
	case <-time.After(time.Second):
		t.Fatal("request did not stop after its context was canceled")
	}
}

func TestCommitRetriesLostResponses(t *testing.T) {
	listener, err := net.Listen("unix", filepath.Join(t.TempDir(), "pagebroker.sock"))
	if err != nil {
		t.Fatal(err)
	}
	defer listener.Close()

	requests := make(chan *Request, 3)
	server := make(chan error, 1)
	go func() {
		for attempt := 0; attempt < 3; attempt++ {
			connection, err := listener.Accept()
			if err != nil {
				server <- err
				return
			}
			message, err := readMessage(connection)
			if err != nil {
				_ = connection.Close()
				server <- err
				return
			}
			request := new(Request)
			if err := proto.Unmarshal(message, request); err != nil {
				_ = connection.Close()
				server <- err
				return
			}
			requests <- request
			if attempt == 2 {
				response := &Response{
					RequestId:     request.RequestId,
					TransactionId: request.TransactionId,
					Result:        &Response_CommitComplete{CommitComplete: &CommitComplete{}},
				}
				message, err = proto.Marshal(response)
				if err == nil {
					err = writeMessage(connection, message)
				}
				if err != nil {
					_ = connection.Close()
					server <- err
					return
				}
			}
			_ = connection.Close()
		}
		server <- nil
	}()

	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	if err := (Client{ControlSocketPath: listener.Addr().String()}).Commit(ctx, "transaction"); err != nil {
		t.Fatal(err)
	}
	if err := <-server; err != nil {
		t.Fatal(err)
	}
	for range 3 {
		request := <-requests
		if request.GetTransactionId() != "transaction" || request.GetCommit() == nil {
			t.Fatalf("unexpected retry request: %v", request)
		}
	}
}

func TestAbortRequiresAbortComplete(t *testing.T) {
	listener, err := net.Listen("unix", filepath.Join(t.TempDir(), "pagebroker.sock"))
	if err != nil {
		t.Fatal(err)
	}
	defer listener.Close()

	server := make(chan error, 1)
	go func() {
		connection, err := listener.Accept()
		if err != nil {
			server <- err
			return
		}
		defer connection.Close()
		message, err := readMessage(connection)
		if err != nil {
			server <- err
			return
		}
		request := new(Request)
		if err := proto.Unmarshal(message, request); err != nil {
			server <- err
			return
		}
		message, err = proto.Marshal(&Response{
			RequestId:     request.RequestId,
			TransactionId: request.TransactionId,
			Result:        &Response_CommitComplete{CommitComplete: &CommitComplete{}},
		})
		if err == nil {
			err = writeMessage(connection, message)
		}
		server <- err
	}()

	if err := (Client{ControlSocketPath: listener.Addr().String()}).Abort(context.Background(), "transaction"); err == nil {
		t.Fatal("abort accepted a commit response")
	}
	if err := <-server; err != nil {
		t.Fatal(err)
	}
}

func TestCommitDoesNotRetryInvalidFrame(t *testing.T) {
	listener, err := net.Listen("unix", filepath.Join(t.TempDir(), "pagebroker.sock"))
	if err != nil {
		t.Fatal(err)
	}
	defer listener.Close()

	server := make(chan error, 1)
	go func() {
		connection, err := listener.Accept()
		if err != nil {
			server <- err
			return
		}
		defer connection.Close()
		if _, err := readMessage(connection); err != nil {
			server <- err
			return
		}
		server <- binary.Write(connection, binary.BigEndian, uint32(maxMessageSize+1))
	}()

	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	err = (Client{ControlSocketPath: listener.Addr().String()}).Commit(ctx, "transaction")
	if !errors.Is(err, errMessageTooLarge) {
		t.Fatalf("Commit() error = %v, want invalid frame error", err)
	}
	if err := <-server; err != nil {
		t.Fatal(err)
	}
}
