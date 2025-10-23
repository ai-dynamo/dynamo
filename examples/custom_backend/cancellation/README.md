# Request Cancellation Demo

This demonstration shows how to implement request cancellation in Dynamo using:
- Client: `context.stop_generating()` to cancel requests
- Middle Server: Forwards requests and passes context through (optional)
- Backend Server: `context.is_stopped()` to check for cancellation

## Architecture

The demo supports two architectures:

**Direct Connection (Default):**
```
Client -> Backend Server
```

**With Middle Server:**
```
Client -> Middle Server -> Backend Server
```

The middle server acts as a proxy that:
1. Receives requests from clients
2. Forwards them to backend servers
3. Passes the original context through for cancellation support
4. Streams responses back to the client

## Usage

### Option 1: Direct Connection (Simple)

1. Start the backend server:
```bash
python3 server.py
```

2. Run the client (connects directly to backend):
```bash
python3 client.py
```

### Option 2: With Middle Server (Proxy)

1. Start the backend server:
```bash
python3 server.py
```

2. Start the middle server:
```bash
python3 middle_server.py
```

3. Run the client (connects through middle server):
```bash
python3 client.py --middle
```

## What happens

### Direct Connection:
1. Backend server generates numbers 0-999 with 0.1 second delays
2. Client receives the first 3 numbers (0, 1, 2) directly from backend
3. Client calls `context.stop_generating()` to cancel
4. Backend server detects cancellation via `context.is_stopped()` and stops
5. Both client and server handle the cancellation gracefully

### With Middle Server:
1. Backend server generates numbers 0-999 with 0.1 second delays
2. Middle server forwards requests and passes context through
3. Client receives the first 3 numbers (0, 1, 2) via the middle server
4. Client calls `context.stop_generating()` to cancel
5. Context cancellation propagates: Client → Middle Server → Backend Server
6. Backend server detects cancellation via `context.is_stopped()` and stops
7. All components handle the cancellation gracefully

## Key Components

- **Backend Server**: Checks `context.is_stopped()` before each yield
- **Middle Server**: Forwards requests and passes context through (when used)
- **Client**: Uses `Context()` object and calls `context.stop_generating()`
- **Graceful shutdown**: All components handle `asyncio.CancelledError`

## Notes

- The client defaults to direct connection for simplicity
- Use `--middle` flag to test the proxy scenario
- Both modes demonstrate the same cancellation behavior
- The middle server shows how to properly forward context in proxy scenarios

For more details on the request cancellation architecture, refer to the [architecture documentation](../../../docs/fault_tolerance/request_cancellation.md).
