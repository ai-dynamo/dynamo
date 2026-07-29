---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: TCP TLS
subtitle: Encrypt TCP response stream connections between frontend and workers
---

Dynamo supports opt-in TLS encryption on the TCP response stream transport
(the call-home path between frontends and workers). When enabled, all TCP
connections are upgraded to TLS using [rustls](https://github.com/rustls/rustls)
with the `ring` cryptographic provider. When no TLS configuration is provided,
the transport operates in plaintext exactly as before.

## Environment variables

All TLS configuration is driven by environment variables. The Rust runtime
reads these directly at first connection (lazy initialization).

### Server side (frontend)

| Variable | Description |
|---|---|
| `DYN_TCP_TLS_CERT_PATH` | Path to the PEM certificate file. When set together with `DYN_TCP_TLS_KEY_PATH`, TLS is enabled on the TCP server. |
| `DYN_TCP_TLS_KEY_PATH` | Path to the PEM private key for the server certificate. |

### Client side (worker)

| Variable | Description |
|---|---|
| `DYN_TCP_TLS_CA_CERT_PATH` | Path to the PEM CA certificate used to verify the server. Required when the server uses a self-signed or internal CA. |
| `DYN_TCP_TLS_INSECURE` | Set to `1` or `true` to skip certificate verification. For local development only. |
| `DYN_TCP_TLS_SERVER_NAME` | Override the TLS SNI hostname. Useful when connecting by IP to a server whose certificate has a DNS SAN. |

### Shared

| Variable | Description |
|---|---|
| `DYN_TCP_TLS_HANDSHAKE_TIMEOUT_SECS` | TLS handshake timeout in seconds (default: 3). |

## CLI flags

The same configuration is available via command-line flags on all backends
(vllm, sglang, trtllm, tokenspeed) through `DynamoRuntimeArgGroup`:

```
--tcp-tls-cert-path PATH      Server certificate (PEM)
--tcp-tls-key-path PATH       Server private key (PEM)
--tcp-tls-ca-cert-path PATH   CA certificate for server verification (PEM)
--tcp-tls-insecure             Disable certificate verification
--tcp-tls-server-name NAME     Override TLS SNI hostname
--tcp-tls-handshake-timeout N  Handshake timeout in seconds (default: 3)
```

The frontend (`dynamo.frontend`) also accepts `--tcp-tls-cert-path`,
`--tcp-tls-key-path`, and `--tcp-tls-ca-cert-path`.

## Quick start

Generate a self-signed certificate for local testing:

```bash
# Generate CA
openssl req -x509 -newkey rsa:2048 -keyout ca-key.pem -out ca-cert.pem \
  -days 365 -nodes -subj "/CN=DynamoCA"

# Generate server cert with SAN
openssl req -newkey rsa:2048 -keyout server-key.pem -out server-csr.pem \
  -nodes -subj "/CN=localhost" \
  -addext "subjectAltName=DNS:localhost,IP:127.0.0.1"

openssl x509 -req -in server-csr.pem -CA ca-cert.pem -CAkey ca-key.pem \
  -CAcreateserial -out server-cert.pem -days 365 -copy_extensions copyall
```

Start the backend (worker) with TLS:

```bash
python -m dynamo.vllm \
  --tcp-tls-cert-path server-cert.pem \
  --tcp-tls-key-path server-key.pem \
  --tcp-tls-ca-cert-path ca-cert.pem \
  --tcp-tls-server-name localhost \
  ...
```

Start the frontend with TLS:

```bash
python -m dynamo.frontend \
  --tcp-tls-cert-path server-cert.pem \
  --tcp-tls-key-path server-key.pem \
  --tcp-tls-ca-cert-path ca-cert.pem \
  ...
```

## Kubernetes deployment

In Kubernetes, TLS certificates are typically delivered by a certificate
management system (e.g., cert-manager) and mounted into pods. Set the
environment variables in the pod spec or via the Dynamo operator's
`InfrastructureConfiguration`:

```yaml
env:
  - name: DYN_TCP_TLS_CERT_PATH
    value: /etc/certs/server/cert.pem
  - name: DYN_TCP_TLS_KEY_PATH
    value: /etc/certs/server/key.pem
  - name: DYN_TCP_TLS_CA_CERT_PATH
    value: /etc/certs/ca/ca.pem
```

## Design notes

- TLS configuration is cached after the first TCP connection via `OnceCell`.
  Certificate rotation requires a process restart.
- The TLS handshake is spawned per-connection on the server side so the accept
  loop is never blocked by a slow handshake.
- When server and client TLS configurations are mismatched (e.g., server has TLS
  but client does not), a warning is logged at startup.
- An empty CA certificate file is detected at load time and rejected with a
  clear error message.
