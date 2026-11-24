---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: etcd Authentication
subtitle: Run Dynamo against an etcd server that requires TLS and client authentication
---

This page shows how to run Dynamo against an etcd server that encrypts traffic
with TLS and accepts only authenticated clients, with a Dynamo user that can
read and write only Dynamo's own keys.

It applies when Dynamo uses etcd for discovery (`--discovery-backend etcd` or
`DYN_DISCOVERY_BACKEND=etcd`). That is the default for Dynamo processes you
start yourself on a local or bare-metal host.

## How Dynamo reads etcd credentials

Every Dynamo component reads its etcd connection settings from these
environment variables:

| Variable | Contents |
|----------|----------|
| `ETCD_ENDPOINTS` | Comma-separated etcd URLs. Default: `http://localhost:2379`. Use `https://` URLs for TLS. |
| `ETCD_AUTH_CA` | PEM contents of the CA certificate that signed the etcd server certificate. |
| `ETCD_AUTH_CLIENT_CERT` | PEM contents of the client certificate Dynamo presents to etcd. |
| `ETCD_AUTH_CLIENT_KEY` | PEM contents of the private key for that client certificate. |
| `ETCD_AUTH_USERNAME` | etcd user name for password authentication. |
| `ETCD_AUTH_PASSWORD` | etcd password for password authentication. |

The client applies these rules:

- The three TLS variables hold **PEM contents, not file paths**. Set them with
  `"$(cat file.pem)"`.
- Dynamo uses TLS only when **all three** TLS variables are set. A CA alone is
  ignored.
- When `ETCD_AUTH_USERNAME` and `ETCD_AUTH_PASSWORD` are both set, Dynamo uses
  password authentication and **ignores the TLS variables**. In that mode it
  trusts no CA, so it cannot connect to an `https://` endpoint, and over an
  `http://` endpoint the password crosses the network unencrypted.

For these reasons, authenticate Dynamo with a **client certificate** (mutual
TLS). etcd maps the certificate's common name (CN) to an etcd user and checks
that user's role on every request. Use password authentication for Dynamo only
on loopback or a network that no one else can reach.

## Prerequisites

The steps below run one etcd member on `127.0.0.1` with TLS and client
certificate authentication, create a `dynamo` user whose role is limited to the
`v1/` key prefix, and start a Dynamo frontend and worker as that user. They use
`openssl` and the etcd v3.5 `etcd` and `etcdctl` binaries. For a server that
other hosts reach, add its host names and IP addresses to the server
certificate and use them in the URLs.

## Create the certificates

Create a CA, a server certificate for etcd, and two client certificates:
`dynamo` for Dynamo components and `admin` for `etcdctl`.

```bash
mkdir -p etcd-tls && cd etcd-tls

# Certificate authority
openssl req -x509 -newkey rsa:4096 -nodes -days 365 \
  -keyout ca-key.pem -out ca.pem -subj "/CN=etcd-ca"

# etcd server certificate. List every name and address clients use.
openssl req -newkey rsa:2048 -nodes -keyout server-key.pem -out server.csr \
  -subj "/CN=etcd-server"
openssl x509 -req -in server.csr -CA ca.pem -CAkey ca-key.pem -CAcreateserial \
  -days 365 -out server.pem \
  -extfile <(printf "subjectAltName=DNS:localhost,IP:127.0.0.1\nextendedKeyUsage=serverAuth")

# Client certificates. The CN is the etcd user the certificate authenticates as.
for user in dynamo admin; do
  openssl req -newkey rsa:2048 -nodes -keyout "$user-key.pem" -out "$user.csr" \
    -subj "/CN=$user"
  openssl x509 -req -in "$user.csr" -CA ca.pem -CAkey ca-key.pem -CAcreateserial \
    -days 365 -out "$user.pem" -extfile <(printf "extendedKeyUsage=clientAuth")
done
chmod 600 ./*-key.pem
```

`admin` is not an etcd user, so its certificate only lets `etcdctl` complete
the TLS handshake; administrative commands also need the `root` password
created in [Create the users and turn on authentication](#create-the-users-and-turn-on-authentication). Keep `ca-key.pem` offline once the certificates exist.

## Start etcd with TLS and client certificate authentication

```bash
etcd --name dynamo-etcd --data-dir ./data \
  --cert-file server.pem --key-file server-key.pem \
  --trusted-ca-file ca.pem --client-cert-auth \
  --listen-client-urls https://127.0.0.1:2379 \
  --advertise-client-urls https://127.0.0.1:2379
```

`--client-cert-auth` makes etcd reject any client that does not present a
certificate signed by `ca.pem`. Run etcd under a process manager such as
systemd for a long-lived deployment; for this walkthrough, run it in a second
terminal from the `etcd-tls` directory.

## Create the users and turn on authentication

```bash
etcd_admin() {
  etcdctl --endpoints=https://127.0.0.1:2379 --cacert=ca.pem \
    --cert=admin.pem --key=admin-key.pem "$@"
}

# root administers etcd once authentication is on.
openssl rand -base64 24 > root-password && chmod 600 root-password
etcd_admin user add root --new-user-password="$(cat root-password)"
etcd_admin user grant-role root root

# dynamo authenticates with its certificate only and may use keys under v1/.
etcd_admin user add dynamo --no-password
etcd_admin role add dynamo
etcd_admin role grant-permission dynamo readwrite v1/ --prefix=true
etcd_admin user grant-role dynamo dynamo

etcd_admin auth enable
```

Dynamo stores all of its discovery data, model cards, and planner state under
the `v1/` prefix, so `readwrite` on `v1/` is all the `dynamo` role needs. etcd
lets any authenticated user create leases, and the `v1/` permission covers the
keys Dynamo attaches to them.

## Start Dynamo with the client certificate

In each shell that starts a Dynamo component, run from the `etcd-tls`
directory:

```bash
export ETCD_ENDPOINTS=https://127.0.0.1:2379
export ETCD_AUTH_CA="$(cat ca.pem)"
export ETCD_AUTH_CLIENT_CERT="$(cat dynamo.pem)"
export ETCD_AUTH_CLIENT_KEY="$(cat dynamo-key.pem)"
```

Then start the components as usual, for example a frontend and a worker:

```bash
python -m dynamo.frontend --http-port 8000 --discovery-backend etcd
```

```bash
python -m dynamo.mocker --model-path Qwen/Qwen3-0.6B --discovery-backend etcd
```

Do not set `ETCD_AUTH_USERNAME` or `ETCD_AUTH_PASSWORD` in these shells; if
both are set, Dynamo ignores the certificate variables.

## Check the setup

List the keys Dynamo registered, as `root`:

```bash
etcd_admin --user=root --password="$(cat root-password)" \
  get --prefix v1/ --keys-only
```

The output includes keys under `v1/instances/` and `v1/mdc/`. Then check that
the server refuses what it should:

```bash
# The dynamo user cannot write outside v1/: "permission denied".
etcdctl --endpoints=https://127.0.0.1:2379 --cacert=ca.pem \
  --cert=dynamo.pem --key=dynamo-key.pem put other/key value

# A client without a certificate cannot connect: "context deadline exceeded".
etcdctl --endpoints=https://127.0.0.1:2379 --cacert=ca.pem \
  --dial-timeout=3s --command-timeout=5s get v1/ --prefix --keys-only
```

A Dynamo component started without the `ETCD_AUTH_*` variables keeps logging
`etcd not reachable yet; retrying startup connection` and does not start.
