# High Availability

All components (in-memory DB, volume/asset storage, code indexer) used by Gitea must be deployed in a HA-ready fashion to achieve a full HA-ready Gitea deployment.
The following document explains how to achieve this for all individual components.

The resulting Gitea deployment will consist of ~ 10 pods (depending on the chosen components and their replicas).
One should evaluate upfront whether a HA-deployment is required as switching between HA/non-HA comes with some effort.
For production instances, HA is always recommended to increase uptime and have a frictionless update process.

A general comment about chart dependencies and external services:
Instead of relying on chart dependencies, it is often better to rely on an external, (managed) instances (in-memory database, asset storage provider, database, etc.).
Many cloud providers offer such services, at least for databases or in-memory databases.
They might cost a bit more than using a self-hosted k8s variant but are usually easier to maintain and scale, if needed.
Also they can be centrally managed and are not linked to the Gitea helm chart or namespace.
Please consider using external services before you start with your Gitea HA setup, it will make your life (and the life of the Gitea maintainers) easier.

This helm chart tries to help as much as possible to simplify and assert the provisioning of a HA-ready Gitea instance by implementing smart conditionals if `replicaCount` is set to a value > 1.
Nevertheless, we cannot guarantee for every possible combination of Gitea settings to work together perfectly in a HA setup.
As a general advice, we recommend to have a test environment aside on which to test possible changes/upgrades before applying these to a production installation.

## Requirements for HA

Storage-wise, the HA-Gitea setup requires a RWX file-system which can be shared among the deployment-based replica pods.
In addition, the following components are required for full HA-readiness:

- A HA-ready issue (and optionally code) indexer: `elasticsearch` or `meilisearch`
- A HA-ready external object/asset storage (`minio`) (optional, assets can also be stored on the RWX file-system)
- A HA-ready cache (`redis-cluster`)
- A HA-ready DB

`postgres.enabled`, which default to `true`, must be set to `false` for a HA setup.
The default `postgres` chart dependency is not HA-ready (there's a dedicated `postgres-ha` chart).

The following sections discuss each of the components in more detail.
Note that for each component discussed, the shown configurations only provides a (working) starting point, not necessarily the most optimal setup.
We try to optimize this document over time as we have gained more experience with HA setups from users.

## Indexers (Issues and code/repo)

The default code indexer `bleve` is not able to allow multiple connections and hence cannot be used in a HA setup.
Alternatives are `elasticsearch` and `meilisearch` (as of >= 1.19.2).
Unless you have an existing `elasticsearch` cluster, we recommend using `meilisearch` as it is faster and requires way less resources.

Unfortunately, `meilisearch` does only support the `ISSUE_INDEXER` and not the `REPO_INDEXER` yet ([tracking issue](https://github.com/go-gitea/gitea/pull/24149)).
This means that the `REPO_INDEXER` must still be disabled for a HA setup right now.
An alternative to the two options above for the `ISSUE_INDEXER` is `"db"`, however we recommend to just go with `meilisearch` in this case and to not bother the DB with indexing.

To configure `meilisearch` within Gitea, do the following:

```yml
gitea:
  config:
    indexer:
      ISSUE_INDEXER_CONN_STR: <http://meilisearch.<namespace>.svc.cluster.local:7700>
      ISSUE_INDEXER_ENABLED: true
      ISSUE_INDEXER_TYPE: meilisearch
      REPO_INDEXER_ENABLED: false
      # REPO_INDEXER_TYPE: meilisearch # not yet working
```

Unfortunately `meilisearch` cannot be deployed in HA as of now.
Nevertheless it allows for multiple Gitea requests at the same time and is therefore required in a HA setup.

Exemplary configuration for the [meilisearch-kubernetes](https://github.com/meilisearch/meilisearch-kubernetes/tree/main/charts/meilisearch) chart:

```yaml
persistence:
  enabled: true
  accessMode: ReadWriteOnce
  size: 5Gi
```

## Cache, session and queue

A `redis` instance is required for the in-memory cache.
Two options exist:

- `redis`
- `redis-cluster`

The chart provides `redis-cluster` as a dependency as this one can be used for both HA and non-HA setups.
You're also welcome to go with `redis` if you prefer or already have a running instance.

It should be noted that `redis-cluster` support is only available starting with Gitea 1.19.2.
You can also configure an external (managed) `redis` instance to be used.
To do so, you need to set the following configuration values yourself:

- `gitea.config.queue.TYPE`: redis`
- `gitea.config.queue.CONN_STR`: `<your redis connection string>`

- `gitea.config.session.PROVIDER`: `redis`
- `gitea.config.session.PROVIDER_CONFIG`: `<your redis connection string>`

- `gitea.config.cache.ENABLED`: `true`
- `gitea.config.cache.ADAPTER`: `redis`
- `gitea.config.cache.HOST`: `<your redis connection string>`

By default, the `redis-cluster` chart provisions three standalone master nodes of which each has a single replica.
To reduce the number of pods for a default Gitea deployment, we opted to omit the replicas (`replicas: 0`) by default.
Only the minimum required number of master pods for a functional `redis-cluster` deployment are provisioned.
For a "proper" `redis-cluster` setup however, we recommend to set `replicas: 1` and `nodes: 6`.

## Object and asset storage

Object/asset storage refers to the storage of attachments, avatars, LFS files, etc.
While most of these can be stored on the RWX file-system, it is recommended to use an external S3-compatible object storage for such, mainly for performance reasons.

By default the chart provisions a single RWO volume to store everything (repos, avatars, packages, etc.).
This volume cannot be mounted by multiple pods.
Hence, a RWX volume is required and (optionally) an external HA-ready object storage.

> **Note:** Double-check that the file permissions are set correctly on the RWX volume! That is everything should be owned by the `git` user which usually has `uid=1000` and `gid=1000`.

To use `minio` you need to deploy and configure an external `minio` instance yourself and explicitly define the `STORAGE_TYPE` values as shown below.

Note that `MINIO_BUCKET` here is just a name and does not refer to a S3 bucket.
It's the root access point for all objects belonging to the respective application, i.e., to Gitea in this case.

```yaml
gitea:
  config:
    attachment:
      STORAGE_TYPE: minio
    lfs:
      STORAGE_TYPE: minio
    picture:
      AVATAR_STORAGE_TYPE: minio
    "storage.packages":
      STORAGE_TYPE: minio

    storage:
      MINIO_ENDPOINT: <minio-headless.<namespace>.svc.cluster.local:9000>
      MINIO_LOCATION: <location>
      MINIO_ACCESS_KEY_ID: <access key>
      MINIO_SECRET_ACCESS_KEY: <secret key>
      MINIO_BUCKET: <bucket name>
      MINIO_USE_SSL: false
```

Exemplary configuration for the [bitnami minio](https://github.com/bitnami/charts/blob/main/bitnami/minio) chart:

```yaml
auth:
  rootUser: minio
mode: distributed
replicaCount: 4
persistence:
  enabled: true
  size: 20Gi
  accessModes:
    - ReadWriteOnce
```

## Database

If you do not have an HA-ready DB, using a managed database service in the cloud might be the easiest and most robust solution.
Remember: disable the built-in `postgres` dependency and configure the database connection manually via `gitea.config.database`:

```yml
gitea:
  database:
    builtIn:
      postgresql:
        enabled: false
  config:
    database:
      DB_TYPE: postgres
      HOST: <host>
      NAME: <name>
      USER: <user>
```

## Known issues

- Currently Cron jobs are run on all replicas as no leader election is implemented.
  See [https://github.com/go-gitea/gitea/issues/13791](https://github.com/go-gitea/gitea/issues/13791) for a discussion and possible solution.

- Running with multiple replicas slows down Gitea a bit, i.e. page loading time increases. 