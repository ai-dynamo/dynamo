# Custom Auth and Storage Providers for Stateful Responses

**Quick Start Guide for Bringing Your Own Implementation**

This guide shows how to implement custom authentication and storage backends for Dynamo's stateful Responses API.

---

## Overview

Dynamo provides **trait-based interfaces** for both authentication and storage, allowing you to plug in your existing infrastructure:

```
┌─────────────────────────────────────────────┐
│          Your Custom Systems                │
│  ┌──────────────┐      ┌──────────────┐    │
│  │ Your Auth    │      │ Your Storage │    │
│  │ (LDAP, SSO)  │      │ (DB, S3)     │    │
│  └──────────────┘      └──────────────┘    │
│         │                       │           │
│         v                       v           │
│  ┌──────────────┐      ┌──────────────┐    │
│  │ AuthProvider │      │ResponseStorage│   │
│  │   Trait      │      │    Trait      │    │
│  └──────────────┘      └──────────────┘    │
└─────────────────────────────────────────────┘
                    │
                    v
            ┌──────────────┐
            │    Dynamo    │
            │  HTTP Server │
            └──────────────┘
```

---

## Custom Authentication Provider

### Step 1: Implement the `AuthProvider` Trait

Create `lib/custom_auth/src/lib.rs`:

```rust
use dynamo_llm::http::auth::{AuthProvider, AuthContext, AuthError, Role};
use async_trait::async_trait;
use axum::http::HeaderMap;
use std::collections::HashMap;

pub struct MyAuthProvider {
    // Your auth system config
    ldap_server: String,
    oauth_client_id: String,
    // etc.
}

impl MyAuthProvider {
    pub fn new(config: MyAuthConfig) -> Self {
        Self {
            ldap_server: config.ldap_server,
            oauth_client_id: config.oauth_client_id,
        }
    }
}

#[async_trait]
impl AuthProvider for MyAuthProvider {
    async fn authenticate(
        &self,
        headers: &HeaderMap,
    ) -> Result<AuthContext, AuthError> {
        // 1. Extract credentials from headers
        let token = headers
            .get("Authorization")
            .and_then(|h| h.to_str().ok())
            .and_then(|s| s.strip_prefix("Bearer "))
            .ok_or(AuthError::MissingCredentials)?;

        // 2. Validate with your auth system (LDAP, OAuth, etc.)
        let user_info = self.validate_token(token).await
            .map_err(|e| AuthError::InvalidCredentials)?;

        // 3. Return auth context
        Ok(AuthContext {
            tenant_id: user_info.organization_id,
            user_id: user_info.user_id,
            role: match user_info.role.as_str() {
                "admin" => Role::Admin,
                "service" => Role::ServiceAccount,
                _ => Role::User,
            },
            permissions: vec![],  // Optional: map from your permissions
            metadata: HashMap::new(),
        })
    }

    async fn authorize(
        &self,
        context: &AuthContext,
        resource: &str,
        action: &str,
    ) -> Result<bool, AuthError> {
        // Optional: Implement fine-grained authorization
        // For example, check if user can delete a specific response
        Ok(true)
    }
}

impl MyAuthProvider {
    async fn validate_token(&self, token: &str) -> Result<UserInfo, MyAuthError> {
        // Your validation logic:
        // - LDAP bind
        // - OAuth token introspection
        // - JWT verification
        // - Database lookup
        // - etc.
        todo!()
    }
}

struct UserInfo {
    user_id: String,
    organization_id: String,
    role: String,
}
```

### Step 2: Register Your Provider

In your main.rs or entrypoint:

```rust
use std::sync::Arc;
use dynamo_llm::http::service::service_v2::State;
use my_auth_provider::MyAuthProvider;

#[tokio::main]
async fn main() {
    // 1. Load config
    let auth_config = MyAuthConfig::from_env();

    // 2. Create your auth provider
    let auth_provider: Arc<dyn AuthProvider> = Arc::new(
        MyAuthProvider::new(auth_config)
    );

    // 3. Pass to Dynamo state
    let state = State::builder()
        .auth_provider(auth_provider)
        .build();

    // 4. Start server
    dynamo_llm::entrypoint::run(state).await;
}
```

---

## Custom Storage Backend

### Step 1: Implement the `ResponseStorage` Trait

Create `lib/custom_storage/src/lib.rs`:

```rust
use dynamo_llm::storage::{
    ResponseStorage, StoredResponse, ResponseQuery, StorageError
};
use dynamo_llm::http::auth::AuthContext;
use async_trait::async_trait;
use std::time::Duration;

pub struct MyStorageBackend {
    // Your storage config
    db_pool: sqlx::PgPool,
    s3_client: aws_sdk_s3::Client,
}

impl MyStorageBackend {
    pub fn new(db_pool: sqlx::PgPool, s3_client: aws_sdk_s3::Client) -> Self {
        Self { db_pool, s3_client }
    }
}

#[async_trait]
impl ResponseStorage for MyStorageBackend {
    async fn store_response(
        &self,
        response: Response,
        auth: &AuthContext,
        ttl: Option<Duration>,
    ) -> Result<String, StorageError> {
        // 1. Create stored response with metadata
        let stored = StoredResponse {
            response_id: response.id.clone(),
            tenant_id: auth.tenant_id.clone(),
            user_id: auth.user_id.clone(),
            response,
            created_at: unix_timestamp(),
            updated_at: unix_timestamp(),
            accessed_at: unix_timestamp(),
            expires_at: ttl.map(|d| unix_timestamp() + d.as_secs()),
            permissions: None,
        };

        // 2. Serialize response
        let data = serde_json::to_vec(&stored)
            .map_err(|e| StorageError::BackendError(e.to_string()))?;

        // 3. Store in your backend
        // Option A: Store in S3 for large responses
        if data.len() > 1_000_000 {  // 1MB threshold
            let s3_key = format!(
                "{}/responses/{}.json",
                auth.tenant_id,
                stored.response_id
            );
            self.s3_client
                .put_object()
                .bucket("dynamo-responses")
                .key(&s3_key)
                .body(data.into())
                .send()
                .await
                .map_err(|e| StorageError::BackendError(e.to_string()))?;
        }

        // Option B: Store metadata in Postgres for fast queries
        sqlx::query(
            r#"
            INSERT INTO responses (
                response_id, tenant_id, user_id,
                s3_key, size_bytes, created_at, expires_at
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            "#
        )
        .bind(&stored.response_id)
        .bind(&auth.tenant_id)
        .bind(&auth.user_id)
        .bind(&s3_key)
        .bind(data.len() as i64)
        .bind(stored.created_at as i64)
        .bind(stored.expires_at.map(|e| e as i64))
        .execute(&self.db_pool)
        .await
        .map_err(|e| StorageError::BackendError(e.to_string()))?;

        Ok(stored.response_id)
    }

    async fn get_response(
        &self,
        response_id: &str,
        auth: &AuthContext,
    ) -> Result<StoredResponse, StorageError> {
        // 1. Query metadata from Postgres
        let row = sqlx::query_as::<_, ResponseMetadata>(
            r#"
            SELECT response_id, tenant_id, user_id, s3_key,
                   created_at, accessed_at, expires_at
            FROM responses
            WHERE response_id = $1 AND tenant_id = $2
            "#
        )
        .bind(response_id)
        .bind(&auth.tenant_id)
        .fetch_optional(&self.db_pool)
        .await
        .map_err(|e| StorageError::BackendError(e.to_string()))?
        .ok_or_else(|| StorageError::NotFound(response_id.to_string()))?;

        // 2. Verify access (tenant isolation)
        if row.tenant_id != auth.tenant_id {
            return Err(StorageError::AccessDenied("Tenant mismatch".to_string()));
        }

        // 3. Check if expired
        if let Some(expires_at) = row.expires_at {
            if expires_at < unix_timestamp() {
                return Err(StorageError::NotFound("Response expired".to_string()));
            }
        }

        // 4. Fetch from S3
        let obj = self.s3_client
            .get_object()
            .bucket("dynamo-responses")
            .key(&row.s3_key)
            .send()
            .await
            .map_err(|e| StorageError::BackendError(e.to_string()))?;

        let data = obj.body.collect().await
            .map_err(|e| StorageError::BackendError(e.to_string()))?
            .into_bytes();

        // 5. Deserialize
        let stored: StoredResponse = serde_json::from_slice(&data)
            .map_err(|e| StorageError::BackendError(e.to_string()))?;

        Ok(stored)
    }

    async fn delete_response(
        &self,
        response_id: &str,
        auth: &AuthContext,
    ) -> Result<(), StorageError> {
        // Get metadata to verify ownership and get S3 key
        let row = sqlx::query_as::<_, ResponseMetadata>(
            "SELECT * FROM responses WHERE response_id = $1 AND tenant_id = $2"
        )
        .bind(response_id)
        .bind(&auth.tenant_id)
        .fetch_optional(&self.db_pool)
        .await
        .map_err(|e| StorageError::BackendError(e.to_string()))?
        .ok_or_else(|| StorageError::NotFound(response_id.to_string()))?;

        // Verify user is owner (or admin)
        if row.user_id != auth.user_id && auth.role != Role::Admin {
            return Err(StorageError::AccessDenied("Not owner".to_string()));
        }

        // Delete from S3
        self.s3_client
            .delete_object()
            .bucket("dynamo-responses")
            .key(&row.s3_key)
            .send()
            .await
            .map_err(|e| StorageError::BackendError(e.to_string()))?;

        // Delete from Postgres
        sqlx::query("DELETE FROM responses WHERE response_id = $1")
            .bind(response_id)
            .execute(&self.db_pool)
            .await
            .map_err(|e| StorageError::BackendError(e.to_string()))?;

        Ok(())
    }

    async fn list_responses(
        &self,
        query: ResponseQuery,
        auth: &AuthContext,
    ) -> Result<Vec<StoredResponse>, StorageError> {
        // Build dynamic query based on filters
        let mut sql = String::from(
            "SELECT response_id FROM responses WHERE tenant_id = $1"
        );

        if let Some(user_id) = &query.user_id {
            sql.push_str(" AND user_id = $2");
        }

        if let Some(created_after) = query.created_after {
            sql.push_str(" AND created_at > $3");
        }

        sql.push_str(" ORDER BY created_at DESC");

        if let Some(limit) = query.limit {
            sql.push_str(&format!(" LIMIT {}", limit));
        }

        // Execute query
        let rows = sqlx::query_as::<_, (String,)>(&sql)
            .bind(&auth.tenant_id)
            .fetch_all(&self.db_pool)
            .await
            .map_err(|e| StorageError::BackendError(e.to_string()))?;

        // Fetch full responses (or just return IDs for efficiency)
        let mut responses = Vec::new();
        for (response_id,) in rows {
            if let Ok(stored) = self.get_response(&response_id, auth).await {
                responses.push(stored);
            }
        }

        Ok(responses)
    }

    async fn touch_response(
        &self,
        response_id: &str,
        auth: &AuthContext,
    ) -> Result<(), StorageError> {
        sqlx::query(
            "UPDATE responses SET accessed_at = $1
             WHERE response_id = $2 AND tenant_id = $3"
        )
        .bind(unix_timestamp() as i64)
        .bind(response_id)
        .bind(&auth.tenant_id)
        .execute(&self.db_pool)
        .await
        .map_err(|e| StorageError::BackendError(e.to_string()))?;

        Ok(())
    }
}

#[derive(sqlx::FromRow)]
struct ResponseMetadata {
    response_id: String,
    tenant_id: String,
    user_id: String,
    s3_key: String,
    created_at: i64,
    accessed_at: i64,
    expires_at: Option<i64>,
}
```

### Step 2: Register Your Storage Backend

```rust
use my_storage_backend::MyStorageBackend;

#[tokio::main]
async fn main() {
    // 1. Setup your storage
    let db_pool = sqlx::PgPool::connect(&config.database_url).await?;
    let s3_client = aws_sdk_s3::Client::new(&aws_config);

    // 2. Create storage backend
    let storage: Arc<dyn ResponseStorage> = Arc::new(
        MyStorageBackend::new(db_pool, s3_client)
    );

    // 3. Pass to Dynamo state
    let state = State::builder()
        .auth_provider(auth_provider)
        .response_storage(storage)
        .build();

    dynamo_llm::entrypoint::run(state).await;
}
```

---

## Example: LDAP Auth + PostgreSQL Storage

Complete example combining custom auth and storage:

```rust
// main.rs
use std::sync::Arc;

mod ldap_auth;
mod postgres_storage;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Load config
    let config = load_config()?;

    // Setup LDAP auth
    let auth: Arc<dyn AuthProvider> = Arc::new(
        ldap_auth::LdapAuthProvider::new(
            config.ldap_server,
            config.ldap_bind_dn,
            config.ldap_bind_password,
        )
    );

    // Setup PostgreSQL storage
    let db_pool = sqlx::PgPool::connect(&config.database_url).await?;
    let storage: Arc<dyn ResponseStorage> = Arc::new(
        postgres_storage::PostgresStorage::new(db_pool)
    );

    // Build Dynamo state with custom providers
    let state = dynamo_llm::http::service::service_v2::State::builder()
        .auth_provider(auth)
        .response_storage(storage)
        .conversation_storage(storage.clone())  // Can reuse same backend
        .build();

    // Start server
    dynamo_llm::entrypoint::run(state).await
}
```

---

## Testing Your Custom Providers

### Unit Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_auth_valid_token() {
        let provider = MyAuthProvider::new(test_config());
        let headers = make_headers("Bearer valid_token");

        let result = provider.authenticate(&headers).await;
        assert!(result.is_ok());

        let context = result.unwrap();
        assert_eq!(context.tenant_id, "test-tenant");
    }

    #[tokio::test]
    async fn test_auth_invalid_token() {
        let provider = MyAuthProvider::new(test_config());
        let headers = make_headers("Bearer invalid_token");

        let result = provider.authenticate(&headers).await;
        assert!(matches!(result, Err(AuthError::InvalidCredentials)));
    }

    #[tokio::test]
    async fn test_storage_store_and_retrieve() {
        let storage = MyStorageBackend::new(test_db_pool()).await;
        let auth = test_auth_context();
        let response = test_response();

        // Store
        let id = storage.store_response(response.clone(), &auth, None).await?;

        // Retrieve
        let stored = storage.get_response(&id, &auth).await?;
        assert_eq!(stored.response.id, response.id);
    }

    #[tokio::test]
    async fn test_storage_tenant_isolation() {
        let storage = MyStorageBackend::new(test_db_pool()).await;

        // User A stores response
        let auth_a = AuthContext { tenant_id: "tenant-a", user_id: "user-1", .. };
        let id = storage.store_response(test_response(), &auth_a, None).await?;

        // User B from different tenant tries to access
        let auth_b = AuthContext { tenant_id: "tenant-b", user_id: "user-2", .. };
        let result = storage.get_response(&id, &auth_b).await;

        assert!(matches!(result, Err(StorageError::NotFound(_))));
    }
}
```

### Integration Tests

```rust
#[tokio::test]
async fn test_end_to_end_stateful_request() {
    // 1. Setup server with custom providers
    let server = test_server_with_providers(my_auth(), my_storage());

    // 2. First request
    let response1 = server
        .post("/v1/responses")
        .header("Authorization", "Bearer valid_token")
        .json(&json!({
            "model": "gpt-4",
            "input": "What is 2+2?",
            "store": true
        }))
        .await;

    assert_eq!(response1.status(), 200);
    let resp_id = response1.json::<Response>().await.id;

    // 3. Second request using previous_response_id
    let response2 = server
        .post("/v1/responses")
        .header("Authorization", "Bearer valid_token")
        .json(&json!({
            "model": "gpt-4",
            "previous_response_id": resp_id,
            "input": "Times 3?"
        }))
        .await;

    assert_eq!(response2.status(), 200);
}
```

---

## Configuration Options

Environment variables for custom providers:

```bash
# Tell Dynamo to use custom providers
DYN_AUTH_PROVIDER=custom
DYN_STORAGE_PROVIDER=custom

# Your provider-specific config
MY_LDAP_SERVER=ldap://ldap.example.com
MY_LDAP_BIND_DN=cn=admin,dc=example,dc=com
MY_LDAP_BIND_PASSWORD=secret

MY_DB_URL=postgres://user:pass@localhost/dynamo
MY_S3_BUCKET=dynamo-responses
MY_S3_REGION=us-west-2
```

---

## Common Patterns

### 1. Database Schema for Responses

```sql
-- Postgres example
CREATE TABLE responses (
    response_id VARCHAR(64) PRIMARY KEY,
    tenant_id VARCHAR(64) NOT NULL,
    user_id VARCHAR(64) NOT NULL,

    -- Store full response or S3 reference
    response_data JSONB,
    s3_key VARCHAR(255),

    -- Metadata
    size_bytes BIGINT,
    created_at BIGINT NOT NULL,
    updated_at BIGINT NOT NULL,
    accessed_at BIGINT NOT NULL,
    expires_at BIGINT,

    -- Indexes
    INDEX idx_tenant_user_created (tenant_id, user_id, created_at DESC),
    INDEX idx_expires (expires_at) WHERE expires_at IS NOT NULL
);

-- Cleanup job
CREATE INDEX idx_expired ON responses (expires_at)
WHERE expires_at < EXTRACT(EPOCH FROM NOW());
```

### 2. Hybrid Storage (Metadata + Blob)

- **Small responses** (< 1MB): Store in database
- **Large responses** (> 1MB): Store in S3/blob storage
- **Metadata**: Always in database for fast queries

### 3. Multi-Region Replication

```rust
pub struct MultiRegionStorage {
    primary: Arc<dyn ResponseStorage>,
    replicas: Vec<Arc<dyn ResponseStorage>>,
}

#[async_trait]
impl ResponseStorage for MultiRegionStorage {
    async fn store_response(...) -> Result<String, StorageError> {
        // Write to primary
        let id = self.primary.store_response(...).await?;

        // Async replicate to other regions (fire and forget)
        for replica in &self.replicas {
            tokio::spawn(async move {
                replica.store_response(...).await.ok();
            });
        }

        Ok(id)
    }
}
```

---

## Migration Examples

### From Redis to Your Database

```rust
pub async fn migrate_from_redis_to_postgres(
    redis: &RedisResponseStorage,
    postgres: &PostgresStorage,
) -> Result<usize, MigrationError> {
    let mut migrated = 0;

    // Scan all Redis keys
    let keys: Vec<String> = redis.scan("*:responses:*").await?;

    for key in keys {
        // Get from Redis
        if let Ok(data) = redis.client.get::<Vec<u8>>(&key).await {
            // Parse
            let stored: StoredResponse = serde_json::from_slice(&data)?;

            // Store in Postgres
            postgres.store_raw(stored).await?;

            migrated += 1;
        }
    }

    Ok(migrated)
}
```

---

## Performance Tips

### 1. Connection Pooling

```rust
// Use connection pools for databases
let db_pool = sqlx::PgPool::builder()
    .max_connections(100)
    .min_connections(10)
    .connect(&db_url)
    .await?;
```

### 2. Caching

```rust
pub struct CachedStorage {
    backend: Arc<dyn ResponseStorage>,
    cache: Arc<Cache<String, StoredResponse>>,
}

#[async_trait]
impl ResponseStorage for CachedStorage {
    async fn get_response(...) -> Result<StoredResponse, StorageError> {
        // Check cache first
        if let Some(cached) = self.cache.get(response_id) {
            return Ok(cached);
        }

        // Fallback to backend
        let stored = self.backend.get_response(...).await?;

        // Cache for next time
        self.cache.insert(response_id, stored.clone());

        Ok(stored)
    }
}
```

### 3. Batch Operations

```rust
async fn list_responses_batch(
    &self,
    ids: &[String],
    auth: &AuthContext,
) -> Result<Vec<StoredResponse>, StorageError> {
    // Fetch all in parallel
    let futures: Vec<_> = ids.iter()
        .map(|id| self.get_response(id, auth))
        .collect();

    let results = futures::future::join_all(futures).await;

    // Filter out errors
    Ok(results.into_iter().filter_map(|r| r.ok()).collect())
}
```

---

## Support & Troubleshooting

### Common Issues

1. **"Access Denied" errors**: Check tenant_id matching in auth and storage
2. **"Not Found" on valid IDs**: Verify tenant isolation isn't blocking access
3. **Slow queries**: Add database indexes on tenant_id + created_at
4. **Memory issues**: Use streaming for large responses, store in blob storage

### Debug Logging

Enable verbose logging:
```bash
RUST_LOG=dynamo_llm::http::auth=debug,dynamo_llm::storage=debug
```

### Metrics to Monitor

- Auth failures / success rate
- Storage latency (p50, p95, p99)
- Storage errors
- Tenant isolation violations (should be 0!)

---

## Next Steps

1. **Review the main plan**: [stateful_responses_plan.md](./stateful_responses_plan.md)
2. **Check existing traits**: Look at `lib/llm/src/http/auth/mod.rs` (once implemented)
3. **Run examples**: See `examples/custom_providers/` (once added)
4. **Join discussion**: Linear issue [DYN-2045](https://linear.app/nvidia/issue/DYN-2045)
