# Stateful Responses API Implementation Plan

**Status**: Planning
**Assignee**: TBD
**Linear**: [DYN-2045](https://linear.app/nvidia/issue/DYN-2045/investigate-v1responses-being-stateful)
**Priority**: Medium (requires security architecture first)

## Executive Summary

This document outlines the implementation plan for adding **stateful conversation management** to Dynamo's Responses API. The current implementation (PR #5854) provides a fully functional **stateless** Responses API. This plan adds server-side state management via `previous_response_id`, `conversation` references, and the Conversation API endpoints.

**Key Design Principle**: **Pluggable Architecture** - Allow users to bring their own authentication and storage backends while providing secure defaults.

---

## Background

### Current State (Stateless Mode) ✅

Clients manage conversation history by sending all previous messages in each request:

```json
{
  "model": "gpt-4",
  "input": [
    {"type": "message", "role": "user", "content": "What's 2+2?"},
    {"type": "message", "role": "assistant", "content": "4"},
    {"type": "message", "role": "user", "content": "Times 3?"}
  ]
}
```

**Benefits**: Simple, no server-side state, works today
**Drawbacks**: High bandwidth, client complexity, no server-side caching optimization

### Target State (Stateful Mode) 🎯

Server manages conversation state; clients send only deltas:

```json
{
  "model": "gpt-4",
  "previous_response_id": "resp_abc123",
  "input": "Times 3?"
}
```

**Benefits**: Lower bandwidth, simplified clients, server-side caching, conversation analytics
**Drawbacks**: Requires storage, security, and distributed state management

---

## Architecture Overview

### Core Components

```
┌─────────────────────────────────────────────────────────────┐
│                     HTTP Request                             │
│                 (with Auth Headers)                          │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       v
┌─────────────────────────────────────────────────────────────┐
│                 Auth Middleware (Pluggable)                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  API Key     │  │     JWT      │  │    mTLS      │      │
│  │  Provider    │  │   Provider   │  │   Provider   │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│                          │                                   │
│                          v                                   │
│                   AuthContext { user_id, tenant_id }        │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       v
┌─────────────────────────────────────────────────────────────┐
│              Response Handler (openai.rs)                    │
│  - Validate previous_response_id / conversation             │
│  - Authorize access (tenant + user checks)                  │
│  - Retrieve stored context from Storage Backend             │
│  - Merge with new input                                     │
│  - Process via Chat Completions Engine                      │
│  - Store response if store=true                             │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       v
┌─────────────────────────────────────────────────────────────┐
│              Storage Backend (Pluggable)                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │    Redis     │  │   Postgres   │  │   S3 + DDB   │      │
│  │  (Default)   │  │   (Custom)   │  │   (Custom)   │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│                                                              │
│  Common Interface: ResponseStorage trait                    │
│  - store_response(response, auth)                           │
│  - get_response(id, auth)                                   │
│  - delete_response(id, auth)                                │
│  - list_responses(filters, auth)                            │
└─────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Pluggable Auth Framework (4 weeks)

### 1.1 Auth Trait Definition (1 week)

**File**: `lib/llm/src/http/auth/mod.rs`

```rust
/// Authentication context extracted from request
#[derive(Debug, Clone)]
pub struct AuthContext {
    /// Unique tenant identifier for multi-tenancy isolation
    pub tenant_id: String,

    /// Unique user identifier within the tenant
    pub user_id: String,

    /// User's role (admin, user, service_account)
    pub role: Role,

    /// Optional: Fine-grained permissions
    pub permissions: Vec<Permission>,

    /// Optional: Metadata (IP, user-agent, etc.)
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Role {
    Admin,
    User,
    ServiceAccount,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Permission {
    ReadResponses,
    WriteResponses,
    DeleteResponses,
    ReadConversations,
    WriteConversations,
    DeleteConversations,
}

/// Trait for authentication providers
#[async_trait]
pub trait AuthProvider: Send + Sync {
    /// Authenticate a request and extract context
    async fn authenticate(
        &self,
        headers: &HeaderMap,
    ) -> Result<AuthContext, AuthError>;

    /// Optional: Validate a specific permission
    async fn authorize(
        &self,
        context: &AuthContext,
        resource: &str,
        action: &str,
    ) -> Result<bool, AuthError>;
}

#[derive(Debug, thiserror::Error)]
pub enum AuthError {
    #[error("Missing authentication credentials")]
    MissingCredentials,

    #[error("Invalid credentials")]
    InvalidCredentials,

    #[error("Insufficient permissions: {0}")]
    Forbidden(String),

    #[error("Authentication provider error: {0}")]
    ProviderError(String),
}
```

### 1.2 Built-in Auth Providers (2 weeks)

#### Option A: API Key Provider (Default)

**File**: `lib/llm/src/http/auth/api_key.rs`

```rust
pub struct ApiKeyAuthProvider {
    key_store: Arc<dyn KeyStore>,
}

#[async_trait]
pub trait KeyStore: Send + Sync {
    /// Look up API key and return associated context
    async fn validate_key(&self, key: &str) -> Result<AuthContext, AuthError>;
}

/// In-memory key store for testing/development
pub struct InMemoryKeyStore {
    keys: HashMap<String, AuthContext>,
}

/// Redis-backed key store for production
pub struct RedisKeyStore {
    client: RedisClient,
}
```

**Configuration** (environment variables):
```bash
DYN_AUTH_PROVIDER=api_key
DYN_AUTH_API_KEY_STORE=redis  # or "in_memory" for dev
DYN_AUTH_REDIS_URL=redis://localhost:6379
```

#### Option B: JWT Provider

**File**: `lib/llm/src/http/auth/jwt.rs`

```rust
pub struct JwtAuthProvider {
    /// Public key for verification (RS256, ES256)
    public_key: DecodingKey,

    /// Algorithm
    algorithm: Algorithm,

    /// Expected issuer
    issuer: String,
}

impl JwtAuthProvider {
    /// Extract tenant_id and user_id from JWT claims
    fn extract_context(&self, claims: &Claims) -> AuthContext {
        AuthContext {
            tenant_id: claims.custom.get("tenant_id").unwrap().clone(),
            user_id: claims.sub.clone(),
            role: self.parse_role(&claims.custom.get("role")),
            permissions: self.parse_permissions(&claims.custom),
            metadata: HashMap::new(),
        }
    }
}
```

**Configuration**:
```bash
DYN_AUTH_PROVIDER=jwt
DYN_AUTH_JWT_PUBLIC_KEY_PATH=/path/to/public.pem
DYN_AUTH_JWT_ALGORITHM=RS256
DYN_AUTH_JWT_ISSUER=https://auth.example.com
```

#### Option C: Custom Provider (User-Provided)

Allow users to implement their own auth:

```rust
// User creates lib/custom_auth/my_auth.rs
pub struct MyCustomAuth {
    // Custom fields
}

#[async_trait]
impl AuthProvider for MyCustomAuth {
    async fn authenticate(&self, headers: &HeaderMap) -> Result<AuthContext, AuthError> {
        // Custom logic: LDAP, OAuth, SSO, etc.
    }
}

// Register in main.rs
let auth_provider = Arc::new(MyCustomAuth::new());
let state = State::new(auth_provider);
```

### 1.3 Auth Middleware (1 week)

**File**: `lib/llm/src/http/middleware/auth.rs`

```rust
pub async fn auth_middleware(
    State(auth_provider): State<Arc<dyn AuthProvider>>,
    mut request: Request<Body>,
    next: Next,
) -> Result<Response, ErrorResponse> {
    // Extract headers
    let headers = request.headers();

    // Authenticate
    let auth_context = auth_provider
        .authenticate(headers)
        .await
        .map_err(|e| match e {
            AuthError::MissingCredentials | AuthError::InvalidCredentials => {
                ErrorMessage::unauthorized(e.to_string())
            }
            AuthError::Forbidden(msg) => ErrorMessage::forbidden(msg),
            AuthError::ProviderError(msg) => ErrorMessage::internal_server_error(&msg),
        })?;

    // Store in request extensions for downstream handlers
    request.extensions_mut().insert(auth_context);

    Ok(next.run(request).await)
}
```

**Apply to routes**:
```rust
// In openai.rs
pub fn responses_router(
    state: Arc<service_v2::State>,
    auth_provider: Arc<dyn AuthProvider>,
) -> Router {
    Router::new()
        .route("/v1/responses", post(handler_responses))
        .layer(middleware::from_fn_state(auth_provider.clone(), auth_middleware))
        .with_state(state)
}
```

---

## Phase 2: Pluggable Storage Backend (4 weeks)

### 2.1 Storage Trait Definition (1 week)

**File**: `lib/llm/src/storage/mod.rs`

```rust
/// Stored response with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredResponse {
    // Identity
    pub response_id: String,
    pub tenant_id: String,
    pub user_id: String,

    // Response data
    pub response: Response,

    // Timestamps
    pub created_at: u64,
    pub updated_at: u64,
    pub accessed_at: u64,
    pub expires_at: Option<u64>,

    // Access control (optional)
    pub permissions: Option<Vec<StoredPermission>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StoredPermission {
    Owner,
    SharedReadOnly { users: Vec<String> },
    SharedReadWrite { users: Vec<String> },
}

/// Query filters for listing responses
#[derive(Debug, Default)]
pub struct ResponseQuery {
    pub tenant_id: String,
    pub user_id: Option<String>,
    pub conversation_id: Option<String>,
    pub created_after: Option<u64>,
    pub created_before: Option<u64>,
    pub limit: Option<usize>,
    pub offset: Option<usize>,
}

/// Trait for response storage backends
#[async_trait]
pub trait ResponseStorage: Send + Sync {
    /// Store a response with auth context
    async fn store_response(
        &self,
        response: Response,
        auth: &AuthContext,
        ttl: Option<Duration>,
    ) -> Result<String, StorageError>;

    /// Retrieve a response by ID
    async fn get_response(
        &self,
        response_id: &str,
        auth: &AuthContext,
    ) -> Result<StoredResponse, StorageError>;

    /// Delete a response
    async fn delete_response(
        &self,
        response_id: &str,
        auth: &AuthContext,
    ) -> Result<(), StorageError>;

    /// List responses matching filters
    async fn list_responses(
        &self,
        query: ResponseQuery,
        auth: &AuthContext,
    ) -> Result<Vec<StoredResponse>, StorageError>;

    /// Update last accessed timestamp
    async fn touch_response(
        &self,
        response_id: &str,
        auth: &AuthContext,
    ) -> Result<(), StorageError>;
}

#[derive(Debug, thiserror::Error)]
pub enum StorageError {
    #[error("Response not found: {0}")]
    NotFound(String),

    #[error("Access denied: {0}")]
    AccessDenied(String),

    #[error("Storage quota exceeded")]
    QuotaExceeded,

    #[error("Storage backend error: {0}")]
    BackendError(String),
}
```

### 2.2 Built-in Storage Providers (2 weeks)

#### Option A: Redis Storage (Default)

**File**: `lib/llm/src/storage/redis.rs`

```rust
pub struct RedisResponseStorage {
    client: RedisClient,
    encryption: Option<Arc<dyn EncryptionProvider>>,
    max_response_size: usize,
}

impl RedisResponseStorage {
    /// Key format: {tenant_id}:responses:{response_id}
    fn make_key(&self, tenant_id: &str, response_id: &str) -> String {
        format!("{}:responses:{}", tenant_id, response_id)
    }

    /// Secondary index: {tenant_id}:responses:user:{user_id}:list
    fn make_user_index_key(&self, tenant_id: &str, user_id: &str) -> String {
        format!("{}:responses:user:{}:list", tenant_id, user_id)
    }
}

#[async_trait]
impl ResponseStorage for RedisResponseStorage {
    async fn store_response(
        &self,
        response: Response,
        auth: &AuthContext,
        ttl: Option<Duration>,
    ) -> Result<String, StorageError> {
        // Create StoredResponse
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

        // Serialize
        let data = serde_json::to_vec(&stored)
            .map_err(|e| StorageError::BackendError(e.to_string()))?;

        // Check size limit
        if data.len() > self.max_response_size {
            return Err(StorageError::QuotaExceeded);
        }

        // Encrypt if provider is configured
        let final_data = if let Some(enc) = &self.encryption {
            enc.encrypt(&data, &auth.tenant_id).await?
        } else {
            data
        };

        // Store in Redis
        let key = self.make_key(&auth.tenant_id, &stored.response_id);
        let ttl_secs = ttl.map(|d| d.as_secs()).unwrap_or(86400); // 24h default

        self.client.set_ex(&key, final_data, ttl_secs).await
            .map_err(|e| StorageError::BackendError(e.to_string()))?;

        // Add to user index
        let index_key = self.make_user_index_key(&auth.tenant_id, &auth.user_id);
        self.client.zadd(&index_key, &stored.response_id, unix_timestamp() as f64).await
            .map_err(|e| StorageError::BackendError(e.to_string()))?;

        Ok(stored.response_id)
    }

    async fn get_response(
        &self,
        response_id: &str,
        auth: &AuthContext,
    ) -> Result<StoredResponse, StorageError> {
        let key = self.make_key(&auth.tenant_id, response_id);

        // Get from Redis
        let data: Vec<u8> = self.client.get(&key).await
            .map_err(|_| StorageError::NotFound(response_id.to_string()))?;

        // Decrypt if needed
        let decrypted = if let Some(enc) = &self.encryption {
            enc.decrypt(&data, &auth.tenant_id).await?
        } else {
            data
        };

        // Deserialize
        let stored: StoredResponse = serde_json::from_slice(&decrypted)
            .map_err(|e| StorageError::BackendError(e.to_string()))?;

        // Verify tenant isolation (defense in depth)
        if stored.tenant_id != auth.tenant_id {
            return Err(StorageError::AccessDenied("Tenant mismatch".to_string()));
        }

        // Check permissions
        if !self.can_access(&stored, auth) {
            return Err(StorageError::AccessDenied("User not authorized".to_string()));
        }

        Ok(stored)
    }
}
```

**Configuration**:
```bash
DYN_STORAGE_PROVIDER=redis
DYN_STORAGE_REDIS_URL=redis://localhost:6379
DYN_STORAGE_REDIS_PASSWORD=<secure-password>
DYN_STORAGE_REDIS_TLS=true
DYN_STORAGE_MAX_RESPONSE_SIZE_MB=10
DYN_STORAGE_DEFAULT_TTL_HOURS=24
DYN_STORAGE_MAX_TTL_HOURS=168  # 7 days
```

#### Option B: PostgreSQL Storage

**File**: `lib/llm/src/storage/postgres.rs`

```sql
-- Schema
CREATE TABLE responses (
    response_id VARCHAR(64) PRIMARY KEY,
    tenant_id VARCHAR(64) NOT NULL,
    user_id VARCHAR(64) NOT NULL,
    response_data BYTEA NOT NULL,  -- Encrypted JSON
    created_at BIGINT NOT NULL,
    updated_at BIGINT NOT NULL,
    accessed_at BIGINT NOT NULL,
    expires_at BIGINT,

    -- Indexes for efficient queries
    INDEX idx_tenant_user (tenant_id, user_id, created_at DESC),
    INDEX idx_expires (expires_at) WHERE expires_at IS NOT NULL
);
```

#### Option C: S3 + DynamoDB

For very large conversations:
- **S3**: Store response blobs
- **DynamoDB**: Store metadata and indexes

### 2.3 Encryption Provider (1 week)

**File**: `lib/llm/src/storage/encryption.rs`

```rust
#[async_trait]
pub trait EncryptionProvider: Send + Sync {
    async fn encrypt(&self, data: &[u8], tenant_id: &str) -> Result<Vec<u8>, StorageError>;
    async fn decrypt(&self, data: &[u8], tenant_id: &str) -> Result<Vec<u8>, StorageError>;
}

/// AES-256-GCM encryption
pub struct Aes256GcmProvider {
    /// Master key for encrypting tenant keys
    master_key: Key<Aes256Gcm>,

    /// Per-tenant key cache
    tenant_keys: Arc<DashMap<String, Key<Aes256Gcm>>>,
}

/// AWS KMS-backed encryption
pub struct AwsKmsProvider {
    kms_client: KmsClient,
    key_id: String,
}
```

---

## Phase 3: Response Storage Integration (3 weeks)

### 3.1 Update Response Handler (1 week)

**File**: `lib/llm/src/http/service/openai.rs`

```rust
async fn handler_responses(
    State(state): State<Arc<service_v2::State>>,
    Extension(auth): Extension<AuthContext>,  // From auth middleware
    Json(request): Json<NvCreateResponse>,
) -> Result<Response, ErrorResponse> {
    // Check if previous_response_id is provided
    if let Some(prev_id) = &request.inner.previous_response_id {
        // Retrieve previous response from storage
        let prev_response = state.storage()
            .get_response(prev_id, &auth)
            .await
            .map_err(|e| match e {
                StorageError::NotFound(_) => ErrorMessage::not_found(),
                StorageError::AccessDenied(_) => ErrorMessage::forbidden("Cannot access previous response"),
                _ => ErrorMessage::internal_server_error(&e.to_string()),
            })?;

        // Merge previous context with new input
        let merged_input = merge_with_previous(
            &prev_response.response,
            &request.inner.input,
        )?;

        // Update request with merged input
        request.inner.input = merged_input;
    }

    // Process request (existing logic)
    let response = process_response_request(state.clone(), request).await?;

    // Store response if requested
    if request.inner.store == Some(true) {
        let ttl = calculate_ttl(&request.inner);
        state.storage()
            .store_response(response.clone(), &auth, ttl)
            .await
            .map_err(|e| {
                tracing::warn!("Failed to store response: {}", e);
                // Don't fail the request if storage fails
            });
    }

    Ok(Json(response))
}

fn merge_with_previous(
    previous: &Response,
    new_input: &InputParam,
) -> Result<InputParam, ErrorResponse> {
    // Extract all output items from previous response
    let prev_items: Vec<InputItem> = previous.output
        .iter()
        .map(|item| convert_output_to_input(item))
        .collect();

    // Combine with new input
    match new_input {
        InputParam::Text(text) => {
            let mut items = prev_items;
            items.push(InputItem::EasyMessage(EasyInputMessage {
                r#type: MessageType::Message,
                role: Role::User,
                content: text.clone(),
            }));
            Ok(InputParam::Items(items))
        }
        InputParam::Items(new_items) => {
            let mut items = prev_items;
            items.extend(new_items.clone());
            Ok(InputParam::Items(items))
        }
    }
}
```

### 3.2 Add GET /v1/responses/{id} (1 week)

```rust
async fn get_response(
    State(state): State<Arc<service_v2::State>>,
    Extension(auth): Extension<AuthContext>,
    Path(response_id): Path<String>,
) -> Result<Json<Response>, ErrorResponse> {
    let stored = state.storage()
        .get_response(&response_id, &auth)
        .await
        .map_err(|e| match e {
            StorageError::NotFound(_) => ErrorMessage::not_found(),
            StorageError::AccessDenied(_) => ErrorMessage::forbidden("Access denied"),
            _ => ErrorMessage::internal_server_error(&e.to_string()),
        })?;

    // Touch to update accessed_at
    let _ = state.storage().touch_response(&response_id, &auth).await;

    Ok(Json(stored.response))
}
```

### 3.3 Add DELETE /v1/responses/{id} (1 week)

```rust
async fn delete_response(
    State(state): State<Arc<service_v2::State>>,
    Extension(auth): Extension<AuthContext>,
    Path(response_id): Path<String>,
) -> Result<StatusCode, ErrorResponse> {
    state.storage()
        .delete_response(&response_id, &auth)
        .await
        .map_err(|e| match e {
            StorageError::NotFound(_) => ErrorMessage::not_found(),
            StorageError::AccessDenied(_) => ErrorMessage::forbidden("Access denied"),
            _ => ErrorMessage::internal_server_error(&e.to_string()),
        })?;

    Ok(StatusCode::NO_CONTENT)
}
```

---

## Phase 4: Conversation API (3 weeks)

### 4.1 Conversation Storage (1 week)

Similar trait to ResponseStorage:

```rust
#[async_trait]
pub trait ConversationStorage: Send + Sync {
    async fn create_conversation(
        &self,
        metadata: Option<serde_json::Value>,
        auth: &AuthContext,
    ) -> Result<String, StorageError>;

    async fn get_conversation(
        &self,
        conversation_id: &str,
        auth: &AuthContext,
    ) -> Result<Conversation, StorageError>;

    async fn add_items(
        &self,
        conversation_id: &str,
        items: Vec<InputItem>,
        auth: &AuthContext,
    ) -> Result<(), StorageError>;

    async fn list_items(
        &self,
        conversation_id: &str,
        auth: &AuthContext,
    ) -> Result<Vec<ConversationItem>, StorageError>;

    async fn delete_conversation(
        &self,
        conversation_id: &str,
        auth: &AuthContext,
    ) -> Result<(), StorageError>;
}
```

### 4.2 Conversation Endpoints (2 weeks)

Implement:
- `POST /v1/conversations` - Create conversation
- `GET /v1/conversations/{id}` - Get conversation
- `POST /v1/conversations/{id}/items` - Add items
- `GET /v1/conversations/{id}/items` - List items
- `DELETE /v1/conversations/{id}` - Delete conversation

---

## Phase 5: Security & Compliance (2 weeks)

### 5.1 Audit Logging (1 week)

**File**: `lib/llm/src/storage/audit.rs`

```rust
pub struct AuditLogger {
    backend: Arc<dyn AuditBackend>,
}

#[derive(Debug, Serialize)]
pub struct AuditEvent {
    pub timestamp: u64,
    pub tenant_id: String,
    pub user_id: String,
    pub action: String,
    pub resource_type: String,
    pub resource_id: String,
    pub success: bool,
    pub ip_address: Option<String>,
    pub user_agent: Option<String>,
}

#[async_trait]
pub trait AuditBackend: Send + Sync {
    async fn log_event(&self, event: AuditEvent) -> Result<(), AuditError>;
}
```

### 5.2 Rate Limiting (1 week)

**File**: `lib/llm/src/http/middleware/rate_limit.rs`

```rust
pub struct RateLimiter {
    store: Arc<dyn RateLimitStore>,
    limits: RateLimits,
}

pub struct RateLimits {
    pub max_responses_per_hour: u32,
    pub max_stored_responses: u32,
    pub max_storage_bytes: u64,
}

pub async fn rate_limit_middleware(
    State(limiter): State<Arc<RateLimiter>>,
    Extension(auth): Extension<AuthContext>,
    request: Request<Body>,
    next: Next,
) -> Result<Response, ErrorResponse> {
    limiter.check_limit(&auth).await?;
    Ok(next.run(request).await)
}
```

---

## Configuration Example

### Development (In-Memory)

```bash
# No auth (trusted environment)
DYN_AUTH_PROVIDER=none

# In-memory storage
DYN_STORAGE_PROVIDER=in_memory
DYN_STORAGE_MAX_RESPONSES=1000
DYN_STORAGE_DEFAULT_TTL_HOURS=1
```

### Production (Secure)

```bash
# API Key Auth
DYN_AUTH_PROVIDER=api_key
DYN_AUTH_API_KEY_STORE=redis
DYN_AUTH_REDIS_URL=rediss://auth-redis:6379
DYN_AUTH_REDIS_PASSWORD=<secret>

# Redis Storage
DYN_STORAGE_PROVIDER=redis
DYN_STORAGE_REDIS_URL=rediss://storage-redis:6379
DYN_STORAGE_REDIS_PASSWORD=<secret>
DYN_STORAGE_REDIS_TLS=true
DYN_STORAGE_MAX_RESPONSE_SIZE_MB=10
DYN_STORAGE_DEFAULT_TTL_HOURS=24
DYN_STORAGE_MAX_TTL_HOURS=168

# Encryption
DYN_STORAGE_ENCRYPTION=aes256gcm
DYN_STORAGE_ENCRYPTION_MASTER_KEY_PATH=/secrets/master.key

# Rate Limiting
DYN_RATE_LIMIT_RESPONSES_PER_HOUR=100
DYN_RATE_LIMIT_MAX_STORED_RESPONSES=1000
DYN_RATE_LIMIT_MAX_STORAGE_GB=10

# Audit Logging
DYN_AUDIT_BACKEND=file
DYN_AUDIT_LOG_PATH=/var/log/dynamo/audit.log
```

### Enterprise (Custom)

```rust
// Custom auth provider
let auth = Arc::new(MyLdapAuthProvider::new(ldap_config));

// Custom storage
let storage = Arc::new(MyPostgresStorage::new(db_pool));

// Build state with custom providers
let state = State::builder()
    .auth_provider(auth)
    .response_storage(storage)
    .build();
```

---

## Testing Strategy

### Unit Tests
- Auth provider trait implementations
- Storage provider trait implementations
- Permission checking logic
- Encryption/decryption

### Integration Tests
- End-to-end flow with in-memory providers
- Auth + storage interaction
- Tenant isolation validation
- TTL expiration

### Security Tests
- Cross-tenant access attempts
- Invalid token handling
- Storage quota enforcement
- Rate limit enforcement

### Load Tests
- Storage performance under load
- Redis connection pooling
- Encryption overhead
- Cache hit rates

---

## Migration Path

### Phase 1: Opt-In (v0.x)
- Stateful features disabled by default
- Require explicit config to enable
- Warning logs if auth is disabled

### Phase 2: Opt-Out (v1.x)
- Stateful features enabled by default
- Require explicit flag to disable
- Error if auth is not configured

### Phase 3: Always-On (v2.x)
- Stateful features always available
- Auth required for all endpoints

---

## Rollout Checklist

- [ ] Phase 1: Auth Framework (4 weeks)
  - [ ] Auth trait + providers
  - [ ] API key provider (default)
  - [ ] JWT provider
  - [ ] Auth middleware
  - [ ] Unit tests
  - [ ] Documentation

- [ ] Phase 2: Storage Backend (4 weeks)
  - [ ] Storage trait
  - [ ] Redis provider (default)
  - [ ] Postgres provider
  - [ ] Encryption provider
  - [ ] Unit tests
  - [ ] Performance tests

- [ ] Phase 3: Response Storage (3 weeks)
  - [ ] Update POST /v1/responses handler
  - [ ] Implement GET /v1/responses/{id}
  - [ ] Implement DELETE /v1/responses/{id}
  - [ ] Integration tests
  - [ ] Documentation

- [ ] Phase 4: Conversation API (3 weeks)
  - [ ] Conversation storage trait
  - [ ] Conversation endpoints
  - [ ] Integration tests
  - [ ] Documentation

- [ ] Phase 5: Security & Compliance (2 weeks)
  - [ ] Audit logging
  - [ ] Rate limiting
  - [ ] Quota management
  - [ ] Security audit
  - [ ] Compliance documentation

- [ ] Launch
  - [ ] Internal dogfooding
  - [ ] Beta release
  - [ ] GA release
  - [ ] Monitor metrics

---

## Open Questions

1. **Default TTL**: Should responses expire after 24h or 7d by default?
2. **Storage Limits**: What per-user/tenant quotas are reasonable?
3. **Conversation Scope**: Should conversations be scoped to models or global?
4. **Sharing**: Do we need response/conversation sharing between users?
5. **Versioning**: How to handle schema migrations for stored responses?
6. **Backup**: Should we provide export/import for conversations?

---

## Success Metrics

- **Adoption**: % of requests using `previous_response_id`
- **Performance**: Latency impact of storage lookups
- **Security**: Zero cross-tenant access violations
- **Reliability**: Storage backend uptime > 99.9%
- **Cost**: Storage cost per 1M responses

---

## References

- [OpenAI Responses API Spec](https://platform.openai.com/docs/api-reference/responses)
- [OpenResponses Compliance Tests](https://www.openresponses.org/compliance)
- [Linear Issue DYN-2045](https://linear.app/nvidia/issue/DYN-2045)
- [Dynamo Architecture Docs](./architecture.md)
