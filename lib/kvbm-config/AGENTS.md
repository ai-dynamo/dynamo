# AI Agent Instructions for kvbm-config

This file provides instructions for AI coding assistants (Claude Code, GitHub Copilot, Cursor, etc.) working on the kvbm-config crate.

## Critical: Keep Documentation Synchronized

When modifying configuration defaults or adding new config options, you **must** update the following files:

1. **`kvbm.example.toml`** - Update with new defaults or add commented-out options
2. **`kvbm.example.json`** - Update to match TOML changes
3. **`README.md`** - Update the configuration reference tables

### Sample Config Convention

- **Active defaults**: Shown uncommented with their default values
- **Optional/disabled configs**: Shown commented out with example values
- Users should be able to copy the sample and have it work with sensible defaults

## Cross-Reference Files

When changing defaults, also check these files for consistency:

| File | Purpose |
|------|---------|
| `src/lib.rs` | Top-level KvbmConfig struct and Figment loading |
| `src/tokio.rs` | TokioConfig defaults |
| `src/rayon.rs` | RayonConfig defaults |
| `src/nova.rs` | NovaConfig and NovaBackendConfig defaults |
| `src/nixl.rs` | NixlConfig defaults (UCX, POSIX backends) |
| `src/cache.rs` | CacheConfig, HostCacheConfig, DiskCacheConfig defaults |
| `src/offload.rs` | OffloadConfig and policy defaults |
| `src/object.rs` | ObjectConfig and S3ObjectConfig defaults |
| `src/discovery.rs` | DiscoveryConfig variants and defaults |
| `../kvbm/src/v2/integrations/connector/leader/init.rs` | **Runtime defaults** for offload policies when config is empty |

## Offload Policy Defaults

The offload policy defaults are applied at **runtime** in `leader/init.rs`, not in the config structs:

- **G1→G2** (GPU→Host): `["presence"]` - Prevents duplicate transfers
- **G2→G3** (Host→Disk): `["presence_lfu"]` with `min_lfu_count = 8` - Only offloads frequently-used blocks

If you change these runtime defaults, update the sample configs to match.

## Enum Serialization Format

Tagged enums use `#[serde(tag = "type")]` for JSON/TOML serialization:

| Config | Tag Field | Example JSON |
|--------|-----------|--------------|
| `DiscoveryConfig` | `"type"` | `{"type": "filesystem", "path": "..."}` |
| `ObjectClientConfig` | `"type"` | `{"type": "s3", "bucket": "..."}` |
| `NixlObjectConfig` | `"backend"` | `{"type": "nixl", "backend": "s3", ...}` |

**Important**: Do NOT use nested format like `{"s3": {...}}`. Always use the tag format.

## Profile-Based Configuration

vLLM uses `leader` and `worker` profiles as top-level JSON keys:

```json
{
  "leader": { /* leader-specific config */ },
  "worker": { /* worker-specific config */ },
  "default": { /* shared config */ }
}
```

Example configs should demonstrate this pattern since it's the primary vLLM integration format.

## Validation Rules

When adding new config fields, ensure validation is added:

- Use `#[validate(range(min = X, max = Y))]` for numeric bounds
- Use `#[serde(default = "default_fn")]` for default values
- Add tests in the module's `#[cfg(test)]` section
