# sccache Setup for Dynamo Project

This guide explains how to use sccache to speed up local Rust compilation in the Dynamo project.

## What is sccache?

sccache is a compiler cache that stores compilation results to speed up subsequent builds. It can significantly reduce build times, especially for incremental builds and when switching between branches.

## Setup (Already Completed)

The sccache setup has been configured for this project:

1. ✅ **sccache binary**: Installed to `~/.local/bin/sccache`
2. ✅ **Cargo configuration**: `.cargo/config.toml` configured to use sccache as rustc wrapper
3. ✅ **Local cache**: Cache directory set to `~/.cache/sccache` with 10GB limit
4. ✅ **Management script**: `setup-sccache.sh` for easy management

## Quick Start

### 1. Apply Environment Settings
```bash
# Option 1: Source the updated bashrc
source ~/.bashrc

# Option 2: Apply environment for current session
source <(./setup-sccache.sh env)
```

### 2. Start sccache Server
```bash
./setup-sccache.sh start
```

### 3. Build Your Project
```bash
# Regular cargo commands will now use sccache automatically
cargo build
cargo build --release
cargo check
cargo test
```

### 4. Check Statistics
```bash
./setup-sccache.sh stats
```

## Management Commands

The `setup-sccache.sh` script provides several useful commands:

```bash
# Show current status
./setup-sccache.sh status

# Start sccache server
./setup-sccache.sh start

# Stop sccache server
./setup-sccache.sh stop

# Show compilation statistics
./setup-sccache.sh stats

# Clear cache and reset statistics
./setup-sccache.sh clear

# Show environment variables
./setup-sccache.sh env

# Show help
./setup-sccache.sh help
```

## Current Performance

Based on the initial test build:
- **Cache hits**: 2 (0.17% hit rate)
- **Cache misses**: 1,189 (first build, expected)
- **Cache size**: 231 MiB / 10 GiB limit
- **Compile requests**: 1,494 total

## Expected Benefits

### First Build
- Minimal speedup (cache is being populated)
- All compilations are cache misses

### Subsequent Builds
- **Incremental builds**: 50-90% faster
- **Clean rebuilds**: 30-70% faster
- **Branch switching**: Significant speedup for unchanged dependencies

### Best Performance Scenarios
1. **Incremental development**: Making small changes and rebuilding
2. **Branch switching**: When dependencies haven't changed
3. **CI/CD**: When using shared cache (can be configured later)
4. **Team development**: When multiple developers work on similar code

## Configuration Details

### Cache Location
- **Directory**: `~/.cache/sccache`
- **Size limit**: 10 GB
- **Type**: Local disk cache

### Cargo Integration
- **Rustc wrapper**: Configured in `.cargo/config.toml`
- **Automatic**: No need to modify build commands
- **Compatible**: Works with all cargo commands

## Troubleshooting

### If builds seem slow
```bash
# Check if sccache is running
./setup-sccache.sh status

# Check statistics for cache hit rate
./setup-sccache.sh stats
```

### If sccache isn't working
```bash
# Restart the server
./setup-sccache.sh stop
./setup-sccache.sh start

# Verify environment
./setup-sccache.sh env
```

### Clear cache if needed
```bash
# Clear cache and statistics
./setup-sccache.sh clear
```

## Advanced Configuration

### Increase Cache Size
Edit the script or set environment variable:
```bash
export SCCACHE_CACHE_SIZE="20G"
```

### Use with Different Linkers
The `.cargo/config.toml` includes commented options for faster linkers like `lld` or `mold`:
```toml
[target.x86_64-unknown-linux-gnu]
linker = "clang"
rustflags = ["-C", "link-arg=-fuse-ld=lld"]
```

### Remote Cache (Future)
sccache supports remote caches (S3, Redis, etc.) for team sharing. This can be configured later if needed.

## Tips for Maximum Benefit

1. **Keep sccache running**: Start it once and leave it running
2. **Don't clear cache unnecessarily**: Let it build up over time
3. **Use incremental builds**: Prefer `cargo check` during development
4. **Monitor statistics**: Check hit rates to verify effectiveness

## Integration with Development Workflow

```bash
# Start of development session
./setup-sccache.sh start

# During development (these will be faster after first build)
cargo check          # Fast syntax/type checking
cargo test           # Run tests
cargo build          # Debug build
cargo build --release # Release build

# End of session (optional)
./setup-sccache.sh stats  # See how much time was saved
```

The cache will persist between sessions, so subsequent development sessions will benefit from previously cached compilations.
