# Maturin Develop Issue and Fix

## Problem

When running `maturin develop`, code changes weren't being reflected in the Python frontend even though the build completed successfully.

## Root Cause

Maturin was building and updating `_core.abi3.so`, but Python was loading the older `_core.cpython-312-x86_64-linux-gnu.so` file. When both files exist, Python prefers the platform-specific one (cpython-312) over the generic ABI3 version.

## Files Involved

- `/home/ubuntu/dynamo/lib/bindings/python/src/dynamo/_core.abi3.so` - Updated by `maturin develop`
- `/home/ubuntu/dynamo/lib/bindings/python/src/dynamo/_core.cpython-312-x86_64-linux-gnu.so` - Loaded by Python

## Solution

Create a symlink so the platform-specific filename points to the ABI3 version:

```bash
cd /home/ubuntu/dynamo/lib/bindings/python/src/dynamo
rm _core.cpython-312-x86_64-linux-gnu.so
ln -s _core.abi3.so _core.cpython-312-x86_64-linux-gnu.so
```

After this fix:
- `maturin develop` updates `_core.abi3.so`
- Python loads via the symlink, so it always gets the latest build
- No manual copying needed!

## Verification

```bash
# Build
cd /home/ubuntu/dynamo/lib/bindings/python
maturin develop --uv

# Verify symlink
ls -lh src/dynamo/_core*.so
# Should show: _core.cpython-312-x86_64-linux-gnu.so -> _core.abi3.so

# Test
python -c "from dynamo import _core; print(_core.__file__)"
# Should show: .../src/dynamo/_core.cpython-312-x86_64-linux-gnu.so
```

## Why This Happened

The old `_core.cpython-312-x86_64-linux-gnu.so` file (from Oct 16) was created during the initial devcontainer setup, possibly by an older version of maturin or a different build process. Since then, `maturin develop` has been updating only the ABI3 version, but the old platform-specific file remained and took precedence.

## Future Prevention

**This fix should be added to the devcontainer `post-create.sh` script** to ensure all developers have the correct setup:

```bash
# After maturin develop runs
cd /home/ubuntu/dynamo/lib/bindings/python/src/dynamo
if [ -f "_core.cpython-312-x86_64-linux-gnu.so" ] && [ ! -L "_core.cpython-312-x86_64-linux-gnu.so" ]; then
    echo "Fixing maturin develop: creating symlink for platform-specific .so"
    rm -f _core.cpython-312-x86_64-linux-gnu.so
    ln -s _core.abi3.so _core.cpython-312-x86_64-linux-gnu.so
fi
```

## Alternative Solutions Considered

1. **Delete the platform-specific file**: Would work, but Python might prefer it if recreated
2. **Configure maturin to use platform-specific builds**: Would require pyproject.toml changes
3. **Symlink (chosen)**: Simplest, works with existing setup, transparent to maturin and Python

