<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Rebuilding and modifying python-soxr 1.1.0

This directory accompanies the `soxr` Python distribution installed in this
container. The distributed wheel's native extension statically incorporates
libsoxr, including its PFFFT implementation. Python-soxr and libsoxr are
licensed under LGPL-2.1-or-later; PFFFT's license and attribution are included
in the source archive and in `/legal/NOTICES-Python.txt`.

The directory contains the official PyPI source distribution:

```text
soxr-1.1.0.tar.gz
SHA256: 9f228ae21c78fa9359ca98d8a5e8e91f30639e438e574133dace62c5b5309e44
Source: https://pypi.org/project/soxr/1.1.0/
```

The archive includes the Python/C++ binding, the complete bundled libsoxr
source under `libsoxr/`, the PFFFT sources under `libsoxr/src/`, license texts,
and the CMake build configuration used to relink them into the extension.

## Build the bundled version

These commands require Python 3.12 or later, a C/C++ toolchain, CMake, and an
internet connection for the build requirements declared by `pyproject.toml`.

```bash
sha256sum -c <<'EOF'
9f228ae21c78fa9359ca98d8a5e8e91f30639e438e574133dace62c5b5309e44  soxr-1.1.0.tar.gz
EOF
tar -xzf soxr-1.1.0.tar.gz
python3 -m venv rebuild-venv
rebuild-venv/bin/python -m pip install --upgrade pip
rebuild-venv/bin/python -m pip wheel --no-deps --wheel-dir dist ./soxr-1.1.0
```

The resulting wheel contains a newly linked `soxr/soxr_ext.abi3.so`. To replace
the installed component in a compatible container or virtual environment:

```bash
python3 -m pip install --force-reinstall --no-deps dist/soxr-1.1.0-*.whl
```

## Patch and relink libsoxr or PFFFT

Extract the archive, edit or replace the LGPL libsoxr implementation under
`soxr-1.1.0/libsoxr/` (PFFFT is under `libsoxr/src/`), and run the bundled-build
commands above. No Dynamo source change is required: rebuilding the wheel
recompiles the Python binding and relinks it against the modified library.

To verify that the replacement is active:

```bash
python3 - <<'PY'
import soxr
import soxr.soxr_ext as soxr_ext

print(soxr.__version__)
print(soxr_ext.__file__)
PY
```

## Build against a shared system libsoxr

As an alternative to the bundled static library, install development headers
for a compatible libsoxr and enable python-soxr's system-library option:

```bash
CMAKE_ARGS='-DUSE_SYSTEM_LIBSOXR=ON' \
  rebuild-venv/bin/python -m pip wheel --no-deps --wheel-dir dist ./soxr-1.1.0
```

For Debian or Ubuntu, the development package is normally `libsoxr-dev`. The
resulting extension resolves `libsoxr.so` at runtime, allowing an
ABI-compatible modified shared library to be substituted without rebuilding
the Python extension. Inspect the wheel and confirm the result with `ldd`:

```bash
rm -rf wheel-root
mkdir wheel-root
unzip -q dist/soxr-1.1.0-*.whl -d wheel-root
ldd wheel-root/soxr/soxr_ext*.so | grep soxr
```

The complete LGPL-2.1 terms and component notices are also installed with the
wheel under `soxr-1.1.0.dist-info/licenses/`.
