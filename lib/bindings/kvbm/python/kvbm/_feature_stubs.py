# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Feature stub utilities for graceful handling of optional Rust feature flags.

When a Rust feature (v1, v2, kernels) is not compiled, these stubs provide
informative error messages telling users how to rebuild with the required feature.
"""


def _make_feature_stub(class_name: str, feature: str):
    """
    Create a stub callable that raises ImportError with feature flag info.

    This is used to replace classes/functions that require a specific Cargo feature
    that wasn't compiled. When users try to instantiate the stub, they get a helpful
    error message explaining which feature flag they need.

    Args:
        class_name: Name of the class/function being stubbed
        feature: The Cargo feature flag required (e.g., "v1", "v2", "kernels")

    Returns:
        A callable that raises ImportError with instructions when called
    """

    def _raise(*args, **kwargs):
        raise ImportError(
            f"'{class_name}' requires the '{feature}' feature. "
            f"Rebuild kvbm with: maturin develop --features {feature}"
        )

    # Make it look like the original class for better error messages
    _raise.__name__ = class_name
    _raise.__qualname__ = class_name
    _raise.__doc__ = f"Stub for {class_name} - requires '{feature}' feature flag"
    return _raise


def _make_module_stub(module_name: str, feature: str):
    """
    Create a stub module object that raises ImportError on attribute access.

    This is used to replace submodules (like v2 or kernels) that require a specific
    Cargo feature. Any attribute access on the stub raises an informative error.

    Args:
        module_name: Name of the module being stubbed
        feature: The Cargo feature flag required

    Returns:
        A stub object that raises ImportError on any attribute access
    """

    class ModuleStub:
        def __getattr__(self, name):
            raise ImportError(
                f"'{module_name}.{name}' requires the '{feature}' feature. "
                f"Rebuild kvbm with: maturin develop --features {feature}"
            )

        def __repr__(self):
            return f"<stub module '{module_name}' - requires '{feature}' feature>"

    return ModuleStub()
