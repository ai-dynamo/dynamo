# Python Guidelines

Rules and conventions for Python code in this repository.

## Critical Rules

These are the conventions that matter most for code quality and maintainability.
Exceptions exist, but they should be rare and justified in a code comment.

### Keep imports at the top of the file

Avoid importing inside functions because it hides dependencies, makes the module
harder to understand at a glance, and can mask missing packages until a specific
code path is hit at runtime.

Avoid wrapping imports in `try/except` because it creates two execution modes --
one with the library and one without -- which doubles the testing surface and
produces confusing behavior when the dependency is unexpectedly absent.

```python
# BAD -- import inside function; dependency is invisible until this path runs
def process_data():
    import json
    return json.loads(data)

# BAD -- try/except hides a missing dependency behind a flag
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

# GOOD -- direct import at module top; fails immediately if missing
import json
import requests
```

Failing immediately on a missing dependency is better than hiding the problem
until a user hits an obscure code path in production. If a package is required,
add it to `requirements.txt` or `pyproject.toml` so it's installed upfront.

### Prefer failing fast over hiding errors

From PEP 20 (The Zen of Python):

> *"Errors should never pass silently. Unless explicitly silenced."*
> *"Explicit is better than implicit."*

Failing immediately when something goes wrong is better than silently continuing
with bad state, because:

- The person who caused the error sees it right away, while the context is fresh.
- The stack trace points directly at the root cause, not at a downstream symptom
  three function calls later.
- Hidden errors compound -- a swallowed exception in one place produces confusing
  behavior somewhere else, and the debugging cost grows exponentially with distance
  from the original failure.

```python
# BAD -- all of these hide errors
except Exception:
    pass

except Exception as e:
    logging.error(e)        # logs but silently continues!
    return []               # returns a fake default

# BAD -- bare except catches KeyboardInterrupt and SystemExit too,
# making the process impossible to kill with Ctrl-C or sys.exit()
try:
    do_work()
except:
    log_error()

# GOOD -- just let it crash
result = something()

# GOOD -- catch SPECIFIC exceptions you can actually handle
try:
    result = json.loads(text)
except json.JSONDecodeError:
    result = {}

# GOOD -- if you must catch broad, catch Exception (not bare except)
# and re-raise after logging
try:
    result = something()
except Exception as e:
    logger.error(f"Failed: {e}")
    raise
```

**Three rules:**
1. Remove the try/except if possible -- let it crash.
2. Catch **specific** exceptions only (`FileNotFoundError`, `ValueError`, `json.JSONDecodeError`, etc.).
3. If you must catch broadly, use `except Exception:` (never bare `except:`) and **always** re-raise after logging.

### NO defensive `getattr()` on known types

When an object has a known type with known attributes, access them directly.
Using `getattr()` with a default hides bugs by silently returning a fallback
when the attribute should always exist.

```python
# BAD -- node has a job_name attribute, getattr hides AttributeError
name = getattr(node, "job_name", "")

# GOOD -- direct access, fails loudly if something is wrong
name = node.job_name
```

---

## Common Pitfalls

### Mutable default arguments

Default argument values are evaluated once at function definition time, not on
each call. A mutable default (list, dict, set) is shared across all invocations.

```python
# BAD -- the list is shared across all calls
def add_item(item, items=[]):
    items.append(item)
    return items

add_item("a")  # ["a"]
add_item("b")  # ["a", "b"] -- not ["b"]!

# GOOD
def add_item(item, items=None):
    if items is None:
        items = []
    items.append(item)
    return items
```

### Always use context managers for resources

Files, network connections, subprocesses, and locks should be opened with `with`
so they are released even if an exception occurs.

```python
# BAD -- file handle leaks if json.load raises
f = open("data.json")
data = json.load(f)
f.close()

# GOOD
with open("data.json") as f:
    data = json.load(f)
```

This is especially important in this project where tests manage subprocesses,
etcd/NATS connections, and temp directories. Use `ManagedProcess`,
`tempfile.TemporaryDirectory`, and similar context managers rather than
manual setup/teardown.

### Do not shadow built-in names

Naming a variable `list`, `dict`, `id`, `type`, `input`, `open`, or `format`
overwrites the built-in and causes confusing errors later in the same scope.

```python
# BAD
list = get_items()          # shadows built-in list()
filtered = list(some_gen)   # TypeError: 'list' object is not callable

# GOOD
items = get_items()
filtered = list(some_gen)
```

### Use `is` for None / True / False comparisons

`==` invokes `__eq__`, which can be overridden. `is` checks identity, which is
what you want for singletons.

```python
# BAD
if result == None:
if flag == True:

# GOOD
if result is None:
if flag is True:     # or just: if flag:
```

### Do not modify a collection while iterating

Adding or removing items during iteration causes skipped elements or
`RuntimeError`. Build a new collection or iterate over a copy.

```python
# BAD -- skips elements
for item in items:
    if item.is_stale():
        items.remove(item)

# GOOD
items = [item for item in items if not item.is_stale()]
```

### Prefer `join()` over string concatenation in loops

Repeated `+=` on strings creates a new string object each time, which is O(n^2)
for large loops.

```python
# BAD
result = ""
for line in lines:
    result += line + "\n"

# GOOD
result = "\n".join(lines)
```

### Watch out for late-binding closures in loops

Closures capture the variable reference, not its value at the time of creation.

```python
# BAD -- all lambdas return 4 (the final value of i)
fns = [lambda: i for i in range(5)]
[f() for f in fns]  # [4, 4, 4, 4, 4]

# GOOD -- default argument captures current value
fns = [lambda i=i: i for i in range(5)]
[f() for f in fns]  # [0, 1, 2, 3, 4]
```

### Do not use `assert` for runtime validation

Assertions are stripped when Python runs with `-O` (optimize). Use explicit
`if/raise` for validation that must always execute.

```python
# BAD -- silently skipped under python -O
assert user_id is not None, "user_id required"

# GOOD
if user_id is None:
    raise ValueError("user_id required")
```

---

## Code Style

- Follow PEP 8.
- `snake_case` for variables and functions.
- `PascalCase` for classes.
- Add type hints where they improve readability.
- Use docstrings for public functions and classes.
- Use `dataclass` instead of plain dicts when a structure has >4 fields (better
  type inference and IDE support).

## File Organization

- `__init__.py` for package initialization.
- Clear module separation.
- Tests in `tests/` directory.

## Formatting and Linting

### Preferred workflow

Auto-fix formatting, then lint:

```bash
ruff format <touched_paths>
ruff check --fix <touched_paths>
```

Or use pre-commit (runs isort, black, flake8, ruff, and more):

```bash
pre-commit run --files <touched_files>
pre-commit run --all-files    # for broad changes
```

### Pre-commit hooks

The repo's `.pre-commit-config.yaml` runs these Python hooks:

- **isort** -- import sorting (`profile = "black"`, configured in `pyproject.toml`)
- **black** -- code formatting
- **flake8** -- style checks (`max-line-length=88`)
- **ruff** -- fast linting with auto-fix
- **codespell** -- spelling checks
- **trailing-whitespace**, **end-of-file-fixer**, **check-yaml**, **check-json**, **check-toml**

### Before committing

Always run:

```bash
pre-commit run --files <changed_files>
```

For broader changes or if in doubt:

```bash
pre-commit run --all-files
```

### Indentation verification

Indentation errors are common and hard to spot visually. After editing Python
files, verify mechanically:

```bash
ruff format <touched_paths>                     # auto-fix (preferred)
python3 -m compileall -q <touched_paths>        # fast parse-only check
```

When fixing indentation errors, always read 20-30 lines of surrounding context
and fix the whole block -- adjacent lines often share the same mistake.

## Error Handling

See the **Critical Rules** section above for the full policy. Summary:

- Let exceptions propagate by default.
- Catch only specific exceptions you can actually handle.
- If you catch `Exception`, you must re-raise after logging.
- No `except Exception: pass`. Ever.

### Regex caution

Be careful with escaping in raw strings (`\s` vs `\\s`). When changing a critical
regex, add a one-line test to prove it matches.

## License Headers

Every `.py` file must have the SPDX header:

```python
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
```

## Import Order

Imports are sorted by isort with `profile = "black"` (configured in `pyproject.toml`).
The order is:

1. Standard library
2. Third-party (known: `vllm`, `tensorrt_llm`, `sglang`, `aiconfigurator`)
3. First-party (`dynamo`, `deploy`)

Run `isort` or `pre-commit run isort` to auto-fix ordering.
