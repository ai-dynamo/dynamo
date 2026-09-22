# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for push_egress.py — the inverted push-egress path for TRT-LLM workers.

These tests are designed to BREAK the implementation; a passing run that finds
nothing is a failed deliverable.  Every test targets a specific claim in the
module docstring or an edge-case the implementation could silently get wrong.

Import strategy
---------------
These tests must run without the compiled bindings, so nothing here imports
dynamo._core.  push_egress.py is loaded by path via importlib, and the Rust
ResponseSender is modelled by :class:`FakeSender`, which enforces the contract
documented on the real one.  Consequences worth knowing: the actual Python/Rust
boundary is NOT exercised here, and neither is backpressure -- the real ``send``
blocks and releases the GIL, which a fake cannot reproduce.

Class-body scoping note
-----------------------
Python class bodies do NOT form closures over enclosing function locals, so a
class defined inside a test method may reference only module-level names (the
decorators and helpers above).  Anything test-local must be passed in or bound
at module scope instead, or the class body raises NameError.
"""

import ast
import asyncio
import functools
import importlib.util
import inspect
import logging
import pathlib
from typing import ClassVar

import pytest

# ---------------------------------------------------------------------------
# Load the module under test by path.
# ---------------------------------------------------------------------------

# tests/ -> trtllm/ -> dynamo/ -> src/ -> components/ -> repo root. Derived, not
# hardcoded: these tests must run from any checkout, worktree, or container path.
_COMPONENTS_SRC = pathlib.Path(__file__).resolve().parents[3]


def _load_push_egress():
    """Load push_egress.py directly, without importing the dynamo package.

    It depends on nothing from dynamo, so a plain spec load keeps these tests
    runnable wherever the compiled bindings are absent.
    """
    spec = importlib.util.spec_from_file_location(
        "dynamo.trtllm.request_handlers.push_egress",
        _COMPONENTS_SRC / "dynamo" / "trtllm" / "request_handlers" / "push_egress.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pe = _load_push_egress()

drive_push_egress = pe.drive_push_egress
drive_push_egress_stream = pe.drive_push_egress_stream
push_egress_capable = pe.push_egress_capable

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def run(coro):
    """Run a coroutine synchronously (no pytest-asyncio required)."""
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Fake ResponseSender — models the Rust contract.
# ---------------------------------------------------------------------------


class FakeSender:
    """Behaves like the real Rust ResponseSender.

    Contract enforced:
    - send() raises after close / close_with_error.
    - close() and close_with_error() are idempotent.
    - close() after close_with_error() is a no-op.
    """

    def __init__(self):
        self.sent = []
        self.close_count = 0
        self.error_close_count = 0
        self.last_error = None
        self._closed = False

    def send(self, obj):
        if self._closed:
            raise RuntimeError("send after close")
        self.sent.append(obj)

    def close(self):
        if self._closed:
            return
        self._closed = True
        self.close_count += 1

    def close_with_error(self, msg: str):
        if self._closed:
            return
        self._closed = True
        self.error_close_count += 1
        self.last_error = msg

    @property
    def total_closes(self):
        return self.close_count + self.error_close_count


class BrokenSender(FakeSender):
    """A sender whose close operations raise — tests Python degrades sanely."""

    def close(self):
        raise RuntimeError("close exploded")

    def close_with_error(self, msg: str):
        raise RuntimeError(f"close_with_error exploded: {msg}")


# ---------------------------------------------------------------------------
# Async helpers
# ---------------------------------------------------------------------------


async def gen(*items):
    for item in items:
        yield item


async def raising_gen(items, exc):
    for item in items:
        yield item
    raise exc


# ---------------------------------------------------------------------------
# Module-level handler classes (class bodies can't see test-local vars).
# ---------------------------------------------------------------------------


async def _gen_one_token(self, request, context):
    yield {"token": "a"}


async def _gen_three_items(self, request, context):
    for v in [1, 2, 3]:
        yield v


async def _gen_empty(self, request, context):
    return
    yield  # make it an async generator


# ---------------------------------------------------------------------------
# Fixture: reset the module-global warning guard between tests.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_logged_no_sender():
    pe._logged_no_sender = False
    yield
    pe._logged_no_sender = False


# ===========================================================================
# 1. Signature claim
# ===========================================================================


class TestSignatureClaim:
    """The Rust opt-in check calls inspect.signature(handler).parameters.

    If response_sender is missing from the outermost signature, every TRT-LLM
    endpoint silently falls back to the pull path and the whole change is a no-op.
    """

    def test_response_sender_in_signature(self):
        decorated = push_egress_capable(_gen_one_token)
        params = inspect.signature(decorated).parameters
        assert "response_sender" in params, (
            "push_egress_capable did not expose response_sender; "
            "Rust will never select push mode"
        )

    def test_no_wrapped_link(self):
        """inspect.signature() follows __wrapped__; it must be deleted."""
        decorated = push_egress_capable(_gen_one_token)
        assert not hasattr(decorated, "__wrapped__"), (
            "__wrapped__ present; inspect.signature() will follow it and hide "
            "response_sender, silently disabling push mode for every endpoint"
        )

    def test_response_sender_in_signature_bound_method(self):
        """Test on a bound method — the actual shape Rust inspects."""

        class Handler:
            generate = push_egress_capable(_gen_one_token)

        h = Handler()
        params = inspect.signature(h.generate).parameters
        assert "response_sender" in params, (
            "response_sender hidden when inspecting a bound method; "
            "Rust handler_supports_push() will return false"
        )

    def test_outer_functools_wraps_still_exposes_response_sender(self):
        """Signature visibility is NOT what makes the ordering constraint real.

        A wrapper applied on top with ``functools.wraps`` sets its
        ``__wrapped__`` to ``dispatch`` — which declares ``response_sender`` —
        so ``inspect.signature`` still finds the parameter and Rust still picks
        push mode. Pinned here so nobody "fixes" this into a hiding assertion:
        the reason ``push_egress_capable`` must be outermost is that any
        decorator inspecting what it wraps has to see a real async-generator
        function, not the plain ``def dispatch`` this returns. The visibility
        failure that *is* real is the ``__wrapped__`` link on ``dispatch``
        itself — see :meth:`test_no_wrapped_link` and the canary in
        :class:`TestRustPathSelection`.
        """
        inner = push_egress_capable(_gen_one_token)

        @functools.wraps(inner)
        def outer_wrapper(self, request, context=None, **kwargs):
            return inner(self, request, context, **kwargs)

        params = inspect.signature(outer_wrapper).parameters
        assert "response_sender" in params
        assert not inspect.isasyncgenfunction(inner), (
            "push_egress_capable returned an async-generator function; a "
            "decorator applied outside it would then see the wrong shape and "
            "the OUTERMOST constraint would need restating"
        )


# ===========================================================================
# 2. Shape claim
# ===========================================================================


class TestShapeClaim:
    """Both push and pull paths must return an async generator (has __anext__)."""

    def _make_push_result(self, sender):
        """Call the decorated function directly with a dummy self."""
        decorated = push_egress_capable(_gen_one_token)
        dummy_self = object.__new__(object)
        return decorated(dummy_self, request={}, context=None, response_sender=sender)

    def _make_pull_result(self):
        decorated = push_egress_capable(_gen_one_token)
        dummy_self = object.__new__(object)
        return decorated(dummy_self, request={}, context=None)

    def test_push_mode_returns_async_generator(self):
        result = self._make_push_result(FakeSender())
        assert hasattr(result, "__anext__"), (
            "push mode returned something without __anext__; "
            "Rust getattr('__anext__') will fail every request"
        )
        assert inspect.isasyncgen(result)

    def test_pull_mode_returns_async_generator(self):
        result = self._make_pull_result()
        assert hasattr(
            result, "__anext__"
        ), "pull mode fallback returned something without __anext__"

    def test_drive_push_egress_stream_is_async_generator_function(self):
        assert inspect.isasyncgenfunction(drive_push_egress_stream), (
            "drive_push_egress_stream is not an async generator function; "
            "calling it returns a coroutine, not an async generator"
        )

    def test_push_generator_advanced_exactly_once_then_stops(self):
        """Rust advances the push generator ONCE per request.

        After that single __anext__ call the generator must raise
        StopAsyncIteration, having run the whole request to completion.
        """
        sender = FakeSender()
        chunks = ["a", "b", "c"]

        async def inner_gen(self, request, context):
            for c in chunks:
                yield c

        async def _test():
            decorated = push_egress_capable(inner_gen)
            dummy = object.__new__(object)
            g = decorated(dummy, request={}, context=None, response_sender=sender)
            try:
                await g.__anext__()
                pytest.fail(
                    "drive_push_egress_stream yielded a value; "
                    "responses must go through sender.send(), not via yield"
                )
            except StopAsyncIteration:
                pass

        run(_test())
        assert sender.sent == chunks

    def test_push_generator_yields_nothing(self):
        """The push generator must never yield items to the Rust driver."""
        sender = FakeSender()

        async def inner_gen(self, request, context):
            yield "response"

        async def _collect():
            decorated = push_egress_capable(inner_gen)
            dummy = object.__new__(object)
            g = decorated(dummy, request={}, context=None, response_sender=sender)
            return [item async for item in g]

        items = run(_collect())
        assert items == [], f"push generator yielded {items}; must go through sender"


# ===========================================================================
# 3. Termination: close() / close_with_error() exactly once per request
# ===========================================================================


class TestTermination:
    """Normal end closes the sender; every failure propagates to Rust instead.

    The split matters downstream. Rust runs an escaping exception through
    `map_python_exception`, which turns `EngineShutdown` into
    `BackendError::EngineShutdown` (request migration + worker inhibition) and
    `ValueError`/`TypeError` into `InvalidArgument`. Catching it here to report
    a string through `close_with_error` would erase that type, so these tests
    assert the exception gets OUT, not that a message was formatted.
    """

    def test_normal_completion_closes_sender_exactly_once(self):
        sender = FakeSender()
        run(drive_push_egress(gen("a", "b", "c"), sender))
        assert sender.close_count == 1, f"expected 1 close(), got {sender.close_count}"
        assert sender.error_close_count == 0

    def test_zero_chunk_stream_closes_sender(self):
        sender = FakeSender()
        run(drive_push_egress(gen(), sender))
        assert (
            sender.close_count == 1
        ), f"empty stream: expected 1 close, got {sender.close_count}"
        assert sender.error_close_count == 0

    @pytest.mark.parametrize(
        "exc",
        [
            ValueError("bad request"),  # -> BackendError::InvalidArgument
            RuntimeError("engine shutting down"),  # stands in for EngineShutdown
            TimeoutError("too slow"),  # -> BackendError::ConnectionTimeout
        ],
        ids=["value-error", "runtime-error", "timeout-error"],
    )
    def test_handler_exception_propagates_untouched(self, exc):
        """The exact exception object must reach Rust, unwrapped and unlogged-over.

        This is the whole typed-error contract: if this ever goes back to being
        caught, every failure reaches the caller as a generic server error and
        an interrupted request stops being migrated to another worker.
        """
        sender = FakeSender()
        with pytest.raises(type(exc)) as caught:
            run(drive_push_egress(raising_gen(["a"], exc), sender))
        assert caught.value is exc, "the original exception must not be re-wrapped"

    def test_handler_exception_leaves_the_stream_to_rust(self):
        """No close of any kind on the error path.

        Closing here would win the race against the Rust driver's
        `close_with_dynamo_error`, which is idempotent — the typed frame would
        be silently dropped in favour of a plain end-of-stream.
        """
        sender = FakeSender()
        with pytest.raises(ValueError):
            run(drive_push_egress(raising_gen(["a"], ValueError("boom")), sender))
        assert sender.total_closes == 0, (
            "the error path must not terminate the stream; Rust does it with "
            f"the mapped backend error (got {sender.total_closes} closes)"
        )

    def test_cancelled_error_propagates(self):
        """Cancellation is an exception like any other.

        `map_python_exception` maps it to `BackendError::Cancelled`, matching
        what the pull path reports for a cancelled generator.
        """
        sender = FakeSender()

        async def cancel_mid_stream():
            yield "first"
            raise asyncio.CancelledError()

        with pytest.raises(asyncio.CancelledError):
            run(drive_push_egress(cancel_mid_stream(), sender))
        assert sender.sent == ["first"], "chunks before the cancel must be delivered"
        assert sender.total_closes == 0

    def test_keyboard_interrupt_propagates(self):
        sender = FakeSender()

        async def ki_stream():
            yield "x"
            raise KeyboardInterrupt()

        with pytest.raises(KeyboardInterrupt):
            run(drive_push_egress(ki_stream(), sender))
        assert sender.total_closes == 0

    def test_at_most_one_close_total(self):
        sender = FakeSender()
        run(drive_push_egress(gen("x"), sender))
        assert (
            sender.total_closes == 1
        ), f"stream closed {sender.total_closes} times (expected exactly 1)"

    def test_send_raising_propagates(self):
        """A failing `send` is how the consumer-gone case surfaces.

        The real sender raises `RuntimeError` once the consumer drops the
        response stream, having already stopped the request context. Letting it
        out unwinds the handler — and its `finally` blocks — instead of looping
        on a dead channel.
        """

        class SenderFailsOnSecond(FakeSender):
            def send(self, obj):
                if len(self.sent) >= 1:
                    raise RuntimeError("consumer dropped the response stream")
                super().send(obj)

        sender = SenderFailsOnSecond()
        with pytest.raises(RuntimeError, match="consumer dropped"):
            run(drive_push_egress(gen("a", "b", "c"), sender))
        assert sender.sent == ["a"]
        assert sender.total_closes == 0

    def test_failing_close_propagates(self):
        """A `close()` that raises is a bridge failure, not something to hide.

        It reaches Rust as the generator's exception; the sink close is
        idempotent, so the stream still terminates.
        """
        sender = BrokenSender()
        with pytest.raises(RuntimeError, match="close exploded"):
            run(drive_push_egress(gen("a"), sender))


# ===========================================================================
# 4. Chunks: order and completeness
# ===========================================================================


class TestChunkDelivery:
    def test_chunks_sent_in_order(self):
        chunks = [{"i": i} for i in range(5)]
        sender = FakeSender()
        run(drive_push_egress(gen(*chunks), sender))
        assert sender.sent == chunks

    def test_no_chunks_dropped(self):
        chunks = list(range(100))
        sender = FakeSender()
        run(drive_push_egress(gen(*chunks), sender))
        assert sender.sent == chunks

    def test_chunks_before_exception_are_sent(self):
        """Items before a handler exception should all be delivered."""
        sender = FakeSender()
        with pytest.raises(ValueError):
            run(
                drive_push_egress(
                    raising_gen([0, 1, 2], ValueError("mid-stream")), sender
                )
            )
        assert sender.sent == [
            0,
            1,
            2,
        ], f"expected [0, 1, 2] before the error, got {sender.sent}"


# ===========================================================================
# 5. Fallback path (no sender)
# ===========================================================================


class TestFallbackPath:
    def test_no_sender_returns_raw_stream(self):
        """Without a sender the handler's own async generator is returned unchanged."""
        decorated = push_egress_capable(_gen_three_items)
        dummy = object.__new__(object)

        async def _test():
            g = decorated(dummy, request={}, context=None)
            assert inspect.isasyncgen(g), "fallback should return an async generator"
            return [item async for item in g]

        collected = run(_test())
        assert collected == [1, 2, 3]

    def test_no_sender_logs_exactly_once(self, caplog):
        """The notice fires on the first call and never again.

        It is INFO, not WARNING: the pull arm is reached in normal operation by
        in-process callers and the canary health check, which go through the
        pull engine registered in the local endpoint registry. Logging it at
        warning level made every canary-enabled deployment report a false alarm.
        """
        decorated = push_egress_capable(_gen_one_token)
        dummy = object.__new__(object)

        with caplog.at_level(logging.INFO):
            decorated(dummy, request={}, context=None)
            decorated(dummy, request={}, context=None)

        notices = [r for r in caplog.records if "response_sender" in r.message]
        assert (
            len(notices) == 1
        ), f"expected exactly 1 notice about missing sender, got {len(notices)}"
        assert notices[0].levelno == logging.INFO, (
            "the pull arm is a normal path, not a fault; logging it at "
            f"{notices[0].levelname} cries wolf on every health check"
        )

    def test_logged_no_sender_global_is_module_mutable_state(self):
        """_logged_no_sender is module-global mutable state.

        This is a correctness risk: if tests don't reset it (or the reset
        fixture fails), warnings bleed across tests.  The autouse fixture covers
        this, but here we verify the guard itself works as a boolean latch.
        """
        assert pe._logged_no_sender is False  # fixture reset it
        decorated = push_egress_capable(_gen_empty)
        dummy = object.__new__(object)
        decorated(dummy, request={}, context=None)
        assert pe._logged_no_sender is True  # latched after first call
        decorated(dummy, request={}, context=None)
        assert pe._logged_no_sender is True  # stays latched


# ===========================================================================
# 7. The Rust path-selection predicate, pinned from Python
# ===========================================================================


def _accepts_response_sender(func) -> bool:
    """Reimplementation of the Rust `handler_supports_push` predicate.

    `push_egress.rs::handler_supports_push` calls
    `context.rs::callable_accepts_kwarg`, which is exactly:

        inspect.signature(callable).parameters.__contains__("response_sender")

    with a `.unwrap_or(false)` around it for callables `inspect` cannot handle.
    Mirrored here so the handler-shape matrix below can be pinned without the
    compiled bindings.

    This tests the PREDICATE, not the Rust code that calls it. It cannot catch
    a change to handler_supports_push itself -- it catches the far likelier
    regression, where a handler's shape starts or stops matching.
    """
    try:
        return "response_sender" in inspect.signature(func).parameters
    except (TypeError, ValueError):
        return False


class TestRustPathSelection:
    """Which handlers the Rust bridge will drive in push mode.

    Both error directions are silent and severe:

    * false negative -> every trtllm endpoint reverts to the pull path and the
      change is a no-op, with no error logged anywhere;
    * false positive -> a non-trtllm handler is driven in push mode, never
      pushes, and its stream is only terminated by the Rust safety net.
    """

    def test_decorated_bound_method_is_selected(self):
        class Handler:
            @push_egress_capable
            async def generate(self, request, context):
                yield {"token": "a"}

        assert _accepts_response_sender(Handler().generate), (
            "the decorated handler is NOT push-capable; every trtllm endpoint "
            "would silently fall back to the pull path"
        )

    def test_undecorated_handler_shapes_are_not_selected(self):
        """Every other handler shape in the repo must stay on the pull path."""

        async def vllm_style(self, request):
            yield

        async def context_style(self, request, context):
            yield

        async def kwargs_only(self, request, **kwargs):
            yield

        async def args_and_kwargs(*args, **kwargs):
            yield

        class CallableHandler:
            async def __call__(self, request, **kwargs):
                yield

        for handler in (
            vllm_style,
            context_style,
            kwargs_only,
            args_and_kwargs,
            CallableHandler(),
            functools.partial(kwargs_only, None),
        ):
            name = getattr(handler, "__name__", type(handler).__name__)
            assert not _accepts_response_sender(handler), (
                f"{name} was classified push-capable; Rust would drive it in "
                "push mode, where it never pushes and never terminates"
            )

    def test_var_keyword_does_not_match(self):
        """`**kwargs` must NOT count as declaring `response_sender`.

        `inspect.signature` reports a VAR_KEYWORD parameter under its own name
        ("kwargs"), so the containment check is False. If this ever flipped,
        every `**kwargs` handler in the repo would be driven in push mode.
        """

        async def bare_kwargs(**kwargs):
            yield

        params = inspect.signature(bare_kwargs).parameters
        assert "kwargs" in params
        assert "response_sender" not in params

    def test_uninspectable_callable_is_not_selected(self):
        """A callable `inspect` cannot handle must default to the pull path.

        Mirrors the Rust `.unwrap_or(false)`: unknown shape means don't push.
        """
        assert not _accepts_response_sender(len)

    def test_keeping_wrapped_link_silently_disables_push(self):
        """Regression canary for the `del dispatch.__wrapped__` line.

        If that deletion is ever removed, `inspect.signature` follows
        `__wrapped__` back to the undecorated `generate(self, request, context)`,
        `response_sender` disappears, and the entire push path is disabled with
        no error anywhere. This proves the failure mode is real, so the
        `test_no_wrapped_link` guard above is understood to be load-bearing.
        """

        def push_egress_capable_without_del(func):
            @functools.wraps(func)
            def dispatch(self, request, context=None, response_sender=None, **kw):
                return func(self, request, context, **kw)

            return dispatch  # __wrapped__ deliberately NOT deleted

        class Handler:
            @push_egress_capable_without_del
            async def generate(self, request, context):
                yield

        assert not _accepts_response_sender(Handler().generate), (
            "expected the __wrapped__ link to hide response_sender; if this "
            "fails the canary is no longer meaningful"
        )


# ===========================================================================
# 8. The decorator is actually applied to the real handlers
# ===========================================================================


class TestDecoratorAppliedToRealHandlers:
    """Every LLM handler `generate` must carry @push_egress_capable.

    Verified against the source with `ast` rather than by importing the
    classes: importing them pulls in tensorrt_llm, which needs CUDA, so an
    import-based test would silently skip everywhere except a GPU container
    and would therefore never guard anything in normal CI.

    Confirmed equivalent in the worker-egress-push container, where all four
    real classes report
    `params=['self', 'request', 'context', 'response_sender', 'kwargs']`.

    Dropping the decorator from a handler is invisible at runtime: that
    endpoint just reverts to the pull path, with no error and no failure.
    """

    HANDLERS: ClassVar[dict[str, list[str]]] = {
        "handlers.py": ["EncodeHandler", "PrefillHandler", "DecodeHandler"],
        "aggregated_handler.py": ["AggregatedHandler"],
    }

    @staticmethod
    def _generate_decorators(path, class_name):
        tree = ast.parse(pathlib.Path(path).read_text())
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                for item in node.body:
                    if (
                        isinstance(item, (ast.AsyncFunctionDef, ast.FunctionDef))
                        and item.name == "generate"
                    ):
                        return [
                            d.id if isinstance(d, ast.Name) else ast.dump(d)
                            for d in item.decorator_list
                        ]
                raise AssertionError(f"{class_name} has no generate method")
        raise AssertionError(f"{class_name} not found in {path}")

    @pytest.mark.parametrize(
        "module,class_name",
        [(m, c) for m, cs in HANDLERS.items() for c in cs],
    )
    def test_generate_is_push_egress_capable(self, module, class_name):
        path = _COMPONENTS_SRC / "dynamo" / "trtllm" / "request_handlers" / module
        decorators = self._generate_decorators(path, class_name)
        assert "push_egress_capable" in decorators, (
            f"{class_name}.generate is missing @push_egress_capable "
            f"(decorators: {decorators}). That endpoint silently reverts to "
            "the pull path with no error anywhere."
        )

    @pytest.mark.parametrize(
        "module,class_name",
        [(m, c) for m, cs in HANDLERS.items() for c in cs],
    )
    def test_push_egress_capable_is_outermost(self, module, class_name):
        """It must be first in the list: `inspect.signature` sees the outermost
        wrapper, and range_decorator must wrap a real async-gen function."""
        path = _COMPONENTS_SRC / "dynamo" / "trtllm" / "request_handlers" / module
        decorators = self._generate_decorators(path, class_name)
        assert decorators[0] == "push_egress_capable", (
            f"{class_name}.generate: @push_egress_capable must be OUTERMOST, "
            f"got {decorators}"
        )


# ===========================================================================
# 9. Push egress must not bleed into the other engines
# ===========================================================================


class TestNoBleedIntoOtherEngines:
    """Only TRT-LLM handlers may be push-capable.

    `Endpoint.serve_endpoint` is shared by every Python worker -- vLLM, SGLang,
    frontend, planner, router, mocker, and more -- and the ONLY thing keeping
    them on the pull path is that `handler_supports_push` finds no
    `response_sender` parameter on their handlers.

    So the moment any other engine's handler declares a parameter with that
    name, for any unrelated reason, Rust drives it with the push engine. It
    never calls `sender.send`, and the driver task fails any request whose
    handler yields instead of pushing -- so that handler's every request dies
    at the first response. This test makes the collision loud at the source.
    """

    @staticmethod
    def _declares_response_sender(path):
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:
            return []  # not importable Python on this interpreter; skip
        hits = []
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            a = node.args
            names = [arg.arg for arg in (*a.posonlyargs, *a.args, *a.kwonlyargs) if arg]
            if "response_sender" in names:
                hits.append(node.name)
        return hits

    def test_only_trtllm_handlers_declare_response_sender(self):
        offenders = {}
        for path in (_COMPONENTS_SRC / "dynamo").rglob("*.py"):
            if "/trtllm/" in path.as_posix():
                continue  # the one engine that is supposed to have it
            hits = self._declares_response_sender(path)
            if hits:
                offenders[path.relative_to(_COMPONENTS_SRC).as_posix()] = hits

        assert not offenders, (
            "non-TRT-LLM code declares a `response_sender` parameter: "
            f"{offenders}. Endpoint.serve_endpoint selects the push egress "
            "engine purely on that parameter name, so these handlers would be "
            "driven in push mode without ever pushing -- silently degrading "
            "to the forward-yield path. Rename the parameter."
        )

    def test_trtllm_side_is_where_it_lives(self):
        """Sanity check on the scan: it must actually find the trtllm ones.

        Guards against the walk silently matching nothing (wrong root, changed
        layout), which would make the test above vacuously green forever.
        """
        found = {}
        for path in (_COMPONENTS_SRC / "dynamo" / "trtllm").rglob("*.py"):
            hits = self._declares_response_sender(path)
            if hits:
                found[path.name] = hits

        assert "push_egress.py" in found, (
            f"scan found no response_sender parameter in trtllm/push_egress.py "
            f"(found: {found}); the AST walk is probably broken, which would "
            "make the no-bleed assertion vacuous"
        )
