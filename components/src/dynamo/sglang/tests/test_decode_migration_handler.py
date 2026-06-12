# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase
from unittest.mock import Mock

from dynamo.sglang.request_handlers.llm.decode_handler import DecodeWorkerHandler


async def collect(generator):
    return [item async for item in generator]


class DecodeMigrationHandlerTests(IsolatedAsyncioTestCase):
    def handler(self):
        handler = object.__new__(DecodeWorkerHandler)
        handler._decode_migration_reservations = {}
        handler.engine = SimpleNamespace(
            tokenizer_manager=SimpleNamespace(abort_request=Mock())
        )
        return handler

    async def test_repeated_arm_returns_existing_ready_session(self):
        handler = self.handler()
        handler.engine.async_generate = Mock(
            side_effect=AssertionError("ready session must not be armed twice")
        )
        handler._decode_migration_reservations["migration"] = {
            "migration_id": "migration",
            "rid": "request",
            "bootstrap_room": 17,
            "source": {
                "bootstrap_host": "127.0.0.1",
                "bootstrap_port": 8998,
            },
            "reserve_tokens": 128,
            "status": "ready",
        }

        responses = await collect(
            handler.prepare_decode_migration(
                {
                    "migration_id": "migration",
                    "rid": "request",
                    "source_state": {"committed_len": 32},
                    "destination_request": {},
                }
            )
        )

        self.assertEqual(responses[0]["status"], "ready")
        self.assertEqual(responses[0]["bootstrap_room"], 17)
        handler.engine.async_generate.assert_not_called()

    async def test_destination_abort_releases_session_and_engine_request(self):
        handler = self.handler()
        handler._decode_migration_reservations["migration"] = {
            "migration_id": "migration",
            "rid": "request",
            "status": "active",
        }

        responses = await collect(
            handler.finalize_decode_migration(
                {
                    "side": "destination",
                    "action": "abort",
                    "migration_id": "migration",
                }
            )
        )

        self.assertTrue(responses[0]["success"])
        self.assertEqual(responses[0]["status"], "aborted")
        self.assertNotIn("migration", handler._decode_migration_reservations)
        handler.engine.tokenizer_manager.abort_request.assert_called_once_with(
            rid="request", abort_all=False
        )

    async def test_repeated_destination_abort_is_idempotent(self):
        handler = self.handler()

        responses = await collect(
            handler.finalize_decode_migration(
                {
                    "side": "destination",
                    "action": "abort",
                    "migration_id": "missing",
                }
            )
        )

        self.assertTrue(responses[0]["success"])
        self.assertEqual(responses[0]["status"], "unknown")
        handler.engine.tokenizer_manager.abort_request.assert_not_called()


if __name__ == "__main__":
    import unittest

    unittest.main()
