import concurrent.futures
import os
import pathlib
import tempfile
import unittest

from _support import load_harness


class ResultsLedgerTests(unittest.TestCase):
    def setUp(self):
        self.harness = load_harness()

    def test_append_preserves_prefix_and_links_canonical_records(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "results.jsonl"
            ledger = self.harness.ResultsLedger(path)
            first = ledger.append({"run_id": "v2-01-01", "status": "ok"})
            prefix = path.read_bytes()
            second = ledger.append({"run_id": "v2-01-02", "status": "ok"})
            self.assertEqual(path.read_bytes()[: len(prefix)], prefix)
            self.assertEqual(first["sequence"], 1)
            self.assertEqual(second["sequence"], 2)
            self.assertIsNone(first["previous_record_digest"])
            self.assertEqual(second["previous_record_digest"], first["record_digest"])
            self.assertEqual(ledger.read(), [first, second])

    def test_duplicate_tampered_and_truncated_ledgers_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "results.jsonl"
            ledger = self.harness.ResultsLedger(path)
            ledger.append({"run_id": "v2-01-01", "status": "ok"})
            with self.assertRaises(ValueError):
                ledger.append({"run_id": "v2-01-01", "status": "ok"})
            body = path.read_text().replace('"status":"ok"', '"status":"forged"')
            path.write_text(body)
            with self.assertRaises(ValueError):
                self.harness.ResultsLedger(path).read()
            path.write_bytes(path.read_bytes() + b'{"run_id":')
            with self.assertRaises(ValueError):
                self.harness.ResultsLedger(path).read()

    def test_concurrent_appends_are_serialized_without_lost_records(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger = self.harness.ResultsLedger(pathlib.Path(directory) / "results.jsonl")

            def append(index):
                return ledger.append({"run_id": f"v2-02-{index:02d}", "status": "ok"})

            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                list(executor.map(append, range(1, 17)))
            rows = ledger.read()
            self.assertEqual(len(rows), 16)
            self.assertEqual({row["run_id"] for row in rows}, {f"v2-02-{i:02d}" for i in range(1, 17)})
            self.assertEqual([row["sequence"] for row in rows], list(range(1, 17)))

    def test_ledger_rejects_hardlinks_and_insecure_modes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "results.jsonl"
            path.touch(mode=0o600)
            alias = pathlib.Path(directory) / "alias.jsonl"
            os.link(path, alias)
            with self.assertRaises(ValueError):
                self.harness.ResultsLedger(path).read()
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "results.jsonl"
            path.touch(mode=0o644)
            with self.assertRaises(ValueError):
                self.harness.ResultsLedger(path).read()


if __name__ == "__main__":
    unittest.main()
