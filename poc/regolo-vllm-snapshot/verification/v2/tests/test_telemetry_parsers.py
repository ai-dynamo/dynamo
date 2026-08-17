import os
import pathlib
import tempfile
import unittest

from _support import load_harness


class TelemetryParserTests(unittest.TestCase):
    def setUp(self):
        self.harness = load_harness()

    def test_meminfo_and_psi_preserve_units_and_reject_malformed_input(self):
        meminfo = """MemTotal:       1000 kB
MemAvailable:    400 kB
Cached:          300 kB
SReclaimable:     50 kB
Shmem:            20 kB
"""
        self.assertEqual(
            self.harness.parse_meminfo(meminfo),
            {"mem_available_bytes": 409600, "page_cache_bytes": 337920},
        )
        psi = self.harness.parse_psi(
            "some avg10=0.10 avg60=0.20 avg300=0.30 total=40\n"
            "full avg10=0.00 avg60=0.01 avg300=0.02 total=3\n"
        )
        self.assertEqual(psi["some"]["avg10"], 0.10)
        self.assertEqual(psi["full"]["total"], 3)
        for parser, bad in ((self.harness.parse_meminfo, "Cached: nine kB\n"), (self.harness.parse_psi, "some avg10=nope\n")):
            with self.subTest(parser=parser.__name__), self.assertRaises(ValueError):
                parser(bad)

    def test_meminfo_ignores_real_unitless_lines_but_rejects_malformed_required_fields(self):
        meminfo = (
            "MemTotal: 1000 kB\nMemAvailable: 400 kB\nCached: 300 kB\n"
            "SReclaimable: 50 kB\nShmem: 20 kB\nHugetlbPages: 0\n"
        )
        self.assertEqual(self.harness.parse_meminfo(meminfo)["mem_available_bytes"], 409600)
        for malformed in (
            "MemAvailable: four kB\nCached: 300 kB\nSReclaimable: 50 kB\nShmem: 20 kB\n",
            "MemAvailable: 400 kB\nCached: 300 MB\nSReclaimable: 50 kB\nShmem: 20 kB\n",
        ):
            with self.subTest(malformed=malformed), self.assertRaises(ValueError):
                self.harness.parse_meminfo(malformed)

    def test_meminfo_accepts_real_nonrequired_digit_named_fields_but_remains_fail_closed(self):
        valid = (
            "MemTotal: 1000 kB\nMemAvailable: 400 kB\nCached: 300 kB\n"
            "SReclaimable: 50 kB\nShmem: 20 kB\nDirectMap4k: 8 kB\n"
            "DirectMap2M: 16 kB\nDirectMap1G: 32 kB\n"
        )
        self.assertEqual(
            self.harness.parse_meminfo(valid),
            {"mem_available_bytes": 409600, "page_cache_bytes": 337920},
        )
        for malformed in (
            valid + "not a meminfo line\n",
            valid + "MemAvailable: 400 kB\n",
            valid.replace("MemAvailable: 400 kB", "MemAvailable: 400"),
        ):
            with self.subTest(malformed=malformed), self.assertRaises(ValueError):
                self.harness.parse_meminfo(malformed)

    def test_io_stat_and_diskstats_keep_device_identity(self):
        io_stat = self.harness.parse_io_stat(
            "253:0 rbytes=10 wbytes=20 rios=1 wios=2\n8:0 rbytes=30 wbytes=40\n"
        )
        self.assertEqual(io_stat["253:0"]["rbytes"], 10)
        self.assertEqual(io_stat["8:0"]["wbytes"], 40)
        diskstats = self.harness.parse_diskstats(
            "8 0 sda 1 2 3 4 5 6 7 8 9 10 11\n"
            "7 6 loop6 11 12 13 14 15 16 17 18 19 20 21\n"
            "253 0 dm-0 21 22 23 24 25 26 27 28 29 30 31\n"
        )
        self.assertEqual(set(diskstats), {"sda", "loop6", "dm-0"})
        self.assertEqual(diskstats["dm-0"]["major"], 253)
        self.assertEqual(diskstats["sda"]["sectors_read"], 3)
        with self.assertRaises(ValueError):
            self.harness.parse_io_stat("8:0 rbytes=wrong\n")
        with self.assertRaises(ValueError):
            self.harness.parse_diskstats("not a diskstats row\n")

    def test_io_stat_uses_only_terminal_device_for_real_stacked_rows_and_rejects_ambiguity(self):
        stacked = "7:7 7:6 rbytes=10 wbytes=20 rios=1 wios=2 dbytes=0 dios=0\n"
        self.assertEqual(
            self.harness.parse_io_stat(stacked),
            {"7:6": {"rbytes": 10, "wbytes": 20, "rios": 1, "wios": 2, "dbytes": 0, "dios": 0}},
        )
        for invalid in (
            stacked + "7:6 rbytes=1 wbytes=2\n",
            "7:7 rbytes=10 wbytes=20 7:6\n",
            "7:7 7:6 rios=1 wios=2\n",
            "7:7 7:6 rbytes=ten wbytes=20\n",
            "7:7 7:6 rbytes=10 wbytes=20 bad=value!\n",
        ):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                self.harness.parse_io_stat(invalid)

    def test_cpu_and_gpu_memory_parsers_normalize_utilization_and_mib(self):
        before = "cpu 100 0 100 800 0 0 0 0 0 0\n"
        after = "cpu 150 0 150 900 0 0 0 0 0 0\n"
        self.assertEqual(self.harness.cpu_utilization(before, after), 0.5)
        self.assertEqual(self.harness.parse_gpu_memory_mib("100\n200.5\n"), 300.5)
        with self.assertRaises(ValueError):
            self.harness.cpu_utilization(after, before)
        with self.assertRaises(ValueError):
            self.harness.parse_gpu_memory_mib("not-a-number\n")

    def test_directory_sizes_rejects_symlinks_instead_of_traversing_them(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            checkpoint = root / "checkpoint"
            pages = root / "pages"
            checkpoint.mkdir()
            pages.mkdir()
            (checkpoint / "image").write_bytes(b"1234")
            (pages / "page").write_bytes(b"12")
            self.assertEqual(
                self.harness.directory_sizes({"checkpoint": checkpoint, "pages": pages}),
                {"checkpoint": 4, "pages": 2},
            )
            outside = root / "outside"
            outside.write_bytes(b"do-not-count")
            os.symlink(outside, checkpoint / "escape")
            with self.assertRaises(ValueError):
                self.harness.directory_sizes({"checkpoint": checkpoint})


if __name__ == "__main__":
    unittest.main()
