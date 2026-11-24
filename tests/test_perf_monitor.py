import unittest

from services import perf_monitor


class PerfMonitorTests(unittest.TestCase):
    def setUp(self) -> None:
        perf_monitor.reset_recent_runs()

    def tearDown(self) -> None:
        perf_monitor.reset_recent_runs()

    def test_keeps_only_the_last_30_runs_and_logs_slow_reruns(self) -> None:
        with self.assertLogs("services.perf_monitor", level="WARNING") as logs:
            for _ in range(35):
                perf_monitor.record_run({"identity": 0.1}, 1.6)

        runs = perf_monitor.recent_runs()
        self.assertEqual(len(runs), 30)
        self.assertEqual(runs[-1]["total_ms"], 1600.0)
        self.assertIn("Slow application rerun", logs.output[0])

    def test_keeps_admin_menu_timings_separately(self) -> None:
        for index in range(35):
            perf_monitor.record_admin_menu_run(f"Menu {index}", 0.02)

        runs = perf_monitor.recent_admin_menu_runs()
        self.assertEqual(len(runs), 30)
        self.assertEqual(runs[-1], {"menu": "Menu 34", "duration_ms": 20.0})


if __name__ == "__main__":
    unittest.main()
