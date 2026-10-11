import unittest
from dataclasses import replace

from src.config import Config


class ReleaseConfigurationTests(unittest.TestCase):
    def test_event_api_host_is_prefilled_and_valid_in_live_mode(self):
        cfg = replace(Config(), api_url="https://mock-api.roostoo.com", api_key="test-key",
                      api_secret="test-secret", paper_mode=False, db_path="/tmp/boss-v9-test.db",
                      log_file="/tmp/boss-v9-test.log")
        cfg.validate(require_credentials=True, require_api_url=True)
        self.assertEqual(cfg.expected_starting_cash, 100_000.0)
        self.assertEqual(cfg.poll_seconds, 60)
        self.assertEqual(cfg.rebalance_seconds, 120)

    def test_offline_research_allows_blank_api_host(self):
        cfg = replace(Config(), api_url="", api_key="", api_secret="", paper_mode=True,
                      db_path="/tmp/boss-v9-research.db", log_file="/tmp/boss-v9-research.log")
        cfg.validate(require_credentials=False, require_api_url=False)

    def test_request_spacing_is_independent_of_turnover_limit(self):
        cfg = replace(Config(), api_min_request_interval_seconds=0.2, max_daily_turnover=0.0,
                      paper_mode=True, db_path="/tmp/boss-v9-spacing.db", log_file="/tmp/boss-v9-spacing.log")
        cfg.validate(require_credentials=False, require_api_url=False)
        self.assertEqual(cfg.max_daily_turnover, 0.0)


if __name__ == "__main__":
    unittest.main()
