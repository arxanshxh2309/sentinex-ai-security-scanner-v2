import unittest

from mcmc.bayesian_model import estimate_threat


class ThreatEstimationTests(unittest.TestCase):
    def test_empty_detection_is_clear(self):
        result = estimate_threat([])

        self.assertEqual(result["threat_level"], "CLEAR")
        self.assertEqual(result["threat_probability"], 0.0)

    def test_high_confidence_handgun_is_critical(self):
        result = estimate_threat(
            [{"class": "handgun", "confidence": 0.95, "bbox_area_ratio": 0.12}]
        )

        self.assertEqual(result["threat_level"], "CRITICAL")
        self.assertGreaterEqual(result["threat_probability"], 0.85)

    def test_unknown_class_falls_back_to_generic_prior(self):
        result = estimate_threat(
            [{"class": "unknown", "confidence": 0.9, "bbox_area_ratio": 0.05}]
        )

        self.assertIn(result["threat_level"], {"LOW", "MEDIUM", "HIGH", "CRITICAL"})
        self.assertGreater(result["threat_probability"], 0.0)


if __name__ == "__main__":
    unittest.main()
