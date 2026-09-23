import unittest

from model.differential_privacy import PrivacyBudget, add_privacy_noise, clip_update


class DifferentialPrivacyTests(unittest.TestCase):
    def test_clip_update_bounds_large_updates(self) -> None:
        weight, bias = clip_update(weight=30.0, bias=40.0, clip_norm=5.0)

        self.assertAlmostEqual((weight**2 + bias**2) ** 0.5, 5.0, places=6)

    def test_clip_update_leaves_small_updates_untouched(self) -> None:
        weight, bias = clip_update(weight=0.1, bias=0.2, clip_norm=5.0)

        self.assertEqual((weight, bias), (0.1, 0.2))

    def test_smaller_epsilon_adds_more_noise_on_average(self) -> None:
        tight_budget = PrivacyBudget(epsilon=0.1, clip_norm=5.0)
        loose_budget = PrivacyBudget(epsilon=10.0, clip_norm=5.0)

        self.assertGreater(tight_budget.noise_multiplier(), loose_budget.noise_multiplier())

    def test_noise_is_reproducible_with_a_seed(self) -> None:
        budget = PrivacyBudget(epsilon=1.0)

        first = add_privacy_noise(1.0, 1.0, budget, seed=99)
        second = add_privacy_noise(1.0, 1.0, budget, seed=99)

        self.assertEqual(first, second)

    def test_rejects_non_positive_epsilon(self) -> None:
        with self.assertRaises(ValueError):
            PrivacyBudget(epsilon=0.0).noise_multiplier()


if __name__ == "__main__":
    unittest.main()
