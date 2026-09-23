import unittest

from model.hospital_node import ClientUpdate
from server.federated_averaging import federated_average
from server.secure_aggregation import mask_contributions, secure_federated_average


class SecureAggregationTests(unittest.TestCase):
    def _sample_updates(self) -> dict[str, ClientUpdate]:
        return {
            "Hospital A": ClientUpdate(weight=1.5, bias=-0.3, sample_count=12, local_loss=0.2),
            "Hospital B": ClientUpdate(weight=2.1, bias=0.4, sample_count=8, local_loss=0.25),
            "Hospital C": ClientUpdate(weight=0.9, bias=0.1, sample_count=5, local_loss=0.18),
        }

    def test_matches_plaintext_fedavg_result(self) -> None:
        updates = self._sample_updates()

        plain_weight, plain_bias = federated_average(list(updates.values()))
        secure_weight, secure_bias = secure_federated_average(updates, seed=42)

        self.assertAlmostEqual(plain_weight, secure_weight, places=8)
        self.assertAlmostEqual(plain_bias, secure_bias, places=8)

    def test_individual_masked_values_hide_the_true_contribution(self) -> None:
        updates = self._sample_updates()

        masked = mask_contributions(updates, seed=1)

        true_weighted = updates["Hospital A"].weight * updates["Hospital A"].sample_count
        # The mask should move the value far away from its true magnitude.
        self.assertGreater(abs(masked["Hospital A"][0] - true_weighted), 10.0)

    def test_result_is_stable_for_a_fixed_seed(self) -> None:
        updates = self._sample_updates()

        first = secure_federated_average(updates, seed=7)
        second = secure_federated_average(updates, seed=7)

        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
