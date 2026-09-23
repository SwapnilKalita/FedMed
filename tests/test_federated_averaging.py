import unittest

from model.hospital_node import ClientUpdate
from server.federated_averaging import federated_average


class FederatedAveragingTests(unittest.TestCase):
    def test_larger_hospital_has_more_influence(self) -> None:
        small_hospital = ClientUpdate(weight=0.0, bias=0.0, sample_count=1, local_loss=0.1)
        large_hospital = ClientUpdate(weight=10.0, bias=10.0, sample_count=99, local_loss=0.1)

        weight, bias = federated_average([small_hospital, large_hospital])

        self.assertGreater(weight, 9.0)
        self.assertGreater(bias, 9.0)

    def test_equal_sample_counts_average_evenly(self) -> None:
        a = ClientUpdate(weight=2.0, bias=4.0, sample_count=10, local_loss=0.1)
        b = ClientUpdate(weight=4.0, bias=8.0, sample_count=10, local_loss=0.1)

        weight, bias = federated_average([a, b])

        self.assertAlmostEqual(weight, 3.0)
        self.assertAlmostEqual(bias, 6.0)

    def test_rejects_empty_update_list(self) -> None:
        with self.assertRaises(ValueError):
            federated_average([])


if __name__ == "__main__":
    unittest.main()
