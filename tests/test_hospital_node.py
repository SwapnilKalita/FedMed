import unittest

from model.hospital_node import HospitalNode
from model.synthetic_data import make_synthetic_mri_cases, partition_across_hospitals


class HospitalNodeTests(unittest.TestCase):
    def test_partition_covers_every_case_exactly_once_and_is_uneven(self) -> None:
        scans, masks = make_synthetic_mri_cases(case_count=30, seed=5)
        partition = partition_across_hospitals(scans, masks, ("A", "B", "C"))

        total_cases = sum(len(s) for s, _ in partition.values())
        self.assertEqual(total_cases, len(scans))
        # With random uneven weights, hospitals almost never get equal shares.
        counts = {name: len(s) for name, (s, _) in partition.items()}
        self.assertGreater(max(counts.values()), min(counts.values()))

    def test_local_train_warm_starts_from_global_weights(self) -> None:
        scans, masks = make_synthetic_mri_cases(case_count=8, seed=2)
        node = HospitalNode(name="Hospital A", scans=scans, masks=masks)

        update = node.local_train(global_weight=0.0, global_bias=0.0, epochs=30)

        self.assertEqual(update.sample_count, len(scans))
        self.assertNotEqual((update.weight, update.bias), (0.0, 0.0))

    def test_raw_data_never_appears_on_the_update(self) -> None:
        scans, masks = make_synthetic_mri_cases(case_count=4, seed=1)
        node = HospitalNode(name="Hospital B", scans=scans, masks=masks)

        update = node.local_train(global_weight=0.0, global_bias=0.0, epochs=5)

        update_fields = vars(update)
        self.assertNotIn("scans", update_fields)
        self.assertNotIn("masks", update_fields)


if __name__ == "__main__":
    unittest.main()
