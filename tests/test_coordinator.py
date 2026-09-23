import unittest

from model.hospital_node import HospitalNode
from model.synthetic_data import make_synthetic_mri_cases, partition_across_hospitals, split_cases
from server.coordinator import run_federated_training


class CoordinatorTests(unittest.TestCase):
    def _build_nodes(self, seed: int = 3) -> tuple[list[HospitalNode], tuple]:
        scans, masks = make_synthetic_mri_cases(case_count=32, seed=seed)
        train_scans, train_masks, test_scans, test_masks = split_cases(scans, masks)
        partition = partition_across_hospitals(train_scans, train_masks, ("Hospital A", "Hospital B", "Hospital C"))
        nodes = [HospitalNode(name=name, scans=s, masks=m) for name, (s, m) in partition.items()]
        return nodes, (test_scans, test_masks)

    def test_federated_training_converges_toward_the_centralized_baseline(self) -> None:
        nodes, (test_scans, test_masks) = self._build_nodes()

        result = run_federated_training(nodes, test_scans, test_masks, rounds=12, local_epochs=15)

        self.assertLess(result.history[-1].mean_local_loss, result.history[0].mean_local_loss)
        self.assertGreater(result.history[-1].held_out_dice, 0.9)

    def test_survives_a_hospital_node_dropping_offline_mid_training(self) -> None:
        nodes, (test_scans, test_masks) = self._build_nodes()

        result = run_federated_training(
            nodes,
            test_scans,
            test_masks,
            rounds=8,
            local_epochs=10,
            offline_rounds={3: {"Hospital B"}, 4: {"Hospital B"}},
        )

        self.assertEqual(len(result.history), 8)
        self.assertGreater(result.history[-1].held_out_dice, 0.85)

    def test_raises_if_every_node_is_offline_in_a_round(self) -> None:
        nodes, (test_scans, test_masks) = self._build_nodes()
        all_names = {node.name for node in nodes}

        with self.assertRaises(RuntimeError):
            run_federated_training(
                nodes, test_scans, test_masks, rounds=2, offline_rounds={1: all_names}
            )


if __name__ == "__main__":
    unittest.main()
