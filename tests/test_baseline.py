import unittest

from model.baseline import IntensitySegmentationBaseline, dice_score
from model.synthetic_data import make_synthetic_mri_cases, split_cases


class BaselineTests(unittest.TestCase):
    def test_training_reduces_loss_and_segments_held_out_cases(self) -> None:
        scans, masks = make_synthetic_mri_cases(case_count=16, seed=42)
        train_scans, train_masks, test_scans, test_masks = split_cases(scans, masks)
        model = IntensitySegmentationBaseline()

        losses = model.fit(train_scans, train_masks, epochs=100)

        self.assertLess(losses[-1], losses[0])
        self.assertGreater(dice_score(model.predict(test_scans), test_masks), 0.90)

    def test_data_generation_is_reproducible(self) -> None:
        first_scans, first_masks = make_synthetic_mri_cases(case_count=2, seed=3)
        second_scans, second_masks = make_synthetic_mri_cases(case_count=2, seed=3)

        self.assertTrue((first_scans == second_scans).all())
        self.assertTrue((first_masks == second_masks).all())


if __name__ == "__main__":
    unittest.main()
