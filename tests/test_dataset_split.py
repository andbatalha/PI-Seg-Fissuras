import unittest

from src.dataset.classes import DamageClass
from src.dataset.split import split_dataset, split_train_validation


class DatasetSplitTests(unittest.TestCase):
    def test_partitions_are_complete_disjoint_and_reproducible(self):
        index = {
            f"image_{number:02}": [DamageClass.BACKGROUND, DamageClass.STAIN]
            for number in reversed(range(10))
        }
        index["rare"] = [DamageClass.EROSION]

        development, reserved = split_dataset(index)
        train, validation = split_train_validation(development, index)
        self.assertEqual(reserved, ["image_08", "image_09"])
        self.assertEqual(len(validation), 2)
        self.assertIn("rare", train)
        self.assertEqual(len(train + validation + reserved), len(index))
        self.assertEqual(set(train + validation + reserved), set(index))
        self.assertEqual((train, validation), split_train_validation(development, index))
