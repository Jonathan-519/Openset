import unittest

from prepro.build_taxosafe_v10_dev_splits import stratified_split


class TaxoSafeV10SplitTest(unittest.TestCase):
    def test_stratified_split_is_disjoint_and_deterministic(self):
        lines = [
            "P/A/1.jpg,0,0",
            "P/A/2.jpg,0,1",
            "P/A/3.jpg,0,2",
            "P/A/4.jpg,0,3",
            "P/B/1.jpg,0,4",
            "P/B/2.jpg,0,5",
            "P/B/3.jpg,0,6",
            "P/B/4.jpg,0,7",
        ]
        a_train, a_val = stratified_split(lines, "intra", 0.5)
        b_train, b_val = stratified_split(lines, "intra", 0.5)
        self.assertEqual(a_train, b_train)
        self.assertEqual(a_val, b_val)
        self.assertFalse(set(a_train) & set(a_val))
        self.assertEqual(set(a_train) | set(a_val), set(lines))
        self.assertEqual(len(a_train), 4)
        self.assertEqual(len(a_val), 4)

    def test_extra_groups_are_preserved_on_both_sides(self):
        lines = [
            "A/1.jpg,-1,0",
            "A/2.jpg,-1,1",
            "B/1.jpg,-1,2",
            "B/2.jpg,-1,3",
        ]
        train, val = stratified_split(lines, "extra", 0.5)
        self.assertEqual({x.split("/")[0] for x in train}, {"A", "B"})
        self.assertEqual({x.split("/")[0] for x in val}, {"A", "B"})


if __name__ == "__main__":
    unittest.main()
