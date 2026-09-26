import unittest

from knowledge_base.fusion import reciprocal_rank_fusion


class FusionTests(unittest.TestCase):

    def test_one_search_engine(self):
        result = reciprocal_rank_fusion(([3, 0, 1], []))
        self.assertEqual(result, [3, 0, 1])

    def test_identical_rankings(self):
        result = reciprocal_rank_fusion(([2, 1, 0], [2, 1, 0]))
        self.assertEqual(result, [2, 1, 0])

    def test_combines_different_rankings(self):
        result = reciprocal_rank_fusion(([0, 1, 2], [1, 2, 0]))
        self.assertEqual(result[0], 1)


if __name__ == "__main__":
    unittest.main()
