import unittest

from rital_ir_project.hierarchy import compute_cluster_sparsity


class HierarchyTests(unittest.TestCase):
    def test_cluster_sparsity_matches_formula(self) -> None:
        sparsity = compute_cluster_sparsity([3, 2], entity_count=5)
        expected = 1.0 - ((3 * 2 + 2 * 1) / (5 * 4))
        self.assertAlmostEqual(sparsity, expected)


if __name__ == "__main__":
    unittest.main()
