import unittest

import numpy as np

from hirag._cluster_utils import GMM_cluster
from rital_ir_project.hierarchy import compute_cluster_sparsity


class HierarchyTests(unittest.TestCase):
    def test_cluster_sparsity_matches_formula(self) -> None:
        sparsity = compute_cluster_sparsity([3, 2], entity_count=5)
        expected = 1.0 - ((3 * 2 + 2 * 1) / (5 * 4))
        self.assertAlmostEqual(sparsity, expected)

    def test_gmm_handles_collapsed_embeddings(self) -> None:
        embeddings = np.ones((5, 2), dtype=np.float32)
        labels, cluster_count = GMM_cluster(embeddings, threshold=0.1, reg_covar=1e-6)
        self.assertGreaterEqual(cluster_count, 1)
        self.assertEqual(len(labels), 5)


if __name__ == "__main__":
    unittest.main()
