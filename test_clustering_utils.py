import unittest
from itertools import permutations
import numpy as np
from clustering_utils import hopkins, align_labels, expected_rand_index
from sklearn.metrics import jaccard_score, rand_score


class StatisticalChecks(unittest.TestCase):
    def test_expected_rand_matches_exhaustive_permutations(self):
        truth = [0, 0, 0, 1, 1]
        labels = [0, 0, 0, 0, 1]
        exact = np.mean([rand_score(truth, candidate)
                         for candidate in set(permutations(labels))])
        self.assertAlmostEqual(expected_rand_index(truth, labels), exact)
        self.assertAlmostEqual(exact, .48)

    def test_hopkins_null_and_clustered_controls(self):
        rng = np.random.default_rng(42)
        uniform = rng.uniform(size=(2000, 2))
        clusters = np.vstack([rng.normal(-3, .2, (1000, 2)),
                              rng.normal(3, .2, (1000, 2))])
        null = np.mean([hopkins(uniform, random_state=s) for s in range(10)])
        self.assertTrue(.4 < null < .6, null)
        self.assertGreater(hopkins(clusters), .9)

    def test_permuted_labels_do_not_change_jaccard(self):
        truth = np.array([0, 0, 1, 1, 2, 2, 3, 3])
        predicted = np.array([8, 8, 6, 6, 9, 9, 7, 7])
        aligned, mapping = align_labels(truth, predicted)
        self.assertEqual(jaccard_score(truth, aligned, average='macro'), 1.)
        self.assertEqual(len(mapping), 4)

    def test_extra_clusters_are_not_merged(self):
        aligned, mapping = align_labels([0, 0, 1, 1], [2, 3, 4, 4])
        self.assertEqual(len(mapping), 2)
        self.assertIn(-1, aligned)


if __name__ == '__main__':
    unittest.main()
