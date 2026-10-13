import numpy as np
import pytest
from CodeEntropy.levels.search import _nearest_stable, Search


@pytest.mark.parametrize('size', [0, 1, 29, 30, 31, 100, 13332])
def test_partial_sort_matches_stable_reference_with_ties(size):
    rng = np.random.default_rng(48)
    for distances in (rng.integers(0, 12, size).astype(float),
                      np.zeros(size), rng.random(size)):
        np.testing.assert_array_equal(_nearest_stable(distances),
                                      np.argsort(distances, kind='mergesort')[:30])
    if size:
        distances[0] = np.nan
        np.testing.assert_array_equal(_nearest_stable(distances),
                                      np.argsort(distances, kind='mergesort')[:30])


def test_fused_distances_bitwise_equal_to_original_including_unwrapped_coordinates():
    rng = np.random.default_rng(91)
    coms = rng.normal(size=(13332, 3)) * 180
    dimensions = np.array([75.02458, 73.5, 80.25], dtype=np.float32)
    coms[:3] = [np.zeros(3), 0.5*dimensions, -0.5*dimensions]
    for center in (np.zeros(3), coms[3], coms[-1]):
        delta = coms-center
        for d in range(3):
            delta[:,d] = np.where(delta[:,d] > 0.5*dimensions[d],
                                  delta[:,d]-dimensions[d], delta[:,d])
            delta[:,d] = np.where(delta[:,d] < -0.5*dimensions[d],
                                  delta[:,d]+dimensions[d], delta[:,d])
        reference = np.sqrt((delta*delta).sum(axis=1))
        np.testing.assert_array_equal(Search()._get_distances(coms, center, dimensions), reference)
