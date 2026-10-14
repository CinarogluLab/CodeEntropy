"""Dihedral states must be assigned with the periodic (circular) angle distance.

Dihedral angles live on a circle: 355 and 5 degrees are 10 degrees apart. Peak
finding already treats the histogram as circular, so nearest-peak assignment
must do the same, or a state centred near 0/360 degrees is split between
peaks and the conformational entropy is wrong.
"""

from __future__ import annotations

import numpy as np
import pytest

from CodeEntropy.entropy.configurational import ConformationalEntropy
from CodeEntropy.levels.dihedrals.kernels import (
    assign_peak_labels_and_count_flexible,
    circular_angle_distance,
    histogram_counts_by_dihedral,
    wrap_degrees_positive,
)
from CodeEntropy.levels.dihedrals.peak_detection import ConformationPeakDetector

GAS_CONSTANT = 8.3144598484848


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        (359.0, 1.0, 2.0),
        (1.0, 359.0, 2.0),
        (0.0, 360.0, 0.0),
        (355.0, 15.0, 20.0),
        (180.0, 0.0, 180.0),
        (90.0, 270.0, 180.0),
        (10.0, 100.0, 90.0),
        (-10.0, 10.0, 20.0),
        (370.0, 10.0, 0.0),
    ],
)
def test_circular_angle_distance(a, b, expected):
    assert circular_angle_distance(a, b) == pytest.approx(expected)


def test_assignment_uses_shortest_way_round_the_circle():
    # Peaks at 15 and 195 degrees. 355 is 20 degrees from 15 going through 0,
    # but 340 degrees from it on the line, which used to make 195 "nearer".
    angles = np.array([[355.0], [5.0], [175.0], [215.0]], dtype=np.float64)
    padded_peaks = np.array([[15.0, 195.0]], dtype=np.float64)
    peak_counts = np.array([2], dtype=np.int64)

    labels, flexible = assign_peak_labels_and_count_flexible(
        angles, padded_peaks, peak_counts
    )

    np.testing.assert_array_equal(labels[:, 0], [0, 0, 1, 1])
    assert flexible == 1


def test_fluctuation_across_the_seam_is_not_flexible():
    # A dihedral oscillating between 350 and 10 degrees stays in one state.
    angles = np.array([[350.0], [10.0], [355.0], [5.0], [0.0]], dtype=np.float64)
    padded_peaks = np.array([[15.0, 195.0]], dtype=np.float64)
    peak_counts = np.array([2], dtype=np.int64)

    labels, flexible = assign_peak_labels_and_count_flexible(
        angles, padded_peaks, peak_counts
    )

    np.testing.assert_array_equal(labels[:, 0], [0, 0, 0, 0, 0])
    assert flexible == 0


def test_tie_across_the_seam_goes_to_the_first_peak():
    # 0 degrees is 10 degrees from both 350 and 10: the first listed peak wins,
    # as for any other tie.
    angles = np.array([[0.0]], dtype=np.float64)
    padded_peaks = np.array([[350.0, 10.0]], dtype=np.float64)
    peak_counts = np.array([2], dtype=np.int64)

    labels, _ = assign_peak_labels_and_count_flexible(
        angles, padded_peaks, peak_counts
    )

    assert labels[0, 0] == 0


def _entropy_through_the_pipeline(raw_angles: np.ndarray, bin_width: float = 30.0):
    """Histogram, find peaks, assign states and compute S, as CodeEntropy does."""
    angles = wrap_degrees_positive(raw_angles.reshape(-1, 1))
    number_bins = int(360 / bin_width)
    counts = histogram_counts_by_dihedral(angles, number_bins)[0]
    centres = [bin_width * (i + 0.5) for i in range(number_bins)]
    peaks = ConformationPeakDetector._find_histogram_peaks(counts, centres)
    labels, _ = assign_peak_labels_and_count_flexible(
        angles,
        np.array([peaks], dtype=np.float64),
        np.array([len(peaks)], dtype=np.int64),
    )
    entropy = ConformationalEntropy.__new__(ConformationalEntropy)
    return entropy.conformational_entropy_calculation(labels[:, 0])


@pytest.mark.parametrize("centre", [0.0, 5.0, 10.0, 60.0])
def test_two_state_entropy_matches_the_analytic_value_wherever_the_state_sits(centre):
    # 700 frames spread evenly over +/-20 degrees around `centre`, 300 around 180.
    # Expected S = -R (0.7 ln 0.7 + 0.3 ln 0.3), independent of where state A is.
    state_a = centre + np.linspace(-20.0, 20.0, 700)
    state_b = 180.0 + np.linspace(-20.0, 20.0, 300)
    raw = np.concatenate([state_a, state_b])
    raw = (raw + 180.0) % 360.0 - 180.0  # MDAnalysis reports (-180, 180]

    expected = -GAS_CONSTANT * (0.7 * np.log(0.7) + 0.3 * np.log(0.3))

    assert _entropy_through_the_pipeline(raw) == pytest.approx(expected, rel=1e-12)
