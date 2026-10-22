import numpy as np
import pytest

from cectcls.matching import monotonic_path, ncc, nmi


def test_similarity_identical_better_than_noise():
    a = np.linspace(-100, 160, 2000)
    b = a[::-1]
    assert ncc(a, a) > 0.999
    assert ncc(a, b) < -0.99
    assert nmi(a, a) >= 1.9


def test_monotonic_dynamic_programming():
    rows = [
        [{"ncct_slice_id": 10, "score": 0.9},
         {"ncct_slice_id": 11, "score": 0.8}],
        [{"ncct_slice_id": 9, "score": 0.95},
         {"ncct_slice_id": 12, "score": 0.7}],
    ]
    best = monotonic_path(rows)
    assert [r["ncct_slice_id"] for r in best] == [10, 12]


def test_no_monotonic_path():
    with pytest.raises(ValueError):
        monotonic_path([[{"ncct_slice_id": 5, "score": 1.}],
                        [{"ncct_slice_id": 3, "score": 1.}]])
