import pytest

from cectcls.sampling import mixed_sample, topk_attention


def test_reproducible_disjoint_paired_sample():
    a = mixed_sample(range(70), "P001", epoch=3, uniform=16, random_count=16)
    b = mixed_sample(range(70), "P001", epoch=3, uniform=16, random_count=16)
    assert a == b
    assert len(a["selected_reference_ids"]) == 32
    assert not set(a["uniform_reference_ids"]) & set(a["random_reference_ids"])
    assert a["selected_ap_ids"] == a["selected_pvp_ids"]


def test_short_sequence_and_topk_tie_break():
    sample = mixed_sample([5, 7, 11], "P002")
    assert sample["selected_reference_ids"] == [5, 7, 11]
    assert topk_attention([2, 1, 3], [0.1, 0.5, 0.5], 2) == [(1, 0.5), (3, 0.5)]


def test_bad_sampling():
    with pytest.raises(ValueError):
        mixed_sample([], "P003")
