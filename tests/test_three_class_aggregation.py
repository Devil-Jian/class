import numpy as np

from cectcls.train import compute_metrics, sqrt_class_weights, summarize_three_class


def test_main_five_to_three_probability_aggregation():
    mapping = {"正常": 0, "肝囊肿": 1, "血管瘤": 2, "HCC": 3, "转移瘤": 4}
    truth = np.asarray([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
    probs = np.eye(5)[truth] * 0.8 + 0.04
    metrics, three = summarize_three_class(truth, probs, mapping)
    assert three.shape == (10, 3)
    assert np.allclose(three.sum(axis=1), 1.0)
    assert np.allclose(three[1], three[2])
    assert np.allclose(three[3], three[4])
    assert metrics["macro_f1"] == 1.0
    assert metrics["macro_auc"] is not None


def test_training_weight_caps_rare_class():
    examples = ([{"label": 0}] * 30 + [{"label": 1}] * 10 +
                [{"label": 2}] * 6 + [{"label": 3}] * 2 + [{"label": 4}] * 1)
    weights = sqrt_class_weights(examples, 5, "cpu").numpy()
    assert np.isclose(weights.mean(), 1.0)
    assert weights[4] > weights[0]
    assert np.isfinite(weights).all()
