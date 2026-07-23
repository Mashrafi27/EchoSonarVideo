from echo_rl.data.balance import class_weights, resample_indices


def test_class_weights_inverse_freq():
    w = class_weights(["no"] * 77 + ["yes"] * 23)
    assert w["yes"] > w["no"]


def test_resample_balances():
    labels = ["no"] * 90 + ["yes"] * 10
    idx = resample_indices(labels, 1000, seed=0)
    picked = [labels[i] for i in idx]
    frac_yes = picked.count("yes") / len(picked)
    assert 0.4 < frac_yes < 0.6          # balanced, not the 10% prior


def test_resample_deterministic():
    labels = ["a", "b", "c"] * 10
    assert resample_indices(labels, 50, seed=1) == resample_indices(labels, 50, seed=1)
