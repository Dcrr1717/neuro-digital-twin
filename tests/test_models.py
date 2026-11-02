"""Tests of the surrogate networks."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from neurotwin.models import (
    Ensemble,
    GRUSurrogate,
    NeuralODESurrogate,
    Normalizer,
    build_model,
    predict_mv,
)


def _norm() -> Normalizer:
    rng = np.random.default_rng(0)
    states = rng.normal(size=(4, 50, 4))
    states[..., 0] = -60 + 10 * states[..., 0]
    return Normalizer.fit([100, 28, 0.2, 0.85], [140, 44, 0.4, 1.15], rng.normal(size=(4, 50)),
                          states)


def test_normalizer_maps_training_box_to_unit_interval() -> None:
    n = _norm()
    th = n.theta(np.array([[100, 28, 0.2, 0.85], [140, 44, 0.4, 1.15], [120, 36, 0.3, 1.0]]))
    np.testing.assert_allclose(th.numpy(), [[-1] * 4, [1] * 4, [0] * 4], atol=1e-6)
    v = np.array([-70.0, 0.0])
    v_n = (v - n.v_mean) / n.v_std
    np.testing.assert_allclose(n.voltage_to_mv(torch.tensor(v_n)), v, atol=1e-9)
    assert Normalizer(**n.to_dict()) == n  # type: ignore[arg-type]


@pytest.mark.parametrize("n_steps", [50, 53])
def test_gru_shapes_and_padding(n_steps: int) -> None:
    m = GRUSurrogate(hidden=8, patch=5)
    out = m(torch.zeros(3, 4), torch.zeros(3, n_steps))
    assert out.shape == (3, n_steps)


def test_gru_is_causal() -> None:
    """The output at time k must not depend on the stimulus after the patch of k."""
    torch.manual_seed(0)
    m = GRUSurrogate(hidden=8, patch=5).eval()
    th = torch.zeros(1, 4)
    a = torch.zeros(1, 40)
    b = a.clone()
    b[0, 20:] = 5.0
    with torch.no_grad():
        ya, yb = m(th, a), m(th, b)
    torch.testing.assert_close(ya[:, :20], yb[:, :20])
    assert not torch.allclose(ya[:, 20:], yb[:, 20:])


def test_node_rollout_and_gradients() -> None:
    torch.manual_seed(0)
    m = NeuralODESurrogate(width=8, depth=2, dt=0.1, substeps=2)
    th = torch.zeros(2, 4)
    cur = torch.randn(2, 12)
    out = m(th, cur)
    assert out.shape == (2, 12)
    states = m.rollout(torch.zeros(2, 4), cur, th)
    assert states.shape == (2, 12, 4)
    out.sum().backward()
    assert all(p.grad is not None for p in m.parameters())


def test_node_step_with_zero_field_is_identity() -> None:
    m = NeuralODESurrogate(width=8, depth=2, deriv_scale=[0.0, 0.0, 0.0, 0.0])
    x = torch.randn(3, 4)
    torch.testing.assert_close(m.step(x, torch.zeros(3), torch.zeros(3, 4)), x)


def test_build_model_roundtrip() -> None:
    for m in (GRUSurrogate(hidden=8), NeuralODESurrogate(width=8)):
        clone = build_model(m.kind, m.config_dict())
        clone.load_state_dict(m.state_dict())
        assert type(clone) is type(m)
    with pytest.raises(ValueError):
        build_model("transformer", {})


def test_ensemble_mean_and_spread() -> None:
    torch.manual_seed(0)
    n = _norm()
    members: list[GRUSurrogate] = [GRUSurrogate(hidden=8) for _ in range(3)]
    ens = Ensemble(list(members), n)
    theta = np.array([[120.0, 36.0, 0.3, 1.0]] * 2)
    cur = np.zeros((2, 25))
    mean, std, per = ens.predict(theta, cur)
    assert mean.shape == std.shape == (2, 25) and per.shape == (3, 2, 25)
    np.testing.assert_allclose(mean, per.mean(axis=0))
    assert np.all(std > 0)
    single = predict_mv(members[0], n, theta, cur)
    np.testing.assert_allclose(single, per[0], atol=1e-6)
    with pytest.raises(ValueError):
        Ensemble([], n)
