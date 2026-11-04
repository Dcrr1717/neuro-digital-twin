"""Neural surrogates of the HH neuron.

Two families are implemented and compared head-to-head (see docs/RESULTS.md):

* :class:`GRUSurrogate` -- a discrete-time sequence model. A 2-layer GRU reads
  the stimulus in *patches* of ``patch`` samples (0.5 ms by default) together
  with the parameters and emits the voltage of the whole patch per step. It
  learns the flow map over a step much longer than the stiff HH time scales.
* :class:`NeuralODESurrogate` -- a neural ODE ``dx/dt = f_phi(x, I, theta)`` in
  the four-dimensional HH state space, integrated with fixed-step RK4. It is
  trained on the full state (V, m, h, n) by multiple shooting.

Both map ``(theta, I(t)) -> V(t)`` in normalised units (see :class:`Normalizer`).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn

FloatArray = NDArray[np.float64]


@dataclass
class Normalizer:
    """Affine normalisation of parameters, current and state.

    Parameters are mapped so that the *training box* becomes ``[-1, 1]``; values
    outside that interval therefore directly signal extrapolation.
    """

    theta_lo: list[float]
    theta_hi: list[float]
    i_scale: float
    state_mean: list[float]
    state_std: list[float]

    @staticmethod
    def fit(
        theta_lo: list[float], theta_hi: list[float], current: FloatArray, states: FloatArray
    ) -> Normalizer:
        flat = states.reshape(-1, states.shape[-1])
        return Normalizer(
            theta_lo=[float(x) for x in theta_lo],
            theta_hi=[float(x) for x in theta_hi],
            i_scale=float(np.std(current)) or 1.0,
            state_mean=[float(x) for x in flat.mean(axis=0)],
            state_std=[float(x) for x in flat.std(axis=0)],
        )

    @property
    def v_mean(self) -> float:
        return self.state_mean[0]

    @property
    def v_std(self) -> float:
        return self.state_std[0]

    def theta(self, theta: FloatArray) -> Tensor:
        lo, hi = np.asarray(self.theta_lo), np.asarray(self.theta_hi)
        return torch.as_tensor(2.0 * (theta - lo) / (hi - lo) - 1.0, dtype=torch.float32)

    def current(self, current: FloatArray) -> Tensor:
        return torch.as_tensor(current / self.i_scale, dtype=torch.float32)

    def state(self, states: FloatArray) -> Tensor:
        m, s = np.asarray(self.state_mean), np.asarray(self.state_std)
        return torch.as_tensor((states - m) / s, dtype=torch.float32)

    def voltage_to_mv(self, v_n: Tensor | FloatArray) -> FloatArray:
        arr = v_n.detach().cpu().numpy() if isinstance(v_n, Tensor) else np.asarray(v_n)
        return np.asarray(arr, dtype=np.float64) * self.v_std + self.v_mean

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _mlp(
    n_in: int, width: int, n_out: int, depth: int, act: type[nn.Module] = nn.SiLU
) -> nn.Sequential:
    layers: list[nn.Module] = []
    d = n_in
    for _ in range(depth - 1):
        layers += [nn.Linear(d, width), act()]
        d = width
    layers.append(nn.Linear(d, n_out))
    return nn.Sequential(*layers)


class Surrogate(nn.Module):
    """Common interface: ``forward(theta_n (B,P), current_n (B,N)) -> v_n (B,N)``."""

    kind: str = "base"

    def config_dict(self) -> dict[str, object]:  # pragma: no cover - overridden
        raise NotImplementedError


class GRUSurrogate(Surrogate):
    """Patch-wise GRU sequence model conditioned on the parameters."""

    kind = "gru"

    def __init__(
        self, n_params: int = 4, hidden: int = 64, layers: int = 2, patch: int = 5
    ) -> None:
        super().__init__()
        self.n_params, self.hidden, self.layers, self.patch = n_params, hidden, layers, patch
        # The initial hidden state encodes the parameter-dependent resting state.
        self.h0 = _mlp(n_params, hidden, hidden * layers, depth=2, act=nn.Tanh)
        self.gru = nn.GRU(patch + n_params, hidden, num_layers=layers, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, patch))

    def config_dict(self) -> dict[str, object]:
        return {"n_params": self.n_params, "hidden": self.hidden, "layers": self.layers,
                "patch": self.patch}

    def forward(self, theta_n: Tensor, current_n: Tensor) -> Tensor:
        b, n = current_n.shape
        pad = (-n) % self.patch
        if pad:
            current_n = torch.nn.functional.pad(current_n, (0, pad))
        steps = current_n.shape[1] // self.patch
        x = current_n.reshape(b, steps, self.patch)
        x = torch.cat([x, theta_n[:, None, :].expand(b, steps, self.n_params)], dim=-1)
        h0 = torch.tanh(self.h0(theta_n)).reshape(b, self.layers, self.hidden)
        h0 = h0.transpose(0, 1).contiguous()
        out, _ = self.gru(x, h0)
        v: Tensor = self.head(out).reshape(b, steps * self.patch)
        return v[:, :n]


class NeuralODESurrogate(Surrogate):
    """Neural ODE in the (normalised) HH state space, fixed-step RK4."""

    kind = "node"
    deriv_scale: Tensor

    def __init__(
        self,
        n_params: int = 4,
        width: int = 64,
        depth: int = 3,
        dt: float = 0.1,
        substeps: int = 2,
        deriv_scale: list[float] | None = None,
    ) -> None:
        super().__init__()
        self.n_params, self.width, self.depth = n_params, width, depth
        self.dt, self.substeps = dt, substeps
        self.field = _mlp(4 + 1 + n_params, width, 4, depth)
        self.init = _mlp(n_params, width, 4, depth=2)
        scale = torch.tensor(deriv_scale or [1.0, 1.0, 1.0, 1.0], dtype=torch.float32)
        self.register_buffer("deriv_scale", scale)

    def config_dict(self) -> dict[str, object]:
        return {"n_params": self.n_params, "width": self.width, "depth": self.depth,
                "dt": self.dt, "substeps": self.substeps,
                "deriv_scale": [float(x) for x in self.deriv_scale]}

    def vector_field(self, x: Tensor, i: Tensor, theta_n: Tensor) -> Tensor:
        """``dx_n/dt`` (per ms) at normalised state ``x`` and current ``i``."""
        out: Tensor = self.field(torch.cat([x, i[:, None], theta_n], dim=-1))
        return out * self.deriv_scale

    def step(self, x: Tensor, i: Tensor, theta_n: Tensor) -> Tensor:
        """Advance one sampling interval ``dt`` with ``substeps`` RK4 steps."""
        h = self.dt / self.substeps
        for _ in range(self.substeps):
            k1 = self.vector_field(x, i, theta_n)
            k2 = self.vector_field(x + 0.5 * h * k1, i, theta_n)
            k3 = self.vector_field(x + 0.5 * h * k2, i, theta_n)
            k4 = self.vector_field(x + h * k3, i, theta_n)
            x = x + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return x

    def initial_state(self, theta_n: Tensor) -> Tensor:
        out: Tensor = self.init(theta_n)
        return out

    def rollout(self, x: Tensor, current_n: Tensor, theta_n: Tensor) -> Tensor:
        """Integrate from ``x`` over all samples of ``current_n``; returns ``(B,N,4)``."""
        states = []
        for k in range(current_n.shape[1]):
            x = self.step(x, current_n[:, k], theta_n)
            states.append(x)
        return torch.stack(states, dim=1)

    def forward(self, theta_n: Tensor, current_n: Tensor) -> Tensor:
        x0 = self.initial_state(theta_n)
        return self.rollout(x0, current_n, theta_n)[:, :, 0]


def build_model(kind: str, config: dict[str, object]) -> Surrogate:
    if kind == "gru":
        return GRUSurrogate(**config)  # type: ignore[arg-type]
    if kind == "node":
        return NeuralODESurrogate(**config)  # type: ignore[arg-type]
    raise ValueError(f"unknown model kind {kind!r}")


class Ensemble:
    """Deep ensemble (Lakshminarayanan et al. 2017) of independently trained members."""

    def __init__(self, members: list[Surrogate], normalizer: Normalizer) -> None:
        if not members:
            raise ValueError("an ensemble needs at least one member")
        self.members = members
        self.normalizer = normalizer

    @torch.no_grad()
    def predict(
        self, theta: FloatArray, current: FloatArray, batch_size: int = 512
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Predict voltage (mV).

        Returns:
            ``(mean, std, per_member)`` with shapes ``(B,N)``, ``(B,N)``,
            ``(M,B,N)``; ``std`` is the across-member standard deviation.
        """
        preds = np.stack(
            [predict_mv(m, self.normalizer, theta, current, batch_size) for m in self.members]
        )
        return preds.mean(axis=0), preds.std(axis=0), preds


@torch.no_grad()
def predict_mv(
    model: Surrogate,
    norm: Normalizer,
    theta: FloatArray,
    current: FloatArray,
    batch_size: int = 512,
) -> FloatArray:
    """Run one surrogate on raw inputs; returns voltage in mV, shape ``(B, N)``."""
    model.eval()
    dev = model_device(model)
    th, cur = norm.theta(theta), norm.current(current)
    # Copying the result back to host memory also synchronises a GPU, so
    # timings of this function include the full device round trip.
    outs = [
        norm.voltage_to_mv(model(th[i : i + batch_size].to(dev), cur[i : i + batch_size].to(dev)))
        for i in range(0, th.shape[0], batch_size)
    ]
    return np.concatenate(outs, axis=0)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def resolve_device(name: str = "auto") -> torch.device:
    """``"auto"`` -> CUDA if available, else CPU; ``"cpu"``, ``"cuda"``, ``"cuda:1"``..."""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(name)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"device {name!r} requested but PyTorch sees no CUDA device")
    if dev.type not in ("cpu", "cuda"):
        raise ValueError(f"unsupported device {name!r}")
    return dev


def device_label(device: torch.device) -> str:
    """Human-readable device name, e.g. ``cpu`` or ``cuda (NVIDIA A100)``."""
    if device.type == "cuda":
        return f"cuda ({torch.cuda.get_device_name(device)})"
    return "cpu"


def model_device(model: nn.Module) -> torch.device:
    return next(model.parameters()).device


def on_device(model: Surrogate, device: torch.device | str) -> Surrogate:
    """``model`` itself if it already lives on ``device``, else a moved copy."""
    device = torch.device(device)
    current = model_device(model)
    # A bare "cuda" means the current CUDA device, whatever its index.
    if current == device or (str(device) == "cuda" and current.type == "cuda"):
        return model
    clone = build_model(model.kind, model.config_dict())
    clone.load_state_dict(model.state_dict())
    return clone.to(device).eval()


def cpu_state_dict(model: nn.Module) -> dict[str, Tensor]:
    """State dict with every tensor on the CPU (portable checkpoints)."""
    return {k: v.detach().cpu() for k, v in model.state_dict().items()}
