"""Training loops for the two surrogate families."""

from __future__ import annotations

import copy
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import Tensor

from .config import GRUConfig, NODEConfig
from .data import Dataset
from .models import GRUSurrogate, NeuralODESurrogate, Normalizer, Surrogate

Logger = Callable[[str], None]


def _silent(_: str) -> None:
    return None


@dataclass
class TrainResult:
    model: Surrogate
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    best_epoch: int = -1
    seconds: float = 0.0


def _tensors(
    ds: Dataset, norm: Normalizer, device: torch.device
) -> tuple[Tensor, Tensor, Tensor]:
    return (norm.theta(ds.theta).to(device), norm.current(ds.current).to(device),
            norm.state(ds.states)[:, :, 0].to(device))


@torch.no_grad()
def _val_mse(model: Surrogate, th: Tensor, cur: Tensor, v: Tensor, batch: int = 256) -> float:
    model.eval()
    total = 0.0
    for i in range(0, th.shape[0], batch):
        pred = model(th[i : i + batch], cur[i : i + batch])
        total += float(((pred - v[i : i + batch]) ** 2).sum())
    return total / v.numel()


def train_gru(
    train: Dataset,
    val: Dataset,
    norm: Normalizer,
    cfg: GRUConfig,
    seed: int,
    log: Logger = _silent,
    device: torch.device | str = "cpu",
) -> TrainResult:
    """Train a :class:`GRUSurrogate` end-to-end on full traces (MSE on V)."""
    device = torch.device(device)
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    model = GRUSurrogate(n_params=train.theta.shape[1], hidden=cfg.hidden, layers=cfg.layers,
                         patch=cfg.patch).to(device)
    th, cur, v = _tensors(train, norm, device)
    vth, vcur, vv = _tensors(val, norm, device)
    n = th.shape[0]
    steps_per_epoch = max(1, (n + cfg.batch_size - 1) // cfg.batch_size)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg.lr, total_steps=cfg.epochs * steps_per_epoch, pct_start=0.15
    )
    res = TrainResult(model=model)
    best = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    t0 = time.perf_counter()
    for epoch in range(cfg.epochs):
        model.train()
        perm = torch.randperm(n, generator=gen)
        running = 0.0
        for s in range(steps_per_epoch):
            idx = perm[s * cfg.batch_size : (s + 1) * cfg.batch_size].to(device)
            pred = model(th[idx], cur[idx])
            loss = torch.mean((pred - v[idx]) ** 2)
            opt.zero_grad(set_to_none=True)
            loss.backward()  # type: ignore[no-untyped-call]
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            running += loss.item()
        res.train_loss.append(running / steps_per_epoch)
        res.val_loss.append(_val_mse(model, vth, vcur, vv))
        if res.val_loss[-1] < best:
            best, res.best_epoch = res.val_loss[-1], epoch
            best_state = copy.deepcopy(model.state_dict())
        log(f"  gru seed={seed} epoch {epoch + 1}/{cfg.epochs} "
            f"train={res.train_loss[-1]:.4f} val={res.val_loss[-1]:.4f}")
    model.load_state_dict(best_state)
    res.seconds = time.perf_counter() - t0
    return res


def node_deriv_scale(train: Dataset, norm: Normalizer) -> list[float]:
    """Typical magnitude of ``dx_n/dt`` per state (used to scale the MLP output)."""
    s = norm.state(train.states).numpy()
    s0 = norm.state(train.x0[:, None, :]).numpy()
    full = np.concatenate([s0, s], axis=1)
    d = np.diff(full, axis=1) / train.dt
    return [float(x) for x in d.reshape(-1, 4).std(axis=0)]


def _window_schedule(cfg: NODEConfig) -> list[int]:
    """Curriculum: window length used at each epoch (short -> long)."""
    k = len(cfg.windows)
    return [cfg.windows[min(k - 1, e * k // max(cfg.epochs, 1))] for e in range(cfg.epochs)]


def train_node(
    train: Dataset,
    val: Dataset,
    norm: Normalizer,
    cfg: NODEConfig,
    seed: int,
    log: Logger = _silent,
    n_val_rollout: int = 64,
    device: torch.device | str = "cpu",
) -> TrainResult:
    """Train a :class:`NeuralODESurrogate` by multiple shooting on (V, m, h, n).

    Each iteration starts short RK4 rollouts (``cfg.windows``, increasing over
    training) from *true* states at random times and penalises the error of
    the whole state trajectory; a small MLP learns the resting state from the
    parameters. Model selection uses full-length rollouts on validation data.
    """
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    device = torch.device(device)
    model = NeuralODESurrogate(
        n_params=train.theta.shape[1], width=cfg.width, depth=cfg.depth, dt=train.dt,
        substeps=cfg.substeps, deriv_scale=node_deriv_scale(train, norm),
    ).to(device)
    th = norm.theta(train.theta).to(device)
    cur = norm.current(train.current).to(device)
    full = torch.cat([norm.state(train.x0[:, None, :]), norm.state(train.states)],
                     dim=1).to(device)
    n_traces, n_steps = cur.shape
    sub = slice(0, min(n_val_rollout, len(val)))
    vth = norm.theta(val.theta[sub]).to(device)
    vcur = norm.current(val.current[sub]).to(device)
    vv = norm.state(val.states[sub])[:, :, 0].to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-6)
    total = cfg.epochs * cfg.iters_per_epoch
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(total, 1))
    schedule = _window_schedule(cfg)
    res = TrainResult(model=model)
    best = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    t0 = time.perf_counter()
    for epoch in range(cfg.epochs):
        model.train()
        k_win = min(schedule[epoch], n_steps)
        running, n_ok = 0.0, 0
        for _ in range(cfg.iters_per_epoch):
            # Indices are drawn on the CPU generator (reproducible on any device).
            b = torch.randint(0, n_traces, (cfg.batch_size,), generator=gen).to(device)
            j = torch.randint(0, n_steps - k_win + 1, (cfg.batch_size,),
                              generator=gen).to(device)
            offs = j[:, None] + torch.arange(k_win, device=device)[None, :]
            x = full[b, j]
            target = full[b[:, None], offs + 1]
            i_win = cur[b[:, None], offs]
            pred = model.rollout(x, i_win, th[b])
            loss = torch.mean((pred - target) ** 2)
            loss = loss + torch.mean((model.initial_state(th[b]) - full[b, 0]) ** 2)
            opt.zero_grad(set_to_none=True)
            if not torch.isfinite(loss):
                sched.step()
                continue
            loss.backward()  # type: ignore[no-untyped-call]
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            running += loss.item()
            n_ok += 1
        res.train_loss.append(running / max(n_ok, 1))
        val_mse = _val_mse(model, vth, vcur, vv)
        res.val_loss.append(val_mse if np.isfinite(val_mse) else float("inf"))
        if res.val_loss[-1] < best:
            best, res.best_epoch = res.val_loss[-1], epoch
            best_state = copy.deepcopy(model.state_dict())
        log(f"  node seed={seed} epoch {epoch + 1}/{cfg.epochs} window={k_win} "
            f"train={res.train_loss[-1]:.4f} val(full rollout)={res.val_loss[-1]:.4f}")
    model.load_state_dict(best_state)
    res.seconds = time.perf_counter() - t0
    return res
