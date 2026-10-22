"""Double DQN for Snake in numpy: features, rewards, network and offline training.

DDQN.md explains the design; section numbers below refer to it. Training
runs on the game logic alone, without Ollama or pygame.

Usage:
    python ddqn.py --episodes 1500 --seed 0 --out ddqn_weights.npz
    python snake_jev.py --headless --ddqn ddqn_weights.npz --ddqn-only   # play it
"""

from __future__ import annotations

import argparse
import random
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from snake_jev import MOVES, Game, starved

# Section 3: actions are relative to the heading; the network's outputs are in this order.
ACTIONS = ("left", "straight", "right")
LEFT_OF = {"UP": "LEFT", "LEFT": "DOWN", "DOWN": "RIGHT", "RIGHT": "UP"}
RIGHT_OF = {turned: heading for heading, turned in LEFT_OF.items()}

# Section 4: the per-action features are listed straight, left, right.
FEATURE_ACTIONS = ("straight", "left", "right")
N_FEATURES = 17
LAYER_SIZES = (N_FEATURES, 64, 64, len(ACTIONS))

# Section 5
APPLE, DEATH, STEP = 10.0, -10.0, -0.01


def absolute(heading: str, action: str) -> str:
    """The game move for a relative action."""
    if action == "left":
        return LEFT_OF[heading]
    if action == "right":
        return RIGHT_OF[heading]
    return heading


def features(game: Game) -> np.ndarray:
    """The 17-number state `s`: everything relative to the head, nothing tied to the board size."""
    moves = [absolute(game.heading, a) for a in FEATURE_ACTIONS]
    danger = [float(not game.is_safe(m)) for m in moves]
    heading = [float(game.heading == h) for h in MOVES]
    food = [0.0] * 4
    if game.food is not None:
        dx, dy = game.food[0] - game.snake[0][0], game.food[1] - game.snake[0][1]
        food = [float(dy < 0), float(dy > 0), float(dx < 0), float(dx > 0)]
    tail, space = [], []
    for m in moves:
        # Same answers as Game.tail_reachable and Game.free_space, with one
        # peek per move instead of two: features run on every training step.
        future = game.peek(m)
        if future.alive:
            region = future._head_region()
            tail.append(float(future.snake[-1] in region))
            space.append(min(1.0, (len(region) - 1) / len(game.snake)))
        else:
            tail.append(float(future.food is None))  # filled the board: won
            space.append(0.0)
    return np.array(danger + heading + food + tail + space, dtype=np.float32)


def play(game: Game, action: int) -> tuple[float, bool]:
    """Play ACTIONS[action]; returns (reward, done). Starvation is left to the caller."""
    score = game.score
    game.step(absolute(game.heading, ACTIONS[action]))
    if game.score > score:
        return APPLE, not game.alive  # still alive unless that apple filled the board
    if not game.alive:
        return DEATH, True
    return STEP, False


# --------------------------------------------------------------------------
# Network and optimiser (section 6)
# --------------------------------------------------------------------------


class QNetwork:
    """MLP with ReLU hidden layers and a linear output: one Q-value per action."""

    def __init__(self, rng: np.random.Generator, sizes: tuple[int, ...] = LAYER_SIZES) -> None:
        self.params: list[np.ndarray] = []  # W0, b0, W1, b1, ...
        for i, (fan_in, fan_out) in enumerate(zip(sizes, sizes[1:])):
            w = rng.normal(0.0, np.sqrt(2.0 / fan_in), (fan_in, fan_out))
            if i == len(sizes) - 2:
                w *= 0.01  # every initial Q-value starts close to 0
            self.params += [w.astype(np.float32), np.zeros(fan_out, dtype=np.float32)]

    def activations(self, x: np.ndarray) -> list[np.ndarray]:
        """Input, each hidden layer's output, then the Q-values."""
        acts = [x]
        layers = len(self.params) // 2
        for i in range(layers):
            z = acts[-1] @ self.params[2 * i] + self.params[2 * i + 1]
            acts.append(z if i == layers - 1 else np.maximum(z, 0.0))
        return acts

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self.activations(x)[-1]

    def gradients(self, acts: list[np.ndarray], dq: np.ndarray) -> list[np.ndarray]:
        """Backpropagate dLoss/dQ through the layers that produced `acts`."""
        grads: list[np.ndarray] = [np.empty(0)] * len(self.params)
        delta = dq
        for i in reversed(range(len(self.params) // 2)):
            grads[2 * i] = acts[i].T @ delta
            grads[2 * i + 1] = delta.sum(axis=0)
            if i:
                delta = (delta @ self.params[2 * i].T) * (acts[i] > 0)
        return grads

    def copy(self) -> QNetwork:
        clone = object.__new__(QNetwork)
        clone.params = [p.copy() for p in self.params]
        return clone

    def save(self, path: str) -> None:
        np.savez(path, *self.params)

    @classmethod
    def load(cls, path: str) -> QNetwork:
        with np.load(path) as data:
            net = object.__new__(cls)
            net.params = [data[f"arr_{i}"] for i in range(len(data.files))]
        return net


class Adam:
    def __init__(self, params: list[np.ndarray], lr: float = 1e-3, beta1: float = 0.9,
                 beta2: float = 0.999, eps: float = 1e-8) -> None:
        self.params, self.lr, self.beta1, self.beta2, self.eps = params, lr, beta1, beta2, eps
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, grads: list[np.ndarray]) -> None:
        self.t += 1
        correction1, correction2 = 1 - self.beta1 ** self.t, 1 - self.beta2 ** self.t
        for p, g, m, v in zip(self.params, grads, self.m, self.v):
            m *= self.beta1
            m += (1 - self.beta1) * g
            v *= self.beta2
            v += (1 - self.beta2) * g * g
            p -= self.lr * (m / correction1) / (np.sqrt(v / correction2) + self.eps)


# --------------------------------------------------------------------------
# Training (sections 2, 7 and 8)
# --------------------------------------------------------------------------


class ReplayBuffer:
    """The last `capacity` transitions (s, a, r, s', done), overwritten oldest first."""

    def __init__(self, capacity: int) -> None:
        self.s = np.zeros((capacity, N_FEATURES), dtype=np.float32)
        self.a = np.zeros(capacity, dtype=np.int64)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.s2 = np.zeros((capacity, N_FEATURES), dtype=np.float32)
        self.done = np.zeros(capacity, dtype=np.float32)
        self.capacity, self.size, self.pos = capacity, 0, 0

    def __len__(self) -> int:
        return self.size

    def add(self, s: np.ndarray, a: int, r: float, s2: np.ndarray, done: bool) -> None:
        i = self.pos
        self.s[i], self.a[i], self.r[i], self.s2[i], self.done[i] = s, a, r, s2, done
        self.pos = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, n: int, rng: np.random.Generator) -> tuple[np.ndarray, ...]:
        idx = rng.integers(0, self.size, n)
        return self.s[idx], self.a[idx], self.r[idx], self.s2[idx], self.done[idx]


def double_dqn_targets(online: QNetwork, target: QNetwork, r: np.ndarray, s2: np.ndarray,
                       done: np.ndarray, gamma: float) -> np.ndarray:
    # The online network chooses the next action, the target network scores it.
    best = online.forward(s2).argmax(axis=1)
    value = target.forward(s2)[np.arange(len(best)), best]
    return r + gamma * value * (1.0 - done)


def train_step(online: QNetwork, target: QNetwork, opt: Adam,
               batch: tuple[np.ndarray, ...], gamma: float) -> float:
    """One Adam step on the mean of (Q_online(s, a) - target)²; returns that loss."""
    s, a, r, s2, done = batch
    y = double_dqn_targets(online, target, r, s2, done, gamma)
    acts = online.activations(s)
    rows = np.arange(len(a))
    error = acts[-1][rows, a] - y
    # Only the action actually played gets a gradient.
    dq = np.zeros_like(acts[-1])
    dq[rows, a] = 2.0 * error / len(a)
    opt.step(online.gradients(acts, dq))
    return float(np.mean(error ** 2))


@dataclass
class TrainConfig:
    episodes: int = 1500
    seed: int = 0
    gamma: float = 0.9
    lr: float = 1e-3
    batch: int = 64
    buffer: int = 50_000
    warmup: int = 1_000  # transitions in the buffer before training starts
    target_every: int = 1_000  # K: training steps between copies to Q_target
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay: float = 0.6  # share of the episodes over which ε falls to eps_end
    min_board: tuple[int, int] = (8, 6)
    max_board: tuple[int, int] = (20, 15)
    log_every: int = 50


def epsilon(episode: int, cfg: TrainConfig) -> float:
    fraction = min(1.0, episode / max(1.0, cfg.eps_decay * cfg.episodes))
    return cfg.eps_start + fraction * (cfg.eps_end - cfg.eps_start)


def train(cfg: TrainConfig, log: Callable[[str], None] | None = print) -> tuple[QNetwork, list[tuple[int, float]]]:
    """Train Q_online; returns it and (score, undiscounted reward) for every episode."""
    rng = np.random.default_rng(cfg.seed)
    online = QNetwork(rng)
    target = online.copy()
    opt = Adam(online.params, cfg.lr)
    replay = ReplayBuffer(cfg.buffer)
    history: list[tuple[int, float]] = []
    updates, t0 = 0, time.perf_counter()
    for episode in range(cfg.episodes):
        eps = epsilon(episode, cfg)
        cols = int(rng.integers(cfg.min_board[0], cfg.max_board[0] + 1))
        rows = int(rng.integers(cfg.min_board[1], cfg.max_board[1] + 1))
        game = Game(cols, rows, rng=random.Random(int(rng.integers(2 ** 32))))
        s, total, done = features(game), 0.0, False
        # A starvation cut-off ends the episode but is stored with done=False (section 5).
        while not done and not starved(game):
            if rng.random() < eps:
                a = int(rng.integers(len(ACTIONS)))
            else:
                a = int(online.forward(s[None])[0].argmax())
            r, done = play(game, a)
            s2 = features(game)
            replay.add(s, a, r, s2, done)
            total += r
            s = s2
            if len(replay) >= cfg.warmup:
                train_step(online, target, opt, replay.sample(cfg.batch, rng), cfg.gamma)
                updates += 1
                if updates % cfg.target_every == 0:
                    target = online.copy()
        history.append((game.score, total))
        if log and (episode + 1) % cfg.log_every == 0:
            recent = history[-cfg.log_every:]
            log(
                f"episode {episode + 1}/{cfg.episodes}  ε={eps:.2f}  "
                f"mean score {np.mean([h[0] for h in recent]):.1f}  "
                f"mean reward {np.mean([h[1] for h in recent]):.1f}  "
                f"updates {updates}  {time.perf_counter() - t0:.0f}s"
            )
    return online, history


def rank_moves(net: QNetwork, game: Game) -> list[str]:
    """The three possible moves as game moves, highest Q-value first."""
    q = net.forward(features(game)[None])[0]
    return [absolute(game.heading, ACTIONS[i]) for i in np.argsort(-q, kind="stable")]


def main(argv: list[str] | None = None) -> None:
    defaults = TrainConfig()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--episodes", type=int, default=defaults.episodes)
    p.add_argument("--seed", type=int, default=defaults.seed)
    p.add_argument("--out", default="ddqn_weights.npz", help="where to save Q_online")
    p.add_argument("--log-every", type=int, default=defaults.log_every)
    args = p.parse_args(argv)
    cfg = TrainConfig(episodes=args.episodes, seed=args.seed, log_every=args.log_every)
    net, _ = train(cfg)
    net.save(args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
