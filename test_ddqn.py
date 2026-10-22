import contextlib
import io
import os
import random
import tempfile
import unittest
from unittest import mock

import numpy as np

from ddqn import (
    ACTIONS, N_FEATURES, Adam, QNetwork, ReplayBuffer, TrainConfig, absolute, double_dqn_targets, epsilon,
    features, play, rank_moves, train, train_step,
)
from snake_jev import OPPOSITE, Game, main


def example_game() -> Game:
    """The 7×5 example state from DDQN.md section 4.

        #.......#
        #..F....#
        #...TSH.#
        #.......#
        #.......#
    """
    g = Game(7, 5, rng=random.Random(0))
    g.snake = [(5, 2), (4, 2), (3, 2)]
    g.heading = "RIGHT"
    g.food = (2, 1)
    return g


class FixedQ:
    """A network stub with fixed Q-values (left, straight, right) for every state."""

    def __init__(self, q: list[float]) -> None:
        self.q = np.array([q], dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.repeat(self.q, len(x), axis=0)


class ActionTests(unittest.TestCase):
    def test_relative_to_absolute_table(self) -> None:
        # DDQN.md section 3
        table = {
            "UP": ("LEFT", "UP", "RIGHT"),
            "RIGHT": ("UP", "RIGHT", "DOWN"),
            "DOWN": ("RIGHT", "DOWN", "LEFT"),
            "LEFT": ("DOWN", "LEFT", "UP"),
        }
        for heading, expected in table.items():
            self.assertEqual(tuple(absolute(heading, a) for a in ACTIONS), expected)

    def test_no_action_reverses(self) -> None:
        for heading in OPPOSITE:
            self.assertNotIn(OPPOSITE[heading], [absolute(heading, a) for a in ACTIONS])


class FeatureTests(unittest.TestCase):
    def test_example_state_from_the_doc(self) -> None:
        expected = [0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1.0, 1.0, 1.0]
        np.testing.assert_array_equal(features(example_game()), np.array(expected, dtype=np.float32))

    def test_next_state_from_the_worked_example(self) -> None:
        # DDQN.md section 7: after `left` (UP) the food is level with the head, to the left.
        g = example_game()
        play(g, ACTIONS.index("left"))
        expected = [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1.0, 1.0, 1.0]
        np.testing.assert_array_equal(features(g), np.array(expected, dtype=np.float32))

    def test_danger_straight_at_the_wall(self) -> None:
        g = example_game()
        g.snake = [(6, 2), (5, 2), (4, 2)]
        s = features(g)
        np.testing.assert_array_equal(s[0:3], [1, 0, 0])  # straight, left, right
        self.assertEqual(s[14], 0.0)  # no room after crashing

    def test_pocket_shows_in_tail_and_space(self) -> None:
        # Head heading UP at the top row: left enters column 0, a 3-cell pocket.
        #   #.H..F#
        #   #.S...#
        #   #.SST.#
        g = Game(5, 3, rng=random.Random(0))
        g.snake = [(1, 0), (1, 1), (1, 2), (2, 2), (3, 2)]
        g.heading = "UP"
        g.food = (4, 0)
        s = features(g)
        np.testing.assert_array_equal(s[11:14], [0, 0, 1])  # tail reachable: straight, left, right
        np.testing.assert_allclose(s[14:17], [0.0, 2 / 5, 1.0])  # 2 cells beyond the head, length 5

    def test_no_food_left(self) -> None:
        g = example_game()
        g.food = None
        np.testing.assert_array_equal(features(g)[7:11], [0, 0, 0, 0])

    def test_features_leave_the_game_untouched(self) -> None:
        g = example_game()
        before, state = g.state_key(), g.rng.getstate()
        features(g)
        self.assertEqual((g.state_key(), g.rng.getstate()), (before, state))


class RewardTests(unittest.TestCase):
    def test_step_apple_and_death(self) -> None:
        g = example_game()
        self.assertEqual(play(g, ACTIONS.index("left")), (-0.01, False))
        g.food = (4, 1)
        self.assertEqual(play(g, ACTIONS.index("left")), (10.0, False))
        self.assertEqual(g.score, 1)
        g.snake = [(0, 1), (1, 1), (2, 1)]
        g.heading = "LEFT"
        self.assertEqual(play(g, ACTIONS.index("straight")), (-10.0, True))

    def test_filling_the_board_is_a_rewarded_end(self) -> None:
        g = Game(3, 1, rng=random.Random(0))  # snake (1,0),(0,0); one free cell
        self.assertEqual(g.food, (2, 0))
        self.assertEqual(play(g, ACTIONS.index("straight")), (10.0, True))


class NetworkTests(unittest.TestCase):
    def test_untrained_network_has_no_opinion(self) -> None:
        net = QNetwork(np.random.default_rng(0))
        self.assertEqual(sum(p.size for p in net.params), 5507)  # DDQN.md section 6
        q = net.forward(features(example_game())[None])
        self.assertEqual(q.shape, (1, 3))
        self.assertLess(float(np.abs(q).max()), 0.05)
        for b in net.params[1::2]:
            self.assertFalse(b.any())

    def test_gradients_match_finite_differences(self) -> None:
        rng = np.random.default_rng(1)
        net = QNetwork(rng, sizes=(4, 5, 3))
        net.params = [p.astype(np.float64) for p in net.params]
        net.params[-2] *= 100  # undo the output scaling so the check isn't lost in rounding
        x, w = rng.normal(size=(6, 4)), rng.normal(size=(6, 3))

        def loss() -> float:
            return float((net.forward(x) * w).sum())

        grads = net.gradients(net.activations(x), w)
        for p, g in zip(net.params, grads):
            for idx in np.ndindex(p.shape):
                old = p[idx]
                p[idx] = old + 1e-6
                up = loss()
                p[idx] = old - 1e-6
                down = loss()
                p[idx] = old
                self.assertAlmostEqual(g[idx], (up - down) / 2e-6, places=5)

    def test_save_and_load(self) -> None:
        net = QNetwork(np.random.default_rng(0))
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "w.npz")
            net.save(path)
            loaded = QNetwork.load(path)
        for a, b in zip(net.params, loaded.params):
            np.testing.assert_array_equal(a, b)

    def test_copy_is_independent(self) -> None:
        net = QNetwork(np.random.default_rng(0))
        clone = net.copy()
        net.params[0] += 1
        self.assertFalse(np.array_equal(net.params[0], clone.params[0]))


class TargetTests(unittest.TestCase):
    def test_worked_example(self) -> None:
        # DDQN.md section 7: online prefers left (3.4), target scores it 2.6.
        online, target = FixedQ([3.4, 2.1, 1.0]), FixedQ([2.6, 2.3, 2.9])
        s2 = np.zeros((4, N_FEATURES), dtype=np.float32)
        r = np.array([-0.01, 10.0, -10.0, -0.01], dtype=np.float32)
        done = np.array([0, 0, 1, 0], dtype=np.float32)
        y = double_dqn_targets(online, target, r, s2, done, gamma=0.9)
        np.testing.assert_allclose(y, [2.33, 12.34, -10.0, 2.33], atol=1e-5)

    def test_train_step_moves_q_toward_the_target(self) -> None:
        rng = np.random.default_rng(0)
        online = QNetwork(rng)
        target = online.copy()
        opt = Adam(online.params)
        s = rng.random((8, N_FEATURES)).astype(np.float32)
        batch = (s, np.zeros(8, dtype=np.int64), np.full(8, 10.0, dtype=np.float32), s,
                 np.ones(8, dtype=np.float32))
        losses = [train_step(online, target, opt, batch, 0.9) for _ in range(50)]
        self.assertLess(losses[-1], losses[0] / 2)


class ReplayTests(unittest.TestCase):
    def test_overwrites_oldest_when_full(self) -> None:
        buf = ReplayBuffer(3)
        s = np.zeros(N_FEATURES, dtype=np.float32)
        for i in range(5):
            buf.add(s, i % 3, float(i), s, False)
        self.assertEqual(len(buf), 3)
        self.assertEqual(sorted(buf.r.tolist()), [2.0, 3.0, 4.0])
        _, _, r, _, _ = buf.sample(10, np.random.default_rng(0))
        self.assertTrue(set(r.tolist()) <= {2.0, 3.0, 4.0})


class TrainingTests(unittest.TestCase):
    def small_config(self) -> TrainConfig:
        return TrainConfig(episodes=6, warmup=50, batch=16, target_every=20,
                           min_board=(6, 4), max_board=(7, 5))

    def test_epsilon_schedule(self) -> None:
        cfg = TrainConfig(episodes=100)
        self.assertEqual(epsilon(0, cfg), 1.0)
        self.assertAlmostEqual(epsilon(30, cfg), 0.525)
        self.assertAlmostEqual(epsilon(60, cfg), 0.05)
        self.assertAlmostEqual(epsilon(99, cfg), 0.05)

    def test_seeded_runs_repeat_exactly(self) -> None:
        net1, history1 = train(self.small_config(), log=None)
        net2, history2 = train(self.small_config(), log=None)
        self.assertEqual(history1, history2)
        for a, b in zip(net1.params, net2.params):
            np.testing.assert_array_equal(a, b)
        self.assertEqual(len(history1), 6)

    def test_rank_moves_lists_the_three_legal_moves(self) -> None:
        net = QNetwork(np.random.default_rng(0))
        ranked = rank_moves(net, example_game())
        self.assertEqual(sorted(ranked), ["DOWN", "RIGHT", "UP"])


class CommandLineTests(unittest.TestCase):
    def test_ddqn_only_plays_without_ollama(self) -> None:
        net = QNetwork(np.random.default_rng(0))
        out = io.StringIO()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "w.npz")
            net.save(path)
            with mock.patch("snake_jev.urllib.request.urlopen", side_effect=AssertionError("no requests")), \
                    contextlib.redirect_stdout(out):
                main(["--ddqn", path, "--ddqn-only", "--headless", "--cols", "6", "--rows", "4", "--seed", "0"])
        self.assertIn("score=", out.getvalue())


if __name__ == "__main__":
    unittest.main()
