import contextlib
import io
import json
import random
import unittest
from unittest import mock

from snake_jev import (
    Game, JevClient, Player, build_request, choose_move, parse_args, timing_summary,
)


def make_game(cols: int = 6, rows: int = 4) -> Game:
    return Game(cols, rows, rng=random.Random(0))


class GameTests(unittest.TestCase):
    def test_start_length_two_heading_right(self) -> None:
        g = make_game()
        self.assertEqual(g.snake, [(3, 2), (2, 2)])
        self.assertEqual(g.heading, "RIGHT")
        self.assertNotIn(g.food, g.snake)

    def test_wall_collision(self) -> None:
        g = make_game()
        g.food = (0, 0)
        g.step("RIGHT")  # (4,2)
        g.step("RIGHT")  # (5,2)
        self.assertEqual(g.outcome("RIGHT"), "wall")
        g.step("RIGHT")
        self.assertFalse(g.alive)

    def test_reverse_is_ignored_and_not_safe(self) -> None:
        g = make_game()
        g.food = (0, 0)
        self.assertFalse(g.is_safe("LEFT"))
        g.step("LEFT")
        self.assertTrue(g.alive)
        self.assertEqual(g.snake[0], (4, 2))
        self.assertEqual(g.heading, "RIGHT")

    def test_eating_grows_and_respawns(self) -> None:
        g = make_game()
        g.food = (4, 2)
        g.step("RIGHT")
        self.assertEqual(len(g.snake), 3)
        self.assertEqual(g.score, 1)
        self.assertNotIn(g.food, g.snake)

    def test_moving_into_vacating_tail_is_safe(self) -> None:
        g = make_game()
        g.snake = [(2, 1), (3, 1), (3, 2), (2, 2)]  # head with tail directly below
        g.heading = "LEFT"
        g.food = (0, 0)
        self.assertEqual(g.outcome("DOWN"), "free")

    def test_ascii_board_dimensions(self) -> None:
        g = make_game(7, 5)
        lines = g.render_ascii().splitlines()
        self.assertEqual(len(lines), 5 + 2)
        self.assertTrue(all(len(line) == 7 + 2 for line in lines))

    def test_food_relative_uses_move_words(self) -> None:
        g = make_game()
        g.food = (5, 0)  # head at (3,2)
        self.assertEqual(g.food_relative(), "food is 2 row(s) UP and 2 column(s) RIGHT from the head")


class QueryTests(unittest.TestCase):
    def test_request_shape_and_text_only_state(self) -> None:
        req = build_request(make_game(), "tev1:0.8b")
        self.assertEqual(req["model"], "tev1:0.8b")
        self.assertTrue(all(isinstance(v, str) for v in req["state"].values()))
        move = req["questions"]["move"]
        self.assertEqual(move["type"], "choice")
        self.assertEqual(set(move["criteria"]), {"UP", "DOWN", "LEFT", "RIGHT"})

    def test_no_assist_omits_hints(self) -> None:
        state = build_request(make_game(), "m", assist=False)["state"]
        self.assertNotIn("next_cell", state)

    def test_static_fields_come_first(self) -> None:
        for compact in (False, True):
            state = build_request(make_game(), "m", compact=compact)["state"]
            self.assertEqual(list(state)[0], "rules")
            self.assertLess(list(state).index("board_size"), list(state).index("heading"))

    def test_compact_drops_board_but_keeps_hints(self) -> None:
        full = build_request(make_game(10, 8), "m")["state"]
        state = build_request(make_game(10, 8), "m", compact=True)["state"]
        for key in ("board", "legend", "head_position", "food_position", "snake_length"):
            self.assertNotIn(key, state)
        self.assertIn("next_cell", state)
        self.assertIn("food_relative_to_head", state)
        self.assertLess(len(json.dumps(state)), len(json.dumps(full)))

    def test_compact_or_lookahead_reject_no_assist(self) -> None:
        for flag in ("--compact", "--lookahead"):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args([flag, "--no-assist"])


class LookaheadRequestTests(unittest.TestCase):
    def test_one_branch_per_free_move_skipping_reverse_and_wall(self) -> None:
        g = make_game()
        g.snake = [(5, 2), (4, 2)]  # facing the right wall
        g.food = (0, 0)
        req = build_request(g, "m", lookahead=True)
        self.assertEqual(set(req["questions"]), {"move", "move_after_UP", "move_after_DOWN"})
        self.assertIn("if_head_moves_UP", req["state"])
        self.assertNotIn("if_head_moves_RIGHT", req["state"])

    def test_no_branch_into_food(self) -> None:
        g = make_game()
        g.food = (4, 2)  # directly RIGHT of the head
        self.assertNotIn("move_after_RIGHT", build_request(g, "m", lookahead=True)["questions"])

    def test_branch_hints_describe_the_next_position(self) -> None:
        g = make_game()
        g.food = (5, 0)  # head at (3,2)
        hint = build_request(g, "m", lookahead=True)["state"]["if_head_moves_UP"]
        self.assertIn("moving UP", hint)
        self.assertIn("food is 1 row(s) UP and 2 column(s) RIGHT", hint)

    def test_peek_leaves_real_game_untouched(self) -> None:
        g = make_game()
        before = g.state_key()
        g.peek("UP")
        self.assertEqual(g.state_key(), before)


def answer(choice: str) -> dict:
    return {"choice": choice, "probabilities": {m: float(m == choice) for m in ("UP", "DOWN", "LEFT", "RIGHT")}}


class FakeClient:
    model = "fake"

    def __init__(self, answers: dict) -> None:
        self.answers = answers
        self.calls = 0

    def ask(self, payload: dict) -> dict:
        self.calls += 1
        return {"answers": {k: v for k, v in self.answers.items() if k in payload["questions"]}}


class PlayerTests(unittest.TestCase):
    def test_lookahead_halves_requests(self) -> None:
        g = make_game()
        g.food = (0, 0)
        client = FakeClient({"move": answer("UP"), "move_after_UP": answer("RIGHT")})
        player = Player(client, lookahead=True)
        d1 = player.decide(g)
        g.step(d1.move)
        d2 = player.decide(g)
        self.assertEqual((d1.move, d2.move), ("UP", "RIGHT"))
        self.assertTrue(d2.from_lookahead)
        self.assertEqual(client.calls, 1)
        g.step(d2.move)
        self.assertFalse(player.decide(g).from_lookahead)  # chain stops after one buffered move
        self.assertEqual(client.calls, 2)

    def test_followup_dropped_when_state_differs(self) -> None:
        g = make_game()
        g.food = (0, 0)
        client = FakeClient({"move": answer("UP"), "move_after_UP": answer("RIGHT")})
        player = Player(client, lookahead=True)
        player.decide(g)
        g.step("DOWN")  # not the move the follow-up was computed for
        self.assertFalse(player.decide(g).from_lookahead)
        self.assertEqual(client.calls, 2)

    def test_followup_uses_guard_override_branch(self) -> None:
        g = make_game()
        g.snake = [(5, 2), (4, 2)]  # facing the right wall: guard turns RIGHT into DOWN
        g.food = (0, 0)
        right_then_down = {"choice": "RIGHT", "probabilities": {"RIGHT": 0.6, "DOWN": 0.3, "UP": 0.1, "LEFT": 0.0}}
        client = FakeClient({"move": right_then_down, "move_after_DOWN": answer("LEFT")})
        player = Player(client, lookahead=True)
        d1 = player.decide(g)
        self.assertEqual(d1.move, "DOWN")
        g.step(d1.move)
        d2 = player.decide(g)
        self.assertTrue(d2.from_lookahead)
        self.assertEqual(d2.move, "LEFT")

    def test_guard_still_checks_followup(self) -> None:
        g = make_game()
        g.snake = [(4, 1), (3, 1)]  # one UP from (4,1) is the top row
        g.food = (0, 3)
        client = FakeClient({"move": answer("UP"), "move_after_UP": answer("UP")})
        player = Player(client, lookahead=True)
        g.step(player.decide(g).move)
        d2 = player.decide(g)
        self.assertTrue(d2.from_lookahead)
        self.assertTrue(d2.overridden)
        self.assertTrue(g.is_safe(d2.move))

    def test_no_followup_without_lookahead(self) -> None:
        g = make_game()
        g.food = (0, 0)
        client = FakeClient({"move": answer("UP"), "move_after_UP": answer("RIGHT")})
        player = Player(client)
        g.step(player.decide(g).move)
        self.assertFalse(player.decide(g).from_lookahead)
        self.assertEqual(client.calls, 2)


class TimingTests(unittest.TestCase):
    def test_ask_records_timing_and_server_fields(self) -> None:
        body = {"answers": {}, "usage": {"input_tokens": 300}, "prompt_eval_duration": 250_000_000}
        resp = mock.MagicMock()
        resp.__enter__.return_value.read.return_value = json.dumps(body).encode()
        client = JevClient("m", debug_timing=True)
        with mock.patch("snake_jev.urllib.request.urlopen", return_value=resp), \
                self.assertLogs("snake_jev", "INFO") as logs:
            self.assertEqual(client.ask({"questions": {"move": {}}}), body)
        (t,) = client.timings
        self.assertEqual(t.questions, 1)
        self.assertEqual(t.server, {"input_tokens": 300, "prompt_eval_ms": 250.0})
        self.assertGreaterEqual(t.total_ms, t.ttfb_ms)
        self.assertIn("request #1", logs.output[0])

    def test_no_timing_recorded_by_default(self) -> None:
        resp = mock.MagicMock()
        resp.__enter__.return_value.read.return_value = b'{"answers": {}}'
        client = JevClient("m")
        with mock.patch("snake_jev.urllib.request.urlopen", return_value=resp):
            client.ask({"questions": {}})
        self.assertEqual(client.timings, [])

    def test_summary_percentiles(self) -> None:
        from snake_jev import RequestTiming
        timings = [RequestTiming(100, 1, 0, ms, 0, ms, {}) for ms in (100, 200, 300, 400)]
        summary = timing_summary(timings)
        self.assertIn("4 requests", summary)
        self.assertIn("total mean=250 p50=200 p95=400 max=400ms", summary)
        self.assertEqual(timing_summary([]), "summary: no requests")


class GuardTests(unittest.TestCase):
    def test_guard_falls_back_to_next_most_probable_safe_move(self) -> None:
        g = make_game()
        g.snake = [(5, 2), (4, 2)]  # facing the right wall
        answer = {
            "choice": "RIGHT",
            "probabilities": {"RIGHT": 0.5, "LEFT": 0.3, "UP": 0.05, "DOWN": 0.15},
        }
        # LEFT is a reverse, so the guard skips it and takes DOWN.
        self.assertEqual(choose_move(g, answer), ("DOWN", True))
        self.assertEqual(choose_move(g, answer, guard=False), ("RIGHT", False))


if __name__ == "__main__":
    unittest.main()
