import contextlib
import io
import json
import random
import unittest
from unittest import mock

from snake_jev import (
    Decision, DDQNPlayer, Game, JevClient, ModelStats, Player, SearchPlayer, bfs_path, build_request,
    choose_move, hybrid_move, main, parse_args, run_headless, safe_food_path, search_move, starved,
    timing_summary,
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

    def test_move_hints_say_whether_the_move_nears_the_food(self) -> None:
        g = make_game()
        g.snake = [(5, 2), (4, 2)]  # facing the right wall, head at (5,2)
        g.food = (5, 0)
        self.assertEqual(
            g.describe_moves(),
            "UP: free, towards the food; DOWN: free, away from the food; "
            "LEFT: not allowed (reverse); RIGHT: wall, game over",
        )
        g.food = (5, 1)
        self.assertEqual(g.move_hint("UP"), "food, eats it")


class QueryTests(unittest.TestCase):
    def test_request_shape_and_text_only_state(self) -> None:
        req = build_request(make_game(), "tev1:0.8b")
        self.assertEqual(req["model"], "tev1:0.8b")
        self.assertTrue(all(isinstance(v, str) for v in req["state"].values()))
        move = req["questions"]["move"]
        self.assertEqual(move["type"], "choice")
        self.assertEqual(set(move["criteria"]), {"UP", "DOWN", "LEFT", "RIGHT"})

    def test_no_assist_omits_hints(self) -> None:
        req = build_request(make_game(), "m", assist=False)
        self.assertNotIn("next_cell", req["state"])
        self.assertEqual(req["questions"]["move"]["criteria"]["UP"], "Move the head one row up")

    def test_criteria_repeat_the_move_hints(self) -> None:
        g = make_game()
        g.food = (3, 0)  # straight UP from the head at (3,2)
        criteria = build_request(g, "m")["questions"]["move"]["criteria"]
        self.assertEqual(criteria["UP"], "Move the head one row up: free, towards the food")
        self.assertEqual(criteria["DOWN"], "Move the head one row down: free, away from the food")
        self.assertEqual(criteria["LEFT"], "Move the head one column left: not allowed (reverse)")

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

    def test_branch_criteria_describe_the_next_position(self) -> None:
        g = make_game()
        g.food = (3, 0)  # head at (3,2); after UP the food is one cell further UP
        criteria = build_request(g, "m", lookahead=True)["questions"]["move_after_UP"]["criteria"]
        self.assertEqual(criteria["UP"], "Move the head one row up: food, eats it")
        self.assertEqual(criteria["DOWN"], "Move the head one row down: not allowed (reverse)")

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

    def test_tail_guard_rejects_move_into_pocket(self) -> None:
        g = pocket_game()
        # LEFT heads for the food but into the pocket; plain guard allows it.
        self.assertEqual(choose_move(g, answer_for("LEFT")), ("LEFT", False))
        self.assertEqual(choose_move(g, answer_for("LEFT"), tail_guard=True), ("RIGHT", True))

    def test_tail_guard_keeps_a_tail_safe_choice(self) -> None:
        self.assertEqual(choose_move(pocket_game(), answer_for("RIGHT"), tail_guard=True), ("RIGHT", False))

    def test_tail_guard_picks_most_room_when_all_moves_trap(self) -> None:
        g = trapped_game()
        self.assertEqual(choose_move(g, answer_for("RIGHT"), tail_guard=True), ("LEFT", True))
        self.assertEqual(choose_move(g, answer_for("RIGHT")), ("RIGHT", False))

    def test_player_applies_tail_guard(self) -> None:
        g = pocket_game()
        d = Player(FakeClient({"move": answer("LEFT")}), tail_guard=True).decide(g)
        self.assertEqual((d.move, d.model_choice, d.overridden), ("RIGHT", "LEFT", True))

    def test_guard_hands_a_repeated_route_to_search(self) -> None:
        g = make_game()  # head (3,2) heading RIGHT
        g.food = (0, 0)
        for move in ("DOWN", "LEFT", "UP", "RIGHT"):
            g.step(move)
        self.assertTrue(g.route_repeated)
        # RIGHT is safe, so without the loop check the guard would keep it.
        self.assertEqual(choose_move(g, answer_for("RIGHT")), (search_move(g), True))
        self.assertEqual(choose_move(g, answer_for("RIGHT"), guard=False), ("RIGHT", False))

    def test_up_only_model_escapes_the_corner_cycle(self) -> None:
        # Reproduces the cycle between the top-right and bottom-right corners:
        # a model that always answers UP, food near the bottom-left corner.
        # The fallback order is the one the real model showed.
        up_first = {"choice": "UP", "probabilities": {"UP": 0.5, "DOWN": 0.25, "RIGHT": 0.15, "LEFT": 0.1}}
        g = Game(20, 15, rng=random.Random(0))
        g.snake = [(10 + i, 0) for i in range(4)]
        g.heading = "LEFT"
        g.food = (2, 12)
        player = Player(FakeClient({"move": up_first}))
        while g.score == 0 and g.alive and not starved(g):
            g.step(player.decide(g).move)
        self.assertEqual(g.score, 1)

    def test_tail_guard_rejects_no_guard(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(["--tail-guard", "--no-guard"])


def answer_for(choice: str) -> dict:
    # The other moves get small, distinct probabilities so the guard's fallback order is fixed.
    probs = {"UP": 0.03, "DOWN": 0.02, "LEFT": 0.01, "RIGHT": 0.04}
    probs[choice] = 0.9
    return {"choice": choice, "probabilities": probs}


def pocket_game() -> Game:
    """Head at the top of a body wall: LEFT enters a closed pocket, RIGHT doesn't.

        #F.H...#
        #..S...#
        #..SST.#
    """
    g = make_game(6, 3)
    g.snake = [(2, 0), (2, 1), (2, 2), (3, 2), (4, 2)]
    g.heading = "UP"
    g.food = (0, 0)
    return g


def trapped_game() -> Game:
    """Both safe moves lose the tail: LEFT leaves 7 cells, RIGHT 1.

        #F.SST#
        #..SSS#
        #..SSS#
        #..H..#
    """
    g = make_game(5, 4)
    g.snake = [(2, 3), (2, 2), (2, 1), (2, 0), (3, 0), (3, 1), (3, 2), (4, 2), (4, 1), (4, 0)]
    g.heading = "DOWN"
    g.food = (0, 0)
    return g


class SafetyHelperTests(unittest.TestCase):
    def test_tail_reachable_detects_pocket(self) -> None:
        g = pocket_game()
        self.assertFalse(g.tail_reachable("LEFT"))
        self.assertTrue(g.tail_reachable("RIGHT"))
        self.assertFalse(g.tail_reachable("UP"))  # wall

    def test_free_space_counts_region_after_move(self) -> None:
        g = pocket_game()
        self.assertEqual(g.free_space("LEFT"), 5)  # columns 0-1 minus the head
        self.assertEqual(g.free_space("RIGHT"), 8)  # columns 3-5, the vacated tail cell included
        self.assertEqual(g.free_space("UP"), 0)

    def test_trapped_position(self) -> None:
        g = trapped_game()
        self.assertEqual([m for m in ("LEFT", "RIGHT") if g.tail_reachable(m)], [])
        self.assertEqual((g.free_space("LEFT"), g.free_space("RIGHT")), (7, 1))

    def test_open_board_tail_always_reachable(self) -> None:
        g = make_game()
        g.food = (0, 0)
        self.assertTrue(all(g.tail_reachable(m) for m in ("UP", "DOWN", "RIGHT")))

    def test_helpers_leave_game_untouched(self) -> None:
        g = pocket_game()
        before = g.state_key()
        g.tail_reachable("LEFT")
        g.free_space("RIGHT")
        self.assertEqual(g.state_key(), before)

    def test_route_repeated_after_a_loop(self) -> None:
        g = make_game()  # head (3,2) heading RIGHT
        g.food = (0, 0)
        for move in ("DOWN", "LEFT", "UP"):
            g.step(move)
            self.assertFalse(g.route_repeated)
        g.step("RIGHT")  # back on (3,2) heading RIGHT
        self.assertTrue(g.route_repeated)
        g.step("RIGHT")  # (4,2) RIGHT is new
        self.assertFalse(g.route_repeated)
        self.assertTrue(g.looped)  # stays set until the next food

    def test_route_resets_on_food(self) -> None:
        g = make_game()
        g.food = (0, 0)
        for move in ("DOWN", "LEFT", "UP"):
            g.step(move)
        g.food = (3, 2)
        g.step("RIGHT")  # same position as the start, but it eats
        self.assertFalse(g.route_repeated)
        self.assertFalse(g.looped)
        self.assertEqual(g.score, 1)


class SearchTests(unittest.TestCase):
    def test_bfs_straight_line(self) -> None:
        g = make_game()  # head (3,2) heading RIGHT
        self.assertEqual(bfs_path(g, (5, 2)), ["RIGHT", "RIGHT"])

    def test_bfs_never_starts_with_a_reverse(self) -> None:
        g = make_game()
        path = bfs_path(g, (1, 2))  # straight behind the tail at (2,2)
        self.assertEqual(len(path), 4)
        self.assertNotEqual(path[0], "LEFT")

    def test_bfs_none_when_walled_off(self) -> None:
        future = pocket_game().peek("LEFT")  # head now inside the pocket
        self.assertIsNone(bfs_path(future, (5, 0)))

    def test_food_path_rejected_when_eating_traps_the_tail(self) -> None:
        g = pocket_game()  # food (0,0) is in the pocket
        self.assertEqual(bfs_path(g, g.food), ["LEFT", "LEFT"])
        self.assertIsNone(safe_food_path(g))
        self.assertEqual(search_move(g), "RIGHT")  # stalls on the open side instead

    def test_search_takes_safe_food_path(self) -> None:
        g = make_game()
        g.food = (5, 2)
        self.assertEqual(safe_food_path(g), ["RIGHT", "RIGHT"])
        self.assertEqual(search_move(g), "RIGHT")

    def test_search_last_resort_picks_most_room(self) -> None:
        self.assertEqual(search_move(trapped_game()), "LEFT")

    def test_search_never_crashes_on_small_board(self) -> None:
        wins = 0
        for seed in range(10):
            g = Game(6, 4, rng=random.Random(seed))
            player = SearchPlayer()
            while g.alive and not starved(g):
                g.step(player.decide(g).move)
            crashed = not g.alive and g.food is not None
            self.assertFalse(crashed, f"seed {seed}")
            wins += g.food is None
        self.assertGreaterEqual(wins, 5)

    def test_search_only_runs_without_ollama(self) -> None:
        out = io.StringIO()
        with mock.patch("snake_jev.urllib.request.urlopen", side_effect=AssertionError("no requests")), \
                contextlib.redirect_stdout(out):
            main(["--search-only", "--headless", "--cols", "6", "--rows", "4", "--seed", "0"])
        self.assertIn("score=22", out.getvalue())
        self.assertNotIn("played unchanged", out.getvalue())  # nothing to report without a model


def decision(choice: str, overridden: bool = False) -> Decision:
    return Decision(move=choice, model_choice=choice, probabilities={}, confidence=0.0,
                    latency_ms=0.0, overridden=overridden)


class ModelStatsTests(unittest.TestCase):
    def test_counts_unchanged_and_search_agreement(self) -> None:
        g = pocket_game()  # search plays RIGHT here
        stats = ModelStats()
        stats.record(g, decision("LEFT"))
        stats.record(g, decision("RIGHT"))
        stats.record(g, decision("UP", overridden=True))
        self.assertEqual((stats.decisions, stats.unchanged, stats.agrees_with_search), (3, 2, 1))

    def test_summary_shows_shares_and_settings(self) -> None:
        stats = ModelStats(decisions=4, unchanged=3, agrees_with_search=1)
        player = Player(FakeClient({}), compact=True, tail_guard=True)
        self.assertEqual(stats.summary(player), (
            "model fake | prompt: hints, compact | guard: tail\n"
            "  model's move played unchanged: 3/4 (75%)\n"
            "  model's choice matches search: 1/4 (25%)"
        ))

    def test_summary_with_no_decisions(self) -> None:
        self.assertIn("0/0 (0%)", ModelStats().summary(Player(FakeClient({}))))

    def test_headless_prints_stats_for_model_games(self) -> None:
        g = make_game()
        args = parse_args(["--headless"])
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            run_headless(g, Player(FakeClient({"move": answer("UP")})), args)
        text = out.getvalue()
        self.assertIn("model fake | prompt: hints, full board | guard: cell", text)
        self.assertIn("model's choice matches search:", text)
        self.assertNotIn("sources:", text)  # only reported with --ddqn


def ranker(*moves: str):
    """Stands in for a trained DDQN: always ranks the moves in this order."""
    return lambda game: list(moves)


def looped_game() -> Game:
    """Head back on its start position (3,2) heading RIGHT; route DOWN, LEFT, UP, RIGHT."""
    g = make_game()
    g.food = (0, 0)
    for move in ("DOWN", "LEFT", "UP", "RIGHT"):
        g.step(move)
    return g


class HybridTests(unittest.TestCase):
    def test_model_plays_a_tail_safe_move(self) -> None:
        self.assertEqual(hybrid_move(pocket_game(), answer_for("RIGHT"), ranker("LEFT")), ("RIGHT", "model"))

    def test_ddqn_takes_over_from_a_move_into_a_pocket(self) -> None:
        # The DDQN's own first choice is the pocket too, so its next one is played.
        self.assertEqual(
            hybrid_move(pocket_game(), answer_for("LEFT"), ranker("LEFT", "UP", "RIGHT")), ("RIGHT", "ddqn")
        )

    def test_ddqn_plays_once_the_route_repeats(self) -> None:
        g = looped_game()
        self.assertTrue(g.looped)
        # RIGHT is safe and new, but the model has had its chance since the last food.
        self.assertEqual(hybrid_move(g, answer_for("RIGHT"), ranker("UP", "RIGHT")), ("UP", "ddqn"))

    def test_search_when_the_ddqn_would_revisit(self) -> None:
        g = make_game()
        g.food = (0, 0)
        for move in ("DOWN", "LEFT", "UP"):
            g.step(move)  # head (2,2) heading UP; RIGHT returns to the start position
        self.assertTrue(g.revisits("RIGHT"))
        self.assertFalse(g.revisits("UP"))
        self.assertEqual(hybrid_move(g, answer_for("DOWN"), ranker("RIGHT")), (search_move(g), "search"))

    def test_player_records_the_source(self) -> None:
        player = Player(FakeClient({"move": answer("LEFT")}), ddqn=ranker("LEFT", "UP", "RIGHT"))
        d = player.decide(pocket_game())
        self.assertEqual((d.move, d.model_choice, d.overridden, d.source), ("RIGHT", "LEFT", True, "ddqn"))

    def test_summary_shows_source_shares(self) -> None:
        stats = ModelStats()
        g = pocket_game()
        for source in ("model", "model", "ddqn", "search"):
            stats.record(g, Decision(move="RIGHT", model_choice="RIGHT", probabilities={}, confidence=0.0,
                                     latency_ms=0.0, overridden=source != "model", source=source))
        summary = stats.summary(Player(FakeClient({}), ddqn=ranker("UP")))
        self.assertIn("guard: hybrid (ddqn, search)", summary)
        self.assertIn("sources: model=2/4 (50%) ddqn=1/4 (25%) search=1/4 (25%)", summary)

    def test_ddqn_player_plays_the_top_move_unchecked(self) -> None:
        d = DDQNPlayer(ranker("UP", "RIGHT")).decide(pocket_game())  # UP is the wall
        self.assertEqual((d.move, d.source), ("UP", "ddqn"))

    def test_ddqn_flag_combinations(self) -> None:
        for argv in (["--ddqn", "w.npz", "--no-guard"], ["--ddqn-only"],
                     ["--ddqn", "w.npz", "--ddqn-only", "--search-only"]):
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()), \
                    self.assertRaises(SystemExit):
                parse_args(argv)


if __name__ == "__main__":
    unittest.main()
