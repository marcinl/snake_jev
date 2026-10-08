import random
import unittest

from snake_jev import Game, build_request, choose_move


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
