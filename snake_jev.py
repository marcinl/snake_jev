"""Snake played by a Jev-style System One decision model served by Ollama.

Each game tick sends one request to Ollama's /v1/systemone endpoint:
the board as text (state) plus a single typed Choice question (move).
The model's per-option probabilities are then used by a safety guard so a
low-confidence answer that would crash can fall back to the next-best move.

Usage:
    python snake_jev.py --cols 20 --rows 15                # pygame window
    python snake_jev.py --cols 10 --rows 8 --headless      # console only
    python snake_jev.py --dump                             # print one request and exit
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import urllib.request
from dataclasses import dataclass, field

DEFAULT_MODEL = "nimble:9b-q4_K_M" #"tev1:0.8b"
DEFAULT_HOST = "http://localhost:11434"
START_LENGTH = 2

Point = tuple[int, int]  # (col, row); row grows downwards like screen coords

HEADINGS: dict[str, Point] = {"UP": (0, -1), "RIGHT": (1, 0), "DOWN": (0, 1), "LEFT": (-1, 0)}
OPPOSITE = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}

# Absolute directions, not straight/left/right: the state can then describe
# food and hazards with the same words as the answer options, and tev1:0.8b
# measurably picks food-ward moves more often (22/30 vs 17/30 in a probe).
MOVES = ("UP", "DOWN", "LEFT", "RIGHT")


# --------------------------------------------------------------------------
# Game logic (pure, no pygame / network)
# --------------------------------------------------------------------------


@dataclass
class Game:
    cols: int  # N
    rows: int  # M
    rng: random.Random = field(default_factory=random.Random)
    snake: list[Point] = field(init=False)  # snake[0] is the head
    heading: str = field(init=False)  # one of MOVES
    food: Point | None = field(init=False)
    alive: bool = field(init=False, default=True)
    score: int = field(init=False, default=0)
    steps_since_food: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        if self.cols < START_LENGTH + 1 or self.rows < 1:
            raise ValueError(f"board {self.cols}x{self.rows} too small for a length-{START_LENGTH} snake")
        head = (self.cols // 2, self.rows // 2)
        self.heading = "RIGHT"
        self.snake = [(head[0] - i, head[1]) for i in range(START_LENGTH)]
        self.food = self._spawn_food()

    def _spawn_food(self) -> Point | None:
        occupied = set(self.snake)
        free = [(c, r) for r in range(self.rows) for c in range(self.cols) if (c, r) not in occupied]
        return self.rng.choice(free) if free else None

    def in_bounds(self, p: Point) -> bool:
        return 0 <= p[0] < self.cols and 0 <= p[1] < self.rows

    def effective(self, move: str) -> str:
        # As on a real Snake keypad, reversing into the neck is ignored.
        return self.heading if move == OPPOSITE[self.heading] else move

    def next_head(self, move: str) -> Point:
        dx, dy = HEADINGS[self.effective(move)]
        return (self.snake[0][0] + dx, self.snake[0][1] + dy)

    def outcome(self, move: str) -> str:
        """'wall', 'body', 'food' or 'free' for the cell the move enters."""
        nxt = self.next_head(move)
        if not self.in_bounds(nxt):
            return "wall"
        eating = nxt == self.food
        # The tail vacates its cell this tick unless the snake grows.
        body = self.snake if eating else self.snake[:-1]
        if nxt in body:
            return "body"
        return "food" if eating else "free"

    def is_safe(self, move: str) -> bool:
        # A reverse is never "safe" on its own merits; the guard should pick
        # the real move it would turn into instead.
        return move != OPPOSITE[self.heading] and self.outcome(move) in ("free", "food")

    def step(self, move: str) -> None:
        if not self.alive:
            return
        result = self.outcome(move)
        nxt = self.next_head(move)
        self.heading = self.effective(move)
        if result in ("wall", "body"):
            self.alive = False
            return
        self.snake.insert(0, nxt)
        if result == "food":
            self.score += 1
            self.steps_since_food = 0
            self.food = self._spawn_food()
            if self.food is None:  # board full: won
                self.alive = False
        else:
            self.snake.pop()
            self.steps_since_food += 1

    def render_ascii(self) -> str:
        grid = [["." for _ in range(self.cols)] for _ in range(self.rows)]
        for c, r in self.snake[1:]:
            grid[r][c] = "S"
        tc, tr = self.snake[-1]
        grid[tr][tc] = "T"
        hc, hr = self.snake[0]
        grid[hr][hc] = "H"
        if self.food:
            grid[self.food[1]][self.food[0]] = "F"
        border = "#" * (self.cols + 2)
        return "\n".join([border, *("#" + "".join(row) + "#" for row in grid), border])

    def food_relative(self) -> str:
        """Food offset from the head, phrased with the same words as MOVES."""
        if self.food is None:
            return "no food left"
        dx, dy = self.food[0] - self.snake[0][0], self.food[1] - self.snake[0][1]
        parts = []
        if dy:
            parts.append(f"{abs(dy)} row(s) {'UP' if dy < 0 else 'DOWN'}")
        if dx:
            parts.append(f"{abs(dx)} column(s) {'LEFT' if dx < 0 else 'RIGHT'}")
        return "food is " + " and ".join(parts) + " from the head"

    def describe_moves(self) -> str:
        return "; ".join(
            f"{m}: not allowed (reverse)" if m == OPPOSITE[self.heading] else f"{m}: {self.outcome(m)}"
            for m in MOVES
        )


# --------------------------------------------------------------------------
# Query construction: this is everything the model sees each tick
# --------------------------------------------------------------------------


def build_state(game: Game, assist: bool = True) -> dict[str, str]:
    # Jev accepts text only, so every value is a string.
    state = {
        "rules": (
            "Classic Snake. The snake moves one cell per tick. Hitting a wall (#) or "
            "its own body ends the game. Eating food (F) grows the snake by one."
        ),
        "board_size": f"{game.cols} columns x {game.rows} rows (walls not counted)",
        "board": game.render_ascii(),
        "legend": "H = head, S = body, T = tail, F = food, . = empty, # = wall",
        "heading": f"the snake is moving {game.heading}",
        "snake_length": str(len(game.snake)),
        "head_position": f"column {game.snake[0][0]}, row {game.snake[0][1]} (0-based, row 0 at top)",
        "food_position": (
            f"column {game.food[0]}, row {game.food[1]}" if game.food else "none"
        ),
    }
    if assist:
        # Small models are weak at 2-D spatial reasoning over ASCII grids;
        # pre-computing local facts turns it into a reading task.
        state["food_relative_to_head"] = game.food_relative()
        state["next_cell"] = game.describe_moves()
    return state


def build_questions() -> dict[str, dict]:
    return {
        "move": {
            "type": "choice",
            "instructions": (
                "Which direction should the snake's head move next? Move towards the food "
                "and never into a wall or the snake's body."
            ),
            "criteria": {
                "UP": "Move the head one row up",
                "DOWN": "Move the head one row down",
                "LEFT": "Move the head one column left",
                "RIGHT": "Move the head one column right",
            },
        }
    }


def build_request(game: Game, model: str, assist: bool = True) -> dict:
    return {"model": model, "state": build_state(game, assist), "questions": build_questions()}


# --------------------------------------------------------------------------
# Model client and decision policy
# --------------------------------------------------------------------------


@dataclass
class Decision:
    move: str
    model_choice: str
    probabilities: dict[str, float]
    confidence: float
    latency_ms: float
    overridden: bool


class JevClient:
    """Minimal stdlib client for Ollama's /v1/systemone endpoint."""

    def __init__(self, model: str = DEFAULT_MODEL, host: str = DEFAULT_HOST, timeout: float = 120) -> None:
        self.model = model
        self.url = host.rstrip("/") + "/v1/systemone"
        self.timeout = timeout

    def ask(self, payload: dict) -> dict:
        req = urllib.request.Request(
            self.url,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.load(resp)


def choose_move(game: Game, answer: dict, guard: bool = True) -> tuple[str, bool]:
    """Pick a move from a Choice answer; returns (move, overridden)."""
    model_choice = answer["choice"]
    if not guard or game.is_safe(model_choice):
        return model_choice, False
    # Fall back through the model's own ranking rather than a hand-coded
    # policy, so the model still drives every decision it gets right.
    ranked = sorted(MOVES, key=lambda m: float(answer["probabilities"].get(m, 0.0)), reverse=True)
    for move in ranked:
        if game.is_safe(move):
            return move, True
    return model_choice, False  # no safe move: doomed either way


def decide(game: Game, client: JevClient, assist: bool, guard: bool) -> Decision:
    t0 = time.perf_counter()
    response = client.ask(build_request(game, client.model, assist))
    latency_ms = (time.perf_counter() - t0) * 1000
    answer = response["answers"]["move"]
    move, overridden = choose_move(game, answer, guard)
    return Decision(
        move=move,
        model_choice=answer["choice"],
        probabilities={k: float(v) for k, v in answer["probabilities"].items()},
        confidence=float(answer.get("confidence", 0.0)),
        latency_ms=latency_ms,
        overridden=overridden,
    )


def starved(game: Game) -> bool:
    # Stops a model that loops forever without eating.
    return game.steps_since_food > 2 * game.cols * game.rows


# --------------------------------------------------------------------------
# Front ends
# --------------------------------------------------------------------------


def run_headless(game: Game, client: JevClient, args: argparse.Namespace) -> None:
    overrides, latencies = 0, []
    while game.alive and not starved(game):
        d = decide(game, client, args.assist, args.guard)
        latencies.append(d.latency_ms)
        overrides += d.overridden
        game.step(d.move)
        if args.verbose:
            print(game.render_ascii())
            print(f"model={d.model_choice} played={d.move} conf={d.confidence:.2f} {d.latency_ms:.0f}ms\n")
    avg = sum(latencies) / len(latencies) if latencies else 0
    print(
        f"score={game.score} length={len(game.snake)} decisions={len(latencies)} "
        f"guard_overrides={overrides} avg_latency={avg:.0f}ms"
    )


def run_pygame(game: Game, client: JevClient, args: argparse.Namespace) -> None:
    import pygame  # imported lazily so headless mode and tests don't need it

    cell, hud = args.cell, 28
    pygame.init()
    screen = pygame.display.set_mode((game.cols * cell, game.rows * cell + hud))
    pygame.display.set_caption(f"Snake x {client.model}")
    font = pygame.font.SysFont("menlo", 14)
    clock = pygame.time.Clock()
    last: Decision | None = None

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_r and not game.alive:
                game = Game(game.cols, game.rows, rng=game.rng)

        if game.alive and not starved(game):
            # Blocking call: the game runs at the model's pace, capped by --fps.
            last = decide(game, client, args.assist, args.guard)
            game.step(last.move)
        elif game.alive:
            game.alive = False

        screen.fill((18, 18, 24))
        for c, r in game.snake[1:]:
            pygame.draw.rect(screen, (60, 170, 90), (c * cell + 1, r * cell + hud + 1, cell - 2, cell - 2))
        hc, hr = game.snake[0]
        pygame.draw.rect(screen, (120, 230, 140), (hc * cell + 1, hr * cell + hud + 1, cell - 2, cell - 2))
        if game.food:
            fc, fr = game.food
            pygame.draw.circle(screen, (230, 80, 80), (fc * cell + cell // 2, fr * cell + hud + cell // 2), cell // 3)

        status = f"score {game.score}"
        if last:
            flag = " [guard]" if last.overridden else ""
            status += f"  |  {last.move}{flag}  conf {last.confidence:.2f}  {last.latency_ms:.0f} ms"
        if not game.alive:
            status += "  |  GAME OVER - R to restart, Esc to quit"
        screen.blit(font.render(status, True, (220, 220, 220)), (8, 6))
        pygame.display.flip()
        clock.tick(args.fps)

    pygame.quit()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cols", type=int, default=20, help="N: board width in cells")
    p.add_argument("--rows", type=int, default=15, help="M: board height in cells")
    p.add_argument("--model", default=DEFAULT_MODEL, help=f"Ollama model (default {DEFAULT_MODEL})")
    p.add_argument("--host", default=DEFAULT_HOST)
    p.add_argument("--no-assist", dest="assist", action="store_false",
                   help="send only the raw board, no pre-computed hints")
    p.add_argument("--no-guard", dest="guard", action="store_false",
                   help="always play the model's top choice, even if fatal")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--verbose", action="store_true", help="headless: print every frame")
    p.add_argument("--dump", action="store_true", help="print the first request JSON and exit")
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--cell", type=int, default=28, help="pixel size of a cell")
    p.add_argument("--seed", type=int)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    game = Game(args.cols, args.rows, rng=random.Random(args.seed))
    client = JevClient(args.model, args.host)
    if args.dump:
        print(json.dumps(build_request(game, args.model, args.assist), indent=2))
        return
    try:
        (run_headless if args.headless else run_pygame)(game, client, args)
    except OSError as exc:  # URLError subclasses OSError
        sys.exit(f"Could not reach Ollama at {client.url}: {exc}. Is `ollama serve` running?")


if __name__ == "__main__":
    main()
