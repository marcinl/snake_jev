"""Snake played by a Jev-style System One decision model served by Ollama.

Each game tick sends one request to Ollama's /v1/systemone endpoint:
the board as text (state) plus a single typed Choice question (move).
With --lookahead, one request also answers the following tick's move.
The model's per-option probabilities are then used by a safety guard so a
low-confidence answer that would crash can fall back to the next-best move.

Usage:
    python snake_jev.py --cols 20 --rows 15                # pygame window
    python snake_jev.py --cols 10 --rows 8 --headless      # console only
    python snake_jev.py --dump                             # print one request and exit
    python snake_jev.py --headless --debug-timing          # log per-request timings
    python snake_jev.py --compact --lookahead              # faster: shorter state, 2 moves/request
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import random
import sys
import time
import urllib.request
from dataclasses import dataclass, field

DEFAULT_MODEL = "basal-2.0-mini:q8_0" # "nimble:9b-q4_K_M"
DEFAULT_HOST = "http://localhost:8000"
#DEFAULT_HOST = "http://localhost:11434"
#DEFAULT_HOST = "http://localhost:49935"
START_LENGTH = 2

log = logging.getLogger("snake_jev")

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

    def state_key(self) -> tuple:
        return (tuple(self.snake), self.food, self.heading, self.alive)

    def peek(self, move: str) -> Game:
        """Copy of the game after `move`; the real game and its RNG are untouched."""
        future = copy.deepcopy(self)
        future.step(move)
        return future

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


MOVE_CRITERIA = {
    "UP": "Move the head one row up",
    "DOWN": "Move the head one row down",
    "LEFT": "Move the head one column left",
    "RIGHT": "Move the head one column right",
}


def build_state(game: Game, assist: bool = True, compact: bool = False) -> dict[str, str]:
    # Jev accepts text only, so every value is a string.
    # Fields that never change go first: if the server reuses a cached prompt
    # prefix between ticks, only the dynamic tail has to be re-evaluated.
    state = {
        "rules": (
            "Classic Snake. The snake moves one cell per tick. Hitting a wall (#) or "
            "its own body ends the game. Eating food (F) grows the snake by one."
        ),
        "legend": "H = head, S = body, T = tail, F = food, . = empty, # = wall",
        "board_size": f"{game.cols} columns x {game.rows} rows (walls not counted)",
        "board": game.render_ascii(),
        "heading": f"the snake is moving {game.heading}",
        "snake_length": str(len(game.snake)),
        "head_position": f"column {game.snake[0][0]}, row {game.snake[0][1]} (0-based, row 0 at top)",
        "food_position": (
            f"column {game.food[0]}, row {game.food[1]}" if game.food else "none"
        ),
    }
    if compact:
        # Prompt evaluation dominates latency and the board is (N+2)*(M+2)
        # characters; the hints below already carry what the model acts on.
        for key in ("legend", "board", "snake_length", "head_position", "food_position"):
            del state[key]
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
            "criteria": MOVE_CRITERIA,
        }
    }


def lookahead_branches(game: Game) -> list[str]:
    # Only moves into a free cell: a fatal move ends the game, and eating
    # respawns food at a random cell the model could not have seen.
    return [m for m in MOVES if game.is_safe(m) and game.outcome(m) == "free"]


def add_lookahead(game: Game, state: dict[str, str], questions: dict[str, dict]) -> None:
    """Ask, in the same request, what the model would play after each possible move.

    One request carries one state, so the next tick's board can't be sent
    directly; instead the hints for each reachable next position are added
    and one conditional question is asked per branch.
    """
    for m in lookahead_branches(game):
        future = game.peek(m)
        state[f"if_head_moves_{m}"] = (
            f"then the snake is moving {future.heading}; {future.food_relative()}; "
            f"next cell after that: {future.describe_moves()}"
        )
        questions[f"move_after_{m}"] = {
            "type": "choice",
            "instructions": (
                f"Suppose the head first moves {m} (see if_head_moves_{m}). Which direction "
                "should it move after that? Move towards the food and never into a wall "
                "or the snake's body."
            ),
            "criteria": MOVE_CRITERIA,
        }


def build_request(
    game: Game, model: str, assist: bool = True, compact: bool = False, lookahead: bool = False
) -> dict:
    state, questions = build_state(game, assist, compact), build_questions()
    if lookahead:
        add_lookahead(game, state, questions)
    return {"model": model, "state": state, "questions": questions}


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
    from_lookahead: bool = False  # answered by the previous request, no new call
    loop_broken: bool = False  # move replaced by a less-visited one to leave a cycle


@dataclass
class RequestTiming:
    payload_bytes: int
    questions: int
    encode_ms: float
    ttfb_ms: float  # until response headers: dominated by Ollama's prompt evaluation
    read_ms: float
    total_ms: float
    server: dict  # whatever usage / *_duration fields the response carries


def server_timing_fields(response: dict) -> dict:
    """Usage and Ollama-style timing fields, if the response includes any."""
    fields = {}
    if isinstance(response.get("usage"), dict):
        fields.update(response["usage"])
    for key, value in response.items():
        # Ollama's native API reports durations in nanoseconds; assumed the same here.
        if key.endswith("_duration") and isinstance(value, (int, float)):
            fields[key.removesuffix("_duration") + "_ms"] = round(value / 1e6, 1)
        elif key.endswith("_count") and isinstance(value, int):
            fields[key] = value
    return fields


class JevClient:
    """Minimal stdlib client for Ollama's /v1/systemone endpoint."""

    def __init__(
        self, model: str = DEFAULT_MODEL, host: str = DEFAULT_HOST, timeout: float = 120,
        debug_timing: bool = False,
    ) -> None:
        self.model = model
        self.url = host.rstrip("/") + "/v1/systemone"
        self.timeout = timeout
        self.debug_timing = debug_timing
        self.timings: list[RequestTiming] = []

    def ask(self, payload: dict) -> dict:
        t0 = time.perf_counter()
        data = json.dumps(payload).encode()
        req = urllib.request.Request(self.url, data=data, headers={"Content-Type": "application/json"})
        t1 = time.perf_counter()
        # urlopen returns once the status line and headers arrive, so t2 - t1
        # is time-to-first-byte; reading the small JSON body is separate.
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            t2 = time.perf_counter()
            body = resp.read()
        t3 = time.perf_counter()
        response = json.loads(body)
        if self.debug_timing:
            timing = RequestTiming(
                payload_bytes=len(data),
                questions=len(payload.get("questions", {})),
                encode_ms=(t1 - t0) * 1000,
                ttfb_ms=(t2 - t1) * 1000,
                read_ms=(t3 - t2) * 1000,
                total_ms=(t3 - t0) * 1000,
                server=server_timing_fields(response),
            )
            self.timings.append(timing)
            log.info(
                "request #%d: payload=%dB questions=%d encode=%.1fms ttfb=%.1fms read=%.1fms "
                "total=%.1fms server=%s",
                len(self.timings), timing.payload_bytes, timing.questions, timing.encode_ms,
                timing.ttfb_ms, timing.read_ms, timing.total_ms, timing.server or "-",
            )
        return response


def timing_summary(timings: list[RequestTiming]) -> str:
    if not timings:
        return "summary: no requests"

    def stats(values: list[float]) -> str:
        ordered = sorted(values)

        def pct(p: float) -> float:  # nearest-rank percentile
            return ordered[max(0, math.ceil(p * len(ordered)) - 1)]

        mean = sum(ordered) / len(ordered)
        return f"mean={mean:.0f} p50={pct(0.5):.0f} p95={pct(0.95):.0f} max={ordered[-1]:.0f}ms"

    avg_bytes = sum(t.payload_bytes for t in timings) / len(timings)
    # The first request often includes loading the model from disk.
    return (
        f"summary: {len(timings)} requests, avg payload {avg_bytes:.0f}B, "
        f"first {timings[0].total_ms:.0f}ms\n"
        f"  total {stats([t.total_ms for t in timings])}\n"
        f"  ttfb  {stats([t.ttfb_ms for t in timings])}"
    )


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


def break_loop(game: Game, answer: dict, move: str, seen: dict[tuple, int]) -> str | None:
    """If `move` re-enters a position in `seen` (visit counts), the safe move into the
    least-visited position, ties going to the model's ranking; None to keep `move`."""
    # The model answers a position the same way every time, so re-entering
    # one replays the whole cycle forever. Least-visited rather than unvisited:
    # when every exit has been seen, the snake still drifts out of the loop.
    if seen.get(game.peek(move).state_key(), 0) == 0:
        return None
    ranked = sorted(MOVES, key=lambda m: float(answer["probabilities"].get(m, 0.0)), reverse=True)
    exits = [m for m in ranked if game.is_safe(m)]
    if not exits:
        return None
    best = min(exits, key=lambda m: seen.get(game.peek(m).state_key(), 0))  # min keeps the first of ties
    return None if game.effective(best) == game.effective(move) else best


class Player:
    """Turns game states into moves, reusing a lookahead answer when it applies."""

    def __init__(
        self, client: JevClient, assist: bool = True, guard: bool = True,
        compact: bool = False, lookahead: bool = False, loop_break: bool = True,
    ) -> None:
        self.client = client
        self.assist = assist
        self.guard = guard
        self.compact = compact
        self.lookahead = lookahead
        self.loop_break = loop_break
        # (state key the answer was computed for, Choice answer)
        self._followup: tuple[tuple, dict] | None = None
        self._seen: dict[tuple, int] = {}  # visits per position since the last food (or game start)

    def decide(self, game: Game) -> Decision:
        if game.steps_since_food == 0:
            self._seen.clear()
        key = game.state_key()
        self._seen[key] = self._seen.get(key, 0) + 1
        seen = self._seen if self.loop_break else None
        followup, self._followup = self._followup, None
        if followup and followup[0] == game.state_key():
            # The guard still checks it against the real board.
            return make_decision(game, followup[1], self.guard, latency_ms=0.0, from_lookahead=True, seen=seen)

        t0 = time.perf_counter()
        response = self.client.ask(
            build_request(game, self.client.model, self.assist, self.compact, self.lookahead)
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        answers = response["answers"]
        decision = make_decision(game, answers["move"], self.guard, latency_ms, seen=seen)
        # Keyed on the move actually played, which may be a guard or loop override.
        played = game.effective(decision.move)
        if f"move_after_{played}" in answers and game.outcome(decision.move) == "free":
            self._followup = (game.peek(decision.move).state_key(), answers[f"move_after_{played}"])
        return decision


def make_decision(
    game: Game, answer: dict, guard: bool, latency_ms: float, from_lookahead: bool = False,
    seen: dict[tuple, int] | None = None,
) -> Decision:
    move, overridden = choose_move(game, answer, guard)
    escape = break_loop(game, answer, move, seen) if seen is not None else None
    return Decision(
        move=escape or move,
        model_choice=answer["choice"],
        probabilities={k: float(v) for k, v in answer["probabilities"].items()},
        confidence=float(answer.get("confidence", 0.0)),
        latency_ms=latency_ms,
        overridden=overridden,
        from_lookahead=from_lookahead,
        loop_broken=escape is not None,
    )


def starved(game: Game) -> bool:
    # Stops a model that loops forever without eating.
    return game.steps_since_food > 2 * game.cols * game.rows


# --------------------------------------------------------------------------
# Front ends
# --------------------------------------------------------------------------


def run_headless(game: Game, player: Player, args: argparse.Namespace) -> None:
    overrides, loop_breaks, decisions, latencies = 0, 0, 0, []
    while game.alive and not starved(game):
        d = player.decide(game)
        decisions += 1
        if not d.from_lookahead:  # lookahead moves cost no request; keep them out of the average
            latencies.append(d.latency_ms)
        overrides += d.overridden
        loop_breaks += d.loop_broken
        game.step(d.move)
        if args.verbose:
            print(game.render_ascii())
            took = "lookahead" if d.from_lookahead else f"{d.latency_ms:.0f}ms"
            print(f"model={d.model_choice} played={d.move} conf={d.confidence:.2f} {took}\n")
    avg = sum(latencies) / len(latencies) if latencies else 0
    print(
        f"score={game.score} length={len(game.snake)} decisions={decisions} requests={len(latencies)} "
        f"guard_overrides={overrides} loop_breaks={loop_breaks} avg_latency={avg:.0f}ms"
    )


def run_pygame(game: Game, player: Player, args: argparse.Namespace) -> None:
    import pygame  # imported lazily so headless mode and tests don't need it

    cell, hud = args.cell, 28
    pygame.init()
    screen = pygame.display.set_mode((game.cols * cell, game.rows * cell + hud))
    pygame.display.set_caption(f"Snake x {player.client.model}")
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
            last = player.decide(game)
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
            flag = " [loop]" if last.loop_broken else " [guard]" if last.overridden else ""
            took = "lookahead" if last.from_lookahead else f"{last.latency_ms:.0f} ms"
            status += f"  |  {last.move}{flag}  conf {last.confidence:.2f}  {took}"
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
    p.add_argument("--no-loop-break", dest="loop_break", action="store_false",
                   help="let the snake re-enter positions visited since the last food")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--verbose", action="store_true", help="headless: print every frame")
    p.add_argument("--dump", action="store_true", help="print the first request JSON and exit")
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--cell", type=int, default=28, help="pixel size of a cell")
    p.add_argument("--seed", type=int)
    p.add_argument("--debug-timing", action="store_true",
                   help="log timing for every Ollama request to stderr, and a summary at exit")
    p.add_argument("--compact", action="store_true",
                   help="drop the ASCII board and fields the hints already cover (shorter prompt)")
    p.add_argument("--lookahead", action="store_true",
                   help="also ask for the follow-up move of each branch: up to 2 moves per request")
    args = p.parse_args(argv)
    # Both rely on the pre-computed hints to describe the position.
    for flag in ("compact", "lookahead"):
        if getattr(args, flag) and not args.assist:
            p.error(f"--{flag} needs the hints; it can't be combined with --no-assist")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    game = Game(args.cols, args.rows, rng=random.Random(args.seed))
    if args.debug_timing:
        logging.basicConfig(level=logging.INFO, format="[timing] %(message)s", stream=sys.stderr)
    client = JevClient(args.model, args.host, debug_timing=args.debug_timing)
    if args.dump:
        print(json.dumps(build_request(game, args.model, args.assist, args.compact, args.lookahead), indent=2))
        return
    player = Player(client, args.assist, args.guard, args.compact, args.lookahead, args.loop_break)
    try:
        (run_headless if args.headless else run_pygame)(game, player, args)
    except OSError as exc:  # URLError subclasses OSError
        sys.exit(f"Could not reach Ollama at {client.url}: {exc}. Is `ollama serve` running?")
    finally:
        if args.debug_timing:
            log.info(timing_summary(client.timings))


if __name__ == "__main__":
    main()
