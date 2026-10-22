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
import collections
import copy
import json
import logging
import math
import random
import sys
import time
import urllib.request
from dataclasses import dataclass, field

#DEFAULT_MODEL = "nimble:9b-q4_K_M" #"tev1:0.8b"
DEFAULT_MODEL = "hf.co/ggml-org/vllm-d3-edge-GGUF:BF16"
DEFAULT_HOST = "http://localhost:11434"
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
    # True when the head is back on a (cell, heading) it already had since the last food.
    route_repeated: bool = field(init=False, default=False)
    # Stays True from the first repeat until the next food.
    looped: bool = field(init=False, default=False)
    _route: set[tuple[Point, str]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.cols < START_LENGTH + 1 or self.rows < 1:
            raise ValueError(f"board {self.cols}x{self.rows} too small for a length-{START_LENGTH} snake")
        head = (self.cols // 2, self.rows // 2)
        self.heading = "RIGHT"
        self.snake = [(head[0] - i, head[1]) for i in range(START_LENGTH)]
        self.food = self._spawn_food()
        self._route = {(head, self.heading)}

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
        position = (nxt, self.heading)
        if result == "food":
            self.score += 1
            self.steps_since_food = 0
            self.food = self._spawn_food()
            if self.food is None:  # board full: won
                self.alive = False
            self._route = {position}
            self.route_repeated = False
            self.looped = False
        else:
            self.snake.pop()
            self.steps_since_food += 1
            self.route_repeated = position in self._route
            self.looped = self.looped or self.route_repeated
            self._route.add(position)

    def state_key(self) -> tuple:
        return (tuple(self.snake), self.food, self.heading, self.alive)

    def peek(self, move: str) -> Game:
        """Copy of the game after `move`; the real game and its RNG are untouched."""
        future = copy.deepcopy(self)
        future.step(move)
        return future

    def _flood(self, start: Point, blocked: set[Point]) -> set[Point]:
        seen, frontier = {start}, [start]
        while frontier:
            c, r = frontier.pop()
            for dx, dy in HEADINGS.values():
                p = (c + dx, r + dy)
                if p not in seen and p not in blocked and self.in_bounds(p):
                    seen.add(p)
                    frontier.append(p)
        return seen

    def _head_region(self) -> set[Point]:
        # Every body cell but the tail counts as a wall. In reality cells free
        # up as the tail moves, so this errs on the side of "trapped".
        return self._flood(self.snake[0], set(self.snake[1:-1]))

    def tail_reachable(self, move: str) -> bool:
        """Whether, after `move`, the head can still reach the tail.

        While it can, following the tail is always a way out, so a move
        that keeps it reachable can't box the snake in. A move that is
        merely safe (`is_safe`) can still enter a dead-end pocket.
        """
        future = self.peek(move)
        if not future.alive:
            return future.food is None  # filled the board: won, not trapped
        return future.snake[-1] in future._head_region()

    def free_space(self, move: str) -> int:
        """Cells the head could still reach after `move` (0 if it dies)."""
        future = self.peek(move)
        if not future.alive:
            return 0
        return len(future._head_region()) - 1

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

    def move_hint(self, move: str) -> str:
        """What `move` does this tick, including whether it gets nearer the food."""
        if move == OPPOSITE[self.heading]:
            return "not allowed (reverse)"
        result = self.outcome(move)
        if result in ("wall", "body"):
            return f"{result}, game over"
        if result == "food":
            return "food, eats it"
        if self.food is None:
            return "free"
        # Every move changes the Manhattan distance by exactly 1, so there are no ties.
        before = abs(self.food[0] - self.snake[0][0]) + abs(self.food[1] - self.snake[0][1])
        nxt = self.next_head(move)
        after = abs(self.food[0] - nxt[0]) + abs(self.food[1] - nxt[1])
        return "free, towards the food" if after < before else "free, away from the food"

    def describe_moves(self) -> str:
        return "; ".join(f"{m}: {self.move_hint(m)}" for m in MOVES)


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


def move_criteria(game: Game, assist: bool = True) -> dict[str, str]:
    if not assist:
        return MOVE_CRITERIA
    # The model scores each option mostly by its criterion text: with static
    # criteria it answered UP in 58 of 60 positions whatever the food hint said.
    # Repeating the per-move hint here took food-ward choices from 23 to 37 of 60
    # (24 to 50 with --compact).
    return {m: f"{MOVE_CRITERIA[m]}: {game.move_hint(m)}" for m in MOVES}


def build_questions(game: Game, assist: bool = True) -> dict[str, dict]:
    return {
        "move": {
            "type": "choice",
            "instructions": (
                "Which direction should the snake's head move next? Move towards the food "
                "and never into a wall or the snake's body."
            ),
            "criteria": move_criteria(game, assist),
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
            "criteria": move_criteria(future),
        }


def build_request(
    game: Game, model: str, assist: bool = True, compact: bool = False, lookahead: bool = False
) -> dict:
    state, questions = build_state(game, assist, compact), build_questions(game, assist)
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


def choose_move(
    game: Game, answer: dict, guard: bool = True, tail_guard: bool = False
) -> tuple[str, bool]:
    """Pick a move from a Choice answer; returns (move, overridden).

    With `tail_guard`, a move must also keep the tail reachable, not just
    avoid crashing this tick. Once the snake is repeating its route, search
    picks the moves until the next food.
    """
    model_choice = answer["choice"]
    if not guard:
        return model_choice, False
    if game.looped:
        # The head came back to a (cell, heading) it already had since the last
        # food. The guard's fallback is deterministic, so if the model keeps
        # answering the same way the snake goes round this cycle until it
        # starves. Handing back to the model as soon as the head leaves the old
        # route isn't enough: the same bias builds a new loop. So search plays
        # until the snake has eaten.
        move = search_move(game)
        return move, move != model_choice

    def acceptable(move: str) -> bool:
        return game.is_safe(move) and (not tail_guard or game.tail_reachable(move))

    if acceptable(model_choice):
        return model_choice, False
    # Fall back through the model's own ranking rather than a hand-coded
    # policy, so the model still drives every decision it gets right.
    ranked = sorted(MOVES, key=lambda m: float(answer["probabilities"].get(m, 0.0)), reverse=True)
    for move in ranked:
        if acceptable(move):
            return move, True
    safe = [m for m in ranked if game.is_safe(m)]
    if tail_guard and safe:
        # Every safe move cuts off the tail: take the one with the most room,
        # which gives the body the longest time to unwind. max() keeps the
        # model's ranking on ties.
        move = max(safe, key=game.free_space)
        return move, move != model_choice
    return model_choice, False  # no safe move: doomed either way


class Player:
    """Turns game states into moves, reusing a lookahead answer when it applies."""

    def __init__(
        self, client: JevClient, assist: bool = True, guard: bool = True,
        compact: bool = False, lookahead: bool = False, tail_guard: bool = False,
    ) -> None:
        self.client = client
        self.assist = assist
        self.guard = guard
        self.compact = compact
        self.lookahead = lookahead
        self.tail_guard = tail_guard
        # (state key the answer was computed for, Choice answer)
        self._followup: tuple[tuple, dict] | None = None

    @property
    def label(self) -> str:
        return self.client.model

    def decide(self, game: Game) -> Decision:
        followup, self._followup = self._followup, None
        if followup and followup[0] == game.state_key():
            # The guard still checks it against the real board.
            return make_decision(
                game, followup[1], self.guard, latency_ms=0.0, from_lookahead=True,
                tail_guard=self.tail_guard,
            )

        t0 = time.perf_counter()
        response = self.client.ask(
            build_request(game, self.client.model, self.assist, self.compact, self.lookahead)
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        answers = response["answers"]
        decision = make_decision(game, answers["move"], self.guard, latency_ms, tail_guard=self.tail_guard)
        # Keyed on the move actually played, which may be a guard override.
        played = game.effective(decision.move)
        if f"move_after_{played}" in answers and game.outcome(decision.move) == "free":
            self._followup = (game.peek(decision.move).state_key(), answers[f"move_after_{played}"])
        return decision


def make_decision(
    game: Game, answer: dict, guard: bool, latency_ms: float, from_lookahead: bool = False,
    tail_guard: bool = False,
) -> Decision:
    move, overridden = choose_move(game, answer, guard, tail_guard)
    return Decision(
        move=move,
        model_choice=answer["choice"],
        probabilities={k: float(v) for k, v in answer["probabilities"].items()},
        confidence=float(answer.get("confidence", 0.0)),
        latency_ms=latency_ms,
        overridden=overridden,
        from_lookahead=from_lookahead,
    )


# --------------------------------------------------------------------------
# Search fallback: no model, plays when nothing else has a safe move
# --------------------------------------------------------------------------


def bfs_path(game: Game, goal: Point) -> list[str] | None:
    """Shortest list of moves from the head to `goal`, treating the body as fixed walls."""
    head = game.snake[0]
    # The neck is always blocked: for a length-2 snake it is also the tail,
    # but moving onto it is a reverse, which the game ignores.
    blocked = (set(game.snake[1:-1]) | {game.snake[1]}) - {goal}
    prev: dict[Point, tuple[Point, str]] = {}
    seen, queue = {head}, collections.deque([head])
    while queue:
        cell = queue.popleft()
        if cell == goal:
            moves = []
            while cell != head:
                cell, move = prev[cell]
                moves.append(move)
            return moves[::-1]
        for move, (dx, dy) in HEADINGS.items():
            nxt = (cell[0] + dx, cell[1] + dy)
            if nxt not in seen and nxt not in blocked and game.in_bounds(nxt):
                seen.add(nxt)
                prev[nxt] = (cell, move)
                queue.append(nxt)
    return None


def safe_food_path(game: Game) -> list[str] | None:
    """Shortest path to the food, if the tail is still reachable once the snake has eaten."""
    if game.food is None:
        return None
    path = bfs_path(game, game.food)
    if path is None:
        return None
    future = copy.deepcopy(game)
    for move in path:
        future.step(move)
    if not future.alive:
        return path if future.food is None else None  # filled the board: won
    return path if future.snake[-1] in future._head_region() else None


def stall_move(game: Game) -> str | None:
    """A move that keeps the tail reachable while staying as far from it as possible.

    Used when the food can't be reached safely: following the tail the long
    way round buys time for the body to move and a path to open.
    """
    options = [m for m in MOVES if game.is_safe(m) and game.tail_reachable(m)]
    if not options:
        return None

    def distance_to_tail(move: str) -> int:
        future = game.peek(move)
        path = bfs_path(future, future.snake[-1])
        return len(path) if path else 0

    return max(options, key=distance_to_tail)


def search_move(game: Game) -> str:
    path = safe_food_path(game)
    if path:
        return path[0]
    move = stall_move(game)
    if move:
        return move
    # Nothing keeps the tail reachable: same last resort as the tail guard.
    safe = [m for m in MOVES if game.is_safe(m)]
    return max(safe, key=game.free_space) if safe else game.heading


class SearchPlayer:
    """Plays `search_move` every tick without the model: a baseline for comparisons."""

    label = "search"

    def decide(self, game: Game) -> Decision:
        t0 = time.perf_counter()
        move = search_move(game)
        return Decision(
            move=move, model_choice=move, probabilities={}, confidence=1.0,
            latency_ms=(time.perf_counter() - t0) * 1000, overridden=False,
        )


@dataclass
class ModelStats:
    """How much of a game the model actually played, printed when the game ends."""

    decisions: int = 0
    unchanged: int = 0  # the model's own choice was played; the guard didn't step in
    agrees_with_search: int = 0

    def record(self, game: Game, decision: Decision) -> None:
        # Call before game.step: search is asked about the same position the model saw.
        self.decisions += 1
        self.unchanged += not decision.overridden
        self.agrees_with_search += decision.model_choice == search_move(game)

    def summary(self, player: Player) -> str:
        def share(n: int) -> str:
            pct = 100 * n / self.decisions if self.decisions else 0
            return f"{n}/{self.decisions} ({pct:.0f}%)"

        # The settings line lets runs with different prompts be compared side by side.
        prompt = ["hints" if player.assist else "no hints", "compact" if player.compact else "full board"]
        if player.lookahead:
            prompt.append("lookahead")
        guard = "off" if not player.guard else "tail" if player.tail_guard else "cell"
        return (
            f"model {player.label} | prompt: {', '.join(prompt)} | guard: {guard}\n"
            f"  model's move played unchanged: {share(self.unchanged)}\n"
            f"  model's choice matches search: {share(self.agrees_with_search)}"
        )


def starved(game: Game) -> bool:
    # Stops a model that loops forever without eating.
    return game.steps_since_food > 2 * game.cols * game.rows


# --------------------------------------------------------------------------
# Front ends
# --------------------------------------------------------------------------


def run_headless(game: Game, player: Player | SearchPlayer, args: argparse.Namespace) -> None:
    overrides, decisions, latencies = 0, 0, []
    stats = ModelStats() if isinstance(player, Player) else None
    while game.alive and not starved(game):
        d = player.decide(game)
        if stats:
            stats.record(game, d)
        decisions += 1
        if not d.from_lookahead:  # lookahead moves cost no request; keep them out of the average
            latencies.append(d.latency_ms)
        overrides += d.overridden
        game.step(d.move)
        if args.verbose:
            print(game.render_ascii())
            took = "lookahead" if d.from_lookahead else f"{d.latency_ms:.0f}ms"
            print(f"model={d.model_choice} played={d.move} conf={d.confidence:.2f} {took}\n")
    avg = sum(latencies) / len(latencies) if latencies else 0
    print(
        f"score={game.score} length={len(game.snake)} decisions={decisions} requests={len(latencies)} "
        f"guard_overrides={overrides} avg_latency={avg:.0f}ms"
    )
    if stats:
        print(stats.summary(player))


def run_pygame(game: Game, player: Player | SearchPlayer, args: argparse.Namespace) -> None:
    import pygame  # imported lazily so headless mode and tests don't need it

    cell, hud = args.cell, 28
    pygame.init()
    screen = pygame.display.set_mode((game.cols * cell, game.rows * cell + hud))
    pygame.display.set_caption(f"Snake x {player.label}")
    font = pygame.font.SysFont("menlo", 14)
    clock = pygame.time.Clock()
    last: Decision | None = None
    track = isinstance(player, Player)
    stats, reported = (ModelStats() if track else None), False

    def report() -> None:
        nonlocal reported
        if stats and stats.decisions and not reported:
            print(f"game over: score {game.score}\n{stats.summary(player)}")
        reported = True

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_r and not game.alive:
                game = Game(game.cols, game.rows, rng=game.rng)
                stats, reported = (ModelStats() if track else None), False

        if game.alive and not starved(game):
            # Blocking call: the game runs at the model's pace, capped by --fps.
            last = player.decide(game)
            if stats:
                stats.record(game, last)
            game.step(last.move)
        elif game.alive:
            game.alive = False
        if not game.alive:
            report()

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
            took = "lookahead" if last.from_lookahead else f"{last.latency_ms:.0f} ms"
            status += f"  |  {last.move}{flag}  conf {last.confidence:.2f}  {took}"
        if not game.alive:
            status += "  |  GAME OVER - R to restart, Esc to quit"
        screen.blit(font.render(status, True, (220, 220, 220)), (8, 6))
        pygame.display.flip()
        clock.tick(args.fps)

    report()  # quitting mid-game still reports the moves played so far
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
    p.add_argument("--tail-guard", action="store_true",
                   help="guard also rejects moves after which the head can't reach the tail")
    p.add_argument("--search-only", action="store_true",
                   help="play with BFS search alone, no model (a baseline; Ollama not needed)")
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
    if args.tail_guard and not args.guard:
        p.error("--tail-guard can't be combined with --no-guard")
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
    if args.search_only:
        player: Player | SearchPlayer = SearchPlayer()
    else:
        player = Player(client, args.assist, args.guard, args.compact, args.lookahead, args.tail_guard)
    try:
        (run_headless if args.headless else run_pygame)(game, player, args)
    except OSError as exc:  # URLError subclasses OSError
        sys.exit(f"Could not reach Ollama at {client.url}: {exc}. Is `ollama serve` running?")
    finally:
        if args.debug_timing:
            log.info(timing_summary(client.timings))


if __name__ == "__main__":
    main()
