# Snake × Jev

Snake played by a Jev-style System One decision model ([TypeSafe](https://typesafe.ai/)) served locally by [Ollama](https://ollama.com/blog/ollama-now-supports-jev-style-decision-models) (0.35+).

On every game tick the game sends the board to Ollama's `/v1/systemone` endpoint as text. It asks one typed question, which direction to move, and plays the answer.

**Status:** the integration works end to end. The default `tev1:0.8b` model plays poorly, as described under [Findings](#findings-with-tev108b). Try `--model nimble` (9B) for stronger play.

## Quick start

Prerequisites: Python 3.10+, Ollama 0.35+ running locally, and a Jev-style model:

```bash
ollama pull tev1:0.8b
```

```bash
pip install -r requirements.txt
```

This installs [pygame-ce](https://pyga.me/), the community fork of pygame; the code still says `import pygame`. Don't install `pygame` and `pygame-ce` together, because both provide the `pygame` package and overwrite each other. If you see `NotImplementedError: font module not available`, run `pip uninstall pygame` and then `pip install pygame-ce`. (pygame 2.6.1 has no Python 3.14 wheel, and building it from source leaves out fonts unless SDL_ttf is installed.)

```bash
python3 snake_jev.py --cols 20 --rows 15
```

| Flag | Default | Purpose |
|---|---|---|
| `--cols` / `--rows` | 20 / 15 | Board size N × M (walls not counted) |
| `--model` | `tev1:0.8b` | Any Jev-style Ollama model, e.g. `nimble`, `tev1` |
| `--host` | `http://localhost:11434` | Ollama server |
| `--no-assist` | off | Send only the raw board, without the hints the game works out |
| `--no-guard` | off | Always play the model's top choice, even if it crashes |
| `--tail-guard` | off | The guard also rejects moves after which the head can no longer reach the tail (see step 3 below) |
| `--search-only` | off | Play with BFS search alone, without the model or Ollama: a baseline to compare the model against |
| `--headless` | off | Console only, no PyGame window; prints a summary line |
| `--verbose` | off | Headless: print every frame and decision |
| `--dump` | off | Print the request for the first tick and exit |
| `--fps` | 10 | Upper limit on ticks per second (the model's speed is usually the real limit) |
| `--cell` | 28 | Size of one cell in pixels |
| `--seed` | random | Fixes food placement so runs can be repeated |
| `--debug-timing` | off | Log the timing of every Ollama request to stderr, plus a summary at exit |
| `--compact` | off | Leave out the ASCII board and the fields the hints already give, for a shorter prompt |
| `--lookahead` | off | Get up to 2 moves from each request (see [Speed](#speed)) |

In the window, **R** restarts after game over and **Esc** quits.

Run the tests (no Ollama or pygame needed):

```bash
python3 -m unittest -v
```

## How it works

1. **Game logic (no model involved).** An N columns × M rows grid. The snake starts with length 2, with its head at `(N//2, M//2)`, its tail one cell to the left, and heading RIGHT. Food spawns on a random free cell. Reverse moves are ignored, as in classic Snake.
2. **Each game tick.** Build a text description of the board, send one `POST /v1/systemone` request, and get back a typed `choice` answer.
3. **Safety check ("guard").** The answer includes a probability for every move. If the model's top move would crash, the game plays its most likely safe move instead. `--no-guard` turns this off.
   **Loop breaker.** If the head comes back to a cell it already visited since the last food, with the same heading, the snake is going round a cycle. The fallback above is deterministic, so a model that keeps giving the same answer would stay in that cycle until it starves. Search (see below) then plays every move until the snake eats, and these moves count as guard overrides. Search has to keep steering until the food: if the model takes over again as soon as the head leaves the old route, the same bias builds a new loop.
   With `--tail-guard`, a move must also leave a path from the head to the tail. A move can be safe for one tick and still enter a dead-end pocket. While the tail is reachable, following it is always a way out. If every safe move cuts the tail off, the guard plays the one that leaves the most free space.
4. **Display.** A PyGame window (or `--headless`). The game runs at the model's pace, up to the `--fps` limit. The status bar shows the score, the move played, the model's confidence and the time each decision took.

A game also ends if the snake goes 2·N·M moves without eating. With the guard on, the loop breaker normally gets there first; the limit matters for `--no-guard`.

**End-of-game report.** After every model game, headless or in the window, the console shows how much of the game the model actually played:

```
model hf.co/ggml-org/vllm-d3-edge-GGUF:BF16 | prompt: hints, compact | guard: tail
  model's move played unchanged: 130/299 (43%)
  model's choice matches search: 49/299 (16%)
```

- *Played unchanged*: the model's top choice was played as-is; the rest of the time the guard replaced it.
- *Matches search*: the model chose the move search would have played in the same position. This is a stricter test, since a move can be safe and still worse than search's.
- The first line records the prompt and guard settings, so runs with different prompts can be compared side by side.

**Search baseline (`--search-only`).** No model is involved. Each tick it takes the shortest path to the food, but only if the head could still reach the tail after eating. Otherwise it follows its tail the long way round until a safe path opens. Over 5 seeds it filled the board (a win) 3 times on 10×8 and once on 20×15; the other games ended 2–8 cells short of a full board, on the starvation limit. None crashed. It is meant as the last-resort layer of a hybrid controller, and as the bar the model has to beat.

## What gets sent to the model each tick

Jev only accepts text, so every value in `state` is a string. This is the first tick on an 8×6 board (`--dump --cols 8 --rows 6 --seed 1`):

```json
{
  "model": "tev1:0.8b",
  "state": {
    "rules": "Classic Snake. The snake moves one cell per tick. Hitting a wall (#) or its own body ends the game. Eating food (F) grows the snake by one.",
    "legend": "H = head, S = body, T = tail, F = food, . = empty, # = wall",
    "board_size": "8 columns x 6 rows (walls not counted)",
    "board": "##########\n#........#\n#F.......#\n#........#\n#...TH...#\n#........#\n#........#\n##########",
    "heading": "the snake is moving RIGHT",
    "snake_length": "2",
    "head_position": "column 4, row 3 (0-based, row 0 at top)",
    "food_position": "column 0, row 1",
    "food_relative_to_head": "food is 2 row(s) UP and 4 column(s) LEFT from the head",
    "next_cell": "UP: free, towards the food; DOWN: free, away from the food; LEFT: not allowed (reverse); RIGHT: free, away from the food"
  },
  "questions": {
    "move": {
      "type": "choice",
      "instructions": "Which direction should the snake's head move next? Move towards the food and never into a wall or the snake's body.",
      "criteria": {
        "UP": "Move the head one row up: free, towards the food",
        "DOWN": "Move the head one row down: free, away from the food",
        "LEFT": "Move the head one column left: not allowed (reverse)",
        "RIGHT": "Move the head one column right: free, away from the food"
      }
    }
  }
}
```

The response looks like this:

```json
{
  "model": "tev1:0.8b",
  "answers": {
    "move": {
      "type": "choice",
      "choice": "UP",
      "probabilities": {"UP": 0.61, "DOWN": 0.12, "LEFT": 0.18, "RIGHT": 0.09},
      "confidence": 0.21
    }
  },
  "usage": {"input_tokens": 512, "output_tokens": 3}
}
```

(The numbers above are illustrative; the format is the real one.)

- **How it scales with N × M:** the `board` field is (N+2)·(M+2) characters, so the input grows with board area. Everything else stays the same size.
- **Hints:** `food_relative_to_head` and `next_cell` are worked out by the game, not the model. Small models are weak at reading positions off an ASCII grid; the hints turn the task into reading a sentence. `--no-assist` removes them.
- **Criteria carry the hints too.** Each option's criterion repeats what that move does (`towards the food`, `away from the food`, `wall, game over`, …). The model scores options mostly by their criterion text. With the plain criteria ("Move the head one row up"), `hf.co/ggml-org/vllm-d3-edge-GGUF:BF16` answered UP in 58 of 60 random positions, whatever the hints said. `--no-assist` keeps the plain criteria.

## Speed

The model's output is only a few tokens, so almost all the time per request goes on reading the prompt. These options let you measure that and cut it down:

- **`--debug-timing`** logs one line per request:
  - payload size and number of questions
  - time to encode the request
  - time to first byte (`ttfb`), which is mostly Ollama reading the prompt
  - time to read the body, and the total
  - any `usage` or `*_duration` fields in the response (durations are assumed to be nanoseconds, as in Ollama's native API)

  At exit it prints the mean, p50, p95 and max. If the first request is much slower than the rest, the model was loading from disk.
- **Order of the state fields.** The fields that never change (`rules`, `legend`, `board_size`) come first. If the server reuses a cached prompt prefix, only the changing part is read again on each tick. This depends on `/v1/systemone` keeping the key order, which hasn't been checked yet.
- **`--compact`** leaves out `board`, `legend`, `snake_length`, `head_position` and `food_position`. The hints already cover them. This changes how big the state is as the board grows: from about N·M characters to a fixed size (on 20×15, 1535 B → 905 B). The model might play worse without the board, so compare scores with and without it.
- **`--lookahead`** gets up to 2 moves from each request:
  - A request has only one `state`, so the next tick's board can't be sent ahead of time.
  - Instead, the request adds the hints for each position the snake can move into next (`if_head_moves_UP`, …) and asks one more Choice question for each (`move_after_UP`, …), so up to 4 questions in total.
  - The answer for the move actually played (including a guard override) is used on the next tick without a new request, and the guard checks it again first.
  - It isn't used if the snake ate, because the new food appears somewhere the model didn't know about.

  Each request is larger, so check with `--debug-timing` whether it saves time overall. Moves taken from the lookahead show as `lookahead` instead of a time in milliseconds.
- **Keep the model loaded.** Ollama unloads a model after 5 minutes without use. On macOS, run `launchctl setenv OLLAMA_KEEP_ALIVE -1` and restart the Ollama app.

`--compact` and `--lookahead` both rely on the hints, so neither can be combined with `--no-assist`.

## Findings with `tev1:0.8b`

Measured on Ollama 0.35.1 with a 10×8 board:

| Setup | Result |
|---|---|
| Guard on, hints on | 0–3 food per game over 5 seeds; mostly loops until the 2·N·M limit; 60–105 ms per decision |
| Guard off | Crashes within 5–7 moves |
| Hints off | Ate nothing in 3 games |
| Moves as straight/left/right (first version) | Stuck in a 2×2 loop, always choosing "right" |
| Moves as UP/DOWN/LEFT/RIGHT (current) | Moved towards the food in 22 of 30 test positions, against 17 of 30 |
| Four yes/no ("Noul") questions per tick instead of one Choice | Always safe (30 of 30), but went towards the food less often (13 vs 16 of 30); not adopted |

UP/DOWN/LEFT/RIGHT likely works better because the answer options then use the same words as the hints. The main limit now is the model's size, not how the question is asked.

## Findings with `hf.co/ggml-org/vllm-d3-edge-GGUF:BF16`

Measured on a 20×15 board, guard on, seeds 1–3:

| Setup | Before (plain criteria, no loop breaker) | After |
|---|---|---|
| Full board | Score 0 in all 3 games; starved at 601 moves; model's move played 16–42% | Score 10 / 23 / 24; model's move played 93–94% |
| `--compact` | Score 0 in all 3 games; starved at 601 moves; model's move played 33–47% | Score 30 / 9 / 20; model's move played 98% |

Before, the model answered UP almost every tick. The guard's fallback then drove the snake round a fixed cycle between the top-right and bottom-right corners. On 60 random positions, putting the per-move hints into the criteria raised food-ward choices from 23 to 37 (full board) and from 24 to 50 (`--compact`). Without the board, the model picks the food-ward move more often in single positions. Over 3 games the scores are mixed, so more seeds are needed before calling `--compact` better.

## Implementation notes

- **Calls Ollama directly over HTTP, not through `typesafe-sdk`.** The SDK has `TypeSafeClient(base_url=..., model=...)`, but its docs don't say how `base_url` maps onto Ollama's `/v1/systemone` path. Calling the endpoint directly needs no packages other than pygame. To use the SDK, replace the `JevClient` class in `snake_jev.py`.
- **pygame is imported only when the window opens**, so `--headless`, `--dump` and the tests work without it.

## Files

| File | Contents |
|---|---|
| `snake_jev.py` | Game logic, query building, Ollama client, guard, PyGame and headless front ends |
| `test_snake_jev.py` | Unit tests for game rules, request shape, guard, lookahead and timing |
| `requirements.txt` | `pygame-ce` |

## References

- [Ollama: Jev-style decision models](https://ollama.com/blog/ollama-now-supports-jev-style-decision-models)
- [TypeSafe](https://typesafe.ai/)
- [TypeSafe docs: state](https://docs.typesafe.ai/concepts/state.md)
- [TypeSafe Python sync client](https://docs.typesafe.ai/sdk/python/api/clients/sync.md)
